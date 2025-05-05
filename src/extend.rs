use std::collections::HashMap;

use crate::ruranges_structs::{GroupType, PositionType};

fn check_ext_options<T: PositionType>(
    ext: Option<T>,
    ext_3: Option<T>,
    ext_5: Option<T>,
) -> Result<(), &'static str> {
    // The condition below is true when either both ext and (ext_3 or ext_5) are provided,
    // or when neither is provided.
    if ext.is_some() == (ext_3.is_some() || ext_5.is_some()) {
        Err("Must use at least one and not both of ext and ext3 or ext5.")
    } else {
        Ok(())
    }
}

/// Extend each interval `[starts[i], ends[i]]` either symmetrically (if `ext` is `Some`)
/// or via strand-dependent 5'/3' extensions (if `ext` is `None`).
///
/// `negative_strand[i] == true` indicates a reverse (negative) strand for the `i`th record.
///
/// Returns new `(starts, ends)` after extension. Panics if any interval ends up invalid.
pub fn extend<T: PositionType>(
    starts: &[T],
    ends: &[T],
    negative_strand: &[bool],
    ext: Option<T>,
    ext_3: Option<T>,
    ext_5: Option<T>,
) -> (Vec<T>, Vec<T>) {
    assert_eq!(starts.len(), ends.len());
    assert_eq!(starts.len(), negative_strand.len());
    assert!(check_ext_options(ext, ext_3, ext_5).is_ok());

    let n = starts.len();
    let mut new_starts = Vec::with_capacity(n);
    let mut new_ends = Vec::with_capacity(n);

    match ext {
        // Symmetrical extension
        Some(e) => {
            for i in 0..n {
                let mut s = starts[i] - e;
                if s < T::zero() {
                    s = T::zero(); // clamp to 0
                }
                let e_ = ends[i] + e;

                new_starts.push(s);
                new_ends.push(e_);
            }
        }
        // Strand-dependent extension
        None => {
            for i in 0..n {
                let mut s = starts[i];
                let mut e_ = ends[i];
                if negative_strand[i] {
                    // Reverse (negative) strand
                    if let Some(ext_5_val) = ext_5 {
                        // 5' extension on reverse strand => add to end
                        e_ = e_ + ext_5_val;
                    }
                    if let Some(ext_3_val) = ext_3 {
                        // 3' extension on reverse strand => subtract from start
                        s = s - ext_3_val;
                    }
                } else {
                    // Forward strand
                    if let Some(ext_5_val) = ext_5 {
                        // 5' extension on forward strand => subtract from start
                        s = s - ext_5_val;
                    }
                    if let Some(ext_3_val) = ext_3 {
                        // 3' extension on forward strand => add to end
                        e_ = e_ + ext_3_val;
                    }
                }
                new_starts.push(s);
                new_ends.push(e_);
            }
        }
    }

    (new_starts, new_ends)
}

/// Extend each group's intervals by modifying only the row with the minimal start
/// and the row with the maximal end for that group.
///
/// Returns `(group_ids, new_starts, new_ends)`.
pub fn extend_grp<G: GroupType, T: PositionType>(
    group_ids: &[G],
    starts: &[T],
    ends: &[T],
    negative_strand: &[bool],
    ext: Option<T>,
    ext_3: Option<T>,
    ext_5: Option<T>,
) -> (Vec<T>, Vec<T>) {
    assert_eq!(group_ids.len(), starts.len());
    assert_eq!(starts.len(), ends.len());
    assert_eq!(ends.len(), negative_strand.len());
    assert!(check_ext_options(ext, ext_3, ext_5).is_ok());

    let n = starts.len();

    // We'll create new vectors from the original starts/ends so we can modify them.
    let mut new_starts = starts.to_vec();
    let mut new_ends = ends.to_vec();

    // 1. Identify min-start-index and max-end-index for each group.
    // group_info: group_id -> (min_start_idx, min_start_val, max_end_idx, max_end_val)
    let mut group_info: HashMap<G, (usize, T, usize, T)> = HashMap::new();

    for i in 0..n {
        let g = group_ids[i];
        let s = starts[i];
        let e = ends[i];

        group_info
            .entry(g)
            .and_modify(|(min_i, min_s, max_i, max_e)| {
                // Update min-start
                if s < *min_s {
                    *min_i = i;
                    *min_s = s;
                }
                // Update max-end
                if e > *max_e {
                    *max_i = i;
                    *max_e = e;
                }
            })
            .or_insert((i, s, i, e));
    }

    // 2. For each group, apply either symmetric extension or strand-specific extension
    match ext {
        // Symmetrical extension
        Some(e) => {
            for (_group_id, (min_i, _min_s, max_i, _max_e)) in group_info.iter() {
                // subtract from the min-start index, clamp to zero
                let s = new_starts[*min_i] - e;
                new_starts[*min_i] = s.max(T::zero());

                // add to the max-end index
                new_ends[*max_i] = new_ends[*max_i] + e;
            }
        }
        // Strand-dependent extension
        None => {
            for (_group_id, (min_i, _min_s, max_i, _max_e)) in group_info.iter() {
                let is_reverse = negative_strand[*min_i];
                // The assumption here is that the entire group has the same strand.
                // If that's not guaranteed, you'd have to define which row's strand matters,
                // or store strand info per group, etc.

                if is_reverse {
                    // Reverse strand
                    if let Some(e5) = ext_5 {
                        // 5' on reverse => add to end, so do it on the max-end index
                        new_ends[*max_i] = new_ends[*max_i] + e5;
                    }
                    if let Some(e3) = ext_3 {
                        // 3' on reverse => subtract from start, so do it on the min-start index
                        new_starts[*min_i] = new_starts[*min_i] - e3;
                    }
                } else {
                    // Forward strand
                    if let Some(e5) = ext_5 {
                        // 5' on forward => subtract from start, so do it on min-start index
                        new_starts[*min_i] = new_starts[*min_i] - e5;
                    }
                    if let Some(e3) = ext_3 {
                        // 3' on forward => add to end, so do it on max-end index
                        new_ends[*max_i] = new_ends[*max_i] + e3;
                    }
                }
            }
        }
    }

    // Return group IDs, new starts, and new ends
    (new_starts, new_ends)
}
