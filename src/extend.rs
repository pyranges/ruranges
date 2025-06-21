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


/// Extend each group's intervals by modifying only the row with the minimal start
/// and the row with the maximal end for that group.
///
/// Returns `(group_ids, new_starts, new_ends)`.
pub fn extend_grp<G: GroupType, T: PositionType>(
    group_ids:        &[G],
    starts:           &[T],
    ends:             &[T],
    negative_strand:  &[bool],
    ext_3:            T,
    ext_5:            T,
) -> (Vec<T>, Vec<T>) {
    /* ─── 0. Basic sanity ─────────────────────────────────────────────────── */
    assert_eq!(group_ids.len(), starts.len());
    assert_eq!(starts.len(),     ends.len());
    assert_eq!(ends.len(),       negative_strand.len());

    let n = starts.len();
    let mut new_start = starts.to_vec();
    let mut new_end   = ends.to_vec();

    let mut extrema: HashMap<G, (usize /*min_i*/, usize /*max_i*/)> =
        HashMap::with_capacity(n);

    for i in 0..n {
        extrema
            .entry(group_ids[i])
            .and_modify(|(min_i, max_i)| {
                if starts[i] < starts[*min_i] { *min_i = i; }
                if ends  [i] > ends  [*max_i] { *max_i = i; }
            })
            .or_insert((i, i));
    }

    for (_gid, (min_i, max_i)) in extrema {
        if negative_strand[min_i] {
            new_end  [max_i] = new_end  [max_i] + ext_5;
            new_start[min_i] = new_start[min_i] - ext_3;
        } else {
            new_start[min_i] = new_start[min_i] - ext_5;
            new_end  [max_i] = new_end  [max_i] + ext_3;
        }
    }

    (new_start, new_end)
}
