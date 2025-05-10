use crate::{ruranges_structs::{GroupType, PositionType}, sorts};

use rustc_hash::FxHashSet;

pub fn sweep_line_non_overlaps<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    chrs2: &[G],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> Vec<u32> {
    let mut no_overlaps = Vec::new();

    // If either set is empty, none can overlap; return everything as “non-overlapping”.
    if chrs.is_empty() || chrs2.is_empty() {
        // Just return all indices as non-overlapping
        return no_overlaps.to_vec();
    }

    // Build up the event list in ascending order (same as before)
    let events = sorts::build_sorted_events_idxs(chrs, starts, ends, chrs2, starts2, ends2, slack);

    let mut overlapped = FxHashSet::default();

    // Active sets
    let mut active1 = FxHashSet::default();
    let mut active2 = FxHashSet::default();

    // Assume the first event determines the “current” chr
    let mut current_chr = events.first().unwrap().chr;

    for e in events {
        // If chromosome changed, clear active sets
        if e.chr != current_chr {
            active1.clear();
            active2.clear();
            current_chr = e.chr;
        }

        if e.is_start {
            // Interval is starting
            if e.first_set {
                // Overlaps with all currently active intervals in set2
                if !active2.is_empty() {
                    overlapped.insert(e.idx);
                }
                // Insert into active1
                active1.insert(e.idx);
            } else {
                // Overlaps with all currently active intervals in set1
                for &idx1 in active1.iter() {
                    overlapped.insert(idx1);
                }
                // Insert into active2
                active2.insert(e.idx);
            }
        } else {
            // Interval is ending
            if e.first_set {
                active1.remove(&e.idx);
                if !overlapped.contains(&e.idx) {
                    no_overlaps.push(e.idx);
                }
            } else {
                active2.remove(&e.idx);
            }

            overlapped.remove(&e.idx);
        }
    }

    radsort::sort(&mut no_overlaps);
    no_overlaps
}
