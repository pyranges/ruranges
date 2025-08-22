use std::str::FromStr;
use std::time::{Duration, Instant};

use radsort::sort_by_key;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::helpers::{keep_first_by_idx, keep_last_by_idx};
use crate::ruranges_structs::{GroupType, MaxEvent, MinEvent, OverlapPair, OverlapType, PositionType};
use crate::sorts::{
    self, build_sorted_events_single_collection_separate_outputs, build_sorted_maxevents_with_starts_ends
};

/// Perform a four-way merge sweep to find cross overlaps.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WhichList {
    StartSet1,
    EndSet1,
    StartSet2,
    EndSet2,
}

impl WhichList {
    #[inline]
    fn is_start(&self) -> bool {
        match self {
            WhichList::StartSet1 | WhichList::StartSet2 => true,
            WhichList::EndSet1 | WhichList::EndSet2 => false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn overlaps<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
    overlap_type: &str,
    sort_output: bool,
    contained: bool,
) -> (Vec<u32>, Vec<u32>) {
    let overlap_type = OverlapType::from_str(overlap_type)
        .expect("invalid overlap_type string");

    let mut pairs = if contained {
        let maxevents = compute_sorted_maxevents(
            chrs, starts, ends, chrs2, starts2, ends2, slack, false,
        );
        sweep_line_overlaps_containment(maxevents)
    } else {
        sweep_line_overlaps(chrs, starts, ends, chrs2, starts2, ends2, slack)
    };

    if sort_output || (overlap_type == OverlapType::First || overlap_type == OverlapType::Last) {
        sort_by_key(&mut pairs,|p| p.idx);
    }

    match overlap_type {
        OverlapType::All => {},
        OverlapType::First => keep_first_by_idx(&mut pairs),
        OverlapType::Last => keep_last_by_idx(&mut pairs),
    }

    pairs.into_iter().map(|pair| (pair.idx, pair.idx2)).unzip()
}

pub fn sweep_line_overlaps_set1<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> Vec<u32> {
    // We'll collect all cross overlaps here
    let mut overlaps = Vec::new();

    if chrs.is_empty() | chrs2.is_empty() {
        return overlaps;
    };

    let events = sorts::build_sorted_events(chrs, starts, ends, chrs2, starts2, ends2, slack);

    // Active sets
    let mut active1 = FxHashSet::default();
    let mut active2 = FxHashSet::default();

    let mut current_chr = events.first().unwrap().chr;

    // Process events in ascending order of position
    for e in events {
        if e.chr != current_chr {
            active1.clear();
            current_chr = e.chr;
        }

        if e.is_start {
            // Interval is starting
            if e.first_set {
                // Overlaps with all currently active intervals in set2
                for &_idx2 in active2.iter() {
                    overlaps.push(e.idx);
                }
                // Now add it to active1
                active1.insert(e.idx);
            } else {
                // Overlaps with all currently active intervals in set1
                for &idx1 in active1.iter() {
                    overlaps.push(idx1);
                }
                // Now add it to active2
                active2.insert(e.idx);
            }
        } else {
            // Interval is ending
            if e.first_set {
                active1.remove(&e.idx);
            } else {
                active2.remove(&e.idx);
            }
        }
    }

    overlaps
}

pub fn count_overlaps<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> Vec<u32> {
    // We'll collect all cross overlaps here
    let mut overlaps = vec![0; chrs.len()];

    if chrs.is_empty() | chrs2.is_empty() {
        return overlaps;
    };

    let events = sorts::build_sorted_events(chrs, starts, ends, chrs2, starts2, ends2, slack);

    // Active sets
    let mut active1 = FxHashSet::default();
    let mut active2 = FxHashSet::default();

    let mut current_chr = events.first().unwrap().chr;

    // Process events in ascending order of position
    for e in events {
        if e.chr != current_chr {
            active1.clear();
            current_chr = e.chr;
        }

        if e.is_start {
            // Interval is starting
            if e.first_set {
                // Overlaps with all currently active intervals in set2
                for &_idx2 in active2.iter() {
                    overlaps[e.idx as usize] += 1;
                }
                // Now add it to active1
                active1.insert(e.idx);
            } else {
                // Overlaps with all currently active intervals in set1
                for &idx1 in active1.iter() {
                    overlaps[idx1 as usize] += 1;
                }
                // Now add it to active2
                active2.insert(e.idx);
            }
        } else {
            // Interval is ending
            if e.first_set {
                active1.remove(&e.idx);
            } else {
                active2.remove(&e.idx);
            }
        }
    }

    overlaps
}

pub fn sweep_line_overlaps_overlap_pair<C: GroupType, T: PositionType>(
    sorted_starts: &[MinEvent<C, T>],  // set 1 starts
    sorted_ends: &[MinEvent<C, T>],    // set 1 ends
    sorted_starts2: &[MinEvent<C, T>], // set 2 starts
    sorted_ends2: &[MinEvent<C, T>],   // set 2 ends
) -> Vec<OverlapPair> {
    let mut out_idxs = Vec::new();
    // Quick check: if no starts exist in either set, no overlaps.
    if sorted_starts.is_empty() || sorted_starts2.is_empty() {
        return out_idxs;
    }
    // Active intervals for set1, set2
    let mut active1 = FxHashSet::default();
    let mut active2 = FxHashSet::default();
    // Pointers into each list
    let mut i1 = 0usize; // pointer into sorted_starts  (set 1)
    let mut i2 = 0usize; // pointer into sorted_starts2 (set 2)
    let mut i3 = 0usize; // pointer into sorted_ends    (set 1)
    let mut i4 = 0usize; // pointer into sorted_ends2   (set 2)
                         // Figure out the very first chromosome we encounter (if any):
                         // We'll look at the heads of each list and pick the lexicographically smallest.
    let first_candidate = pick_winner_of_four(
        sorted_starts.get(i1).map(|e| (WhichList::StartSet1, e)),
        sorted_starts2.get(i2).map(|e| (WhichList::StartSet2, e)),
        sorted_ends.get(i3).map(|e| (WhichList::EndSet1, e)),
        sorted_ends2.get(i4).map(|e| (WhichList::EndSet2, e)),
    );
    // Unwrap the first candidate’s chromosome
    let mut current_chr = first_candidate.unwrap().1.chr;
    // Main sweep-line loop
    while i1 < sorted_starts.len()
        || i2 < sorted_starts2.len()
        || i3 < sorted_ends.len()
        || i4 < sorted_ends2.len()
    {
        let (which_list, event) = if let Some((which_list, event)) = pick_winner_of_four(
            sorted_starts.get(i1).map(|e| (WhichList::StartSet1, e)),
            sorted_starts2.get(i2).map(|e| (WhichList::StartSet2, e)),
            sorted_ends.get(i3).map(|e| (WhichList::EndSet1, e)),
            sorted_ends2.get(i4).map(|e| (WhichList::EndSet2, e)),
        ) {
            (which_list, event)
        } else {
            break;
        };
        // If we've moved to a new chromosome, reset active sets
        if event.chr != current_chr {
            active1.clear();
            active2.clear();
            current_chr = event.chr;
        }
        // Advance the pointer for whichever list we took an event from
        match which_list {
            WhichList::StartSet1 => {
                for &idx2 in active2.iter() {
                    out_idxs.push(OverlapPair {
                        idx: event.idx,
                        idx2: idx2,
                    })
                }
                // Now add it to active1
                active1.insert(event.idx);
                i1 += 1
            }
            WhichList::StartSet2 => {
                for &idx1 in active1.iter() {
                    out_idxs.push(OverlapPair {
                        idx: idx1,
                        idx2: event.idx,
                    })
                }
                // Now add it to active2
                active2.insert(event.idx);
                i2 += 1
            }
            WhichList::EndSet1 => {
                active1.remove(&event.idx);
                i3 += 1
            }
            WhichList::EndSet2 => {
                active2.remove(&event.idx);
                i4 += 1
            }
        }
    }
    out_idxs
}

pub fn sweep_line_overlaps_containment<C: GroupType, T: PositionType>(
    events: Vec<MaxEvent<C, T>>,
) -> (Vec<OverlapPair>) {
    // We'll collect all cross overlaps here
    let mut overlaps = Vec::new();

    if events.is_empty() {
        return overlaps;
    };

    // Active sets
    let mut active1 = FxHashMap::default();
    let mut active2 = FxHashMap::default();

    let mut current_chr = events.first().unwrap().chr;

    // Process events in ascending order of position
    for e in events {
        if e.chr != current_chr {
            active1.clear();
            active2.clear();
            current_chr = e.chr;
        }

        if e.is_start {
            // Interval is starting
            if e.first_set {
                // Overlaps with all currently active intervals in set2
                for (&idx2, &(start2, end2)) in active2.iter() {
                    if e.start >= start2 && e.end <= end2 {
                        overlaps.push(OverlapPair {
                            idx: e.idx,
                            idx2: idx2,
                        });
                    };
                }
                // Now add it to active1
                active1.insert(e.idx, (e.start, e.end));
            } else {
                // Overlaps with all currently active intervals in set1
                for (&idx, &(start, end)) in active1.iter() {
                    if e.start <= start && e.end >= end {
                        overlaps.push(OverlapPair {
                            idx: idx,
                            idx2: e.idx,
                        });
                    };
                }
                // Now add it to active2
                active2.insert(e.idx, (e.start, e.end));
            }
        } else {
            // Interval is ending
            if e.first_set {
                active1.remove(&e.idx);
            } else {
                active2.remove(&e.idx);
            }
        }
    }

    overlaps
}

fn pick_winner_of_four<'a, C: GroupType, T: PositionType>(
    s1: Option<(WhichList, &'a MinEvent<C, T>)>,
    s2: Option<(WhichList, &'a MinEvent<C, T>)>,
    e1: Option<(WhichList, &'a MinEvent<C, T>)>,
    e2: Option<(WhichList, &'a MinEvent<C, T>)>,
) -> Option<(WhichList, &'a MinEvent<C, T>)> {
    let starts_winner = pick_winner_of_two_choose_first_if_equal(s1, e1);
    let ends_winner = pick_winner_of_two_choose_first_if_equal(s2, e2);
    pick_winner_of_two_choose_first_if_equal(starts_winner, ends_winner)
}

fn pick_winner_of_two_choose_first_if_equal<'a, C: GroupType, T: PositionType>(
    a: Option<(WhichList, &'a MinEvent<C, T>)>,
    b: Option<(WhichList, &'a MinEvent<C, T>)>,
) -> Option<(WhichList, &'a MinEvent<C, T>)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some((wh_a, ev_a)), Some((wh_b, ev_b))) => {
            // Compare by chromosome
            if ev_a.chr < ev_b.chr {
                return Some((wh_a, ev_a));
            } else if ev_b.chr < ev_a.chr {
                return Some((wh_b, ev_b));
            }
            // Same chr => compare by pos
            if ev_a.pos < ev_b.pos {
                return Some((wh_a, ev_a));
            } else if ev_b.pos < ev_a.pos {
                return Some((wh_b, ev_b));
            }
            // Same (chr, pos) => tie break: end < start
            let a_is_end = !wh_a.is_start();
            let b_is_end = !wh_b.is_start();
            match (a_is_end, b_is_end) {
                // If both are ends or both are starts, just pick either. We'll pick `a`.
                (true, true) | (false, false) => Some((wh_a, ev_a)),
                // If only one is end, that one is “smaller”
                (true, false) => Some((wh_a, ev_a)),
                (false, true) => Some((wh_b, ev_b)),
            }
        }
    }
}

pub fn compute_sorted_events<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    slack: T,
    invert: bool,
) -> (Vec<MinEvent<C, T>>, Vec<MinEvent<C, T>>) {
    if !invert {
        // "Normal" path
        let sorted_starts =
            build_sorted_events_single_collection_separate_outputs(chrs, starts, slack);
        let sorted_ends =
            build_sorted_events_single_collection_separate_outputs(chrs, ends, T::zero());
        (sorted_starts, sorted_ends)
    } else {
        // "Inverted" path
        let new_starts: Vec<_> = starts.iter().map(|&v| -v).collect();
        let new_ends: Vec<_> = ends.iter().map(|&v| -v).collect();

        let sorted_starts =
            build_sorted_events_single_collection_separate_outputs(chrs, &new_ends, slack);
        let sorted_ends =
            build_sorted_events_single_collection_separate_outputs(chrs, &new_starts, T::zero());
        (sorted_starts, sorted_ends)
    }
}

pub fn compute_sorted_maxevents<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
    invert: bool,
) -> Vec<MaxEvent<C, T>> {
    if !invert {
        // "Normal" path
        build_sorted_maxevents_with_starts_ends(chrs, starts, ends, chrs2, starts2, ends2, slack)
    } else {
        // "Inverted" path
        let new_starts_vec: Vec<T> = starts.iter().map(|&v| -v).collect();
        let new_starts = new_starts_vec.as_slice();
        let new_ends_vec: Vec<T> = ends.iter().map(|&v| -v).collect();
        let new_ends = new_ends_vec.as_slice();

        let new_starts_vec2: Vec<T> = starts2.iter().map(|&v| -v).collect();
        let new_starts2 = new_starts_vec2.as_slice();
        let new_ends_vec2: Vec<T> = ends2.iter().map(|&v| -v).collect();
        let new_ends2 = new_ends_vec2.as_slice();
        build_sorted_maxevents_with_starts_ends(
            chrs,
            new_ends,
            new_starts,
            chrs2,
            new_ends2,
            new_starts2,
            slack,
        )
    }
}


pub fn sweep_line_overlaps<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> (Vec<OverlapPair>) {
    // We'll collect all cross overlaps here
    let mut overlaps = Vec::new();

    let events = sorts::build_sorted_events(chrs, starts, ends, chrs2, starts2, ends2, slack);

    if events.is_empty() {
        return overlaps;
    };

    // Active sets
    let mut active1 = FxHashSet::default();
    let mut active2 =FxHashSet::default();

    let mut current_chr = events.first().unwrap().chr;

    // Process events in ascending order of position
    for e in events {
        if e.chr != current_chr {
            active1.clear();
            active2.clear();
            current_chr = e.chr;
        }

        if e.is_start {
            // Interval is starting
            if e.first_set {
                // Overlaps with all currently active intervals in set2
                for &idx2 in active2.iter() {
                    overlaps.push(OverlapPair {
                        idx: e.idx,
                        idx2: idx2,
                    });
                }
                // Now add it to active1
                active1.insert(e.idx);
            } else {
                // Overlaps with all currently active intervals in set1
                for &idx in active1.iter() {
                        overlaps.push(OverlapPair {
                            idx: idx,
                            idx2: e.idx,
                        });
                    };
                active2.insert(e.idx);
            }
        } else {
            // Interval is ending
            if e.first_set {
                active1.remove(&e.idx);
            } else {
                active2.remove(&e.idx);
            }
        }
    }

    overlaps
}