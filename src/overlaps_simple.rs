#![allow(dead_code)]

use core::fmt::Debug;
use core::ops::Add;
use std::str::FromStr;

use crate::ruranges_structs::{GroupType, OverlapType, PositionType};

#[inline(always)]
fn overlaps_with_slack<T: PositionType>(a_start: T, a_end: T, b_start: T, b_end: T, slack: T) -> bool {
    a_start < (b_end + slack) && b_start < (a_end + slack)
}

#[inline(always)]
fn contains_with_slack<T: PositionType>(outer_start: T, outer_end: T, inner_start: T, inner_end: T, slack: T) -> bool {
    outer_start <= (inner_start + slack) && inner_end <= (outer_end + slack)
}

#[inline(always)]
fn assert_sorted_by_group_then_start<C: GroupType, T: PositionType>(
    grp: &[C],
    start: &[T],
    end: &[T],
    label: &str,
) {
    debug_assert_eq!(grp.len(), start.len(), "{label}: grp/start length mismatch");
    debug_assert_eq!(grp.len(), end.len(), "{label}: grp/end length mismatch");

    if grp.is_empty() {
        return;
    }

    for i in 1..grp.len() {
        let g_prev = grp[i - 1];
        let g_cur = grp[i];
        if g_cur == g_prev {
            let s_prev = start[i - 1];
            let s_cur = start[i];
            if s_cur < s_prev {
                panic!(
                    "{label}: not sorted by start within group at index {i}: grp={:?} prev_start={:?} cur_start={:?}",
                    g_cur, s_prev, s_cur
                );
            }
        }
    }
}

pub fn sweep_line_overlaps<C: GroupType, T: PositionType>(
    grp1: &[C],
    start1: &[T],
    end1: &[T],
    grp2: &[C],
    start2: &[T],
    end2: &[T],
    slack: T,
    overlap_type: &str,
    contained: bool,
    no_checks: bool,
) -> (Vec<usize>, Vec<usize>) {
    let multiple = OverlapType::from_str(overlap_type)
        .expect("invalid overlap_type string");

    if !no_checks {
        assert_sorted_by_group_then_start(grp1, start1, end1, "Left collection");
        assert_sorted_by_group_then_start(grp2, start2, end2, "Right collection");
    }
    let n1 = grp1.len();
    let n2 = grp2.len();

    let mut out1: Vec<usize> = Vec::new();
    let mut out2: Vec<usize> = Vec::new();

    // Pointer into right collection (by group).
    let mut j: usize = 0;

    // Active set of right indices for current group.
    let mut active: Vec<usize> = Vec::new();
    let mut active_head: usize = 0; // logical head (retired items remain until occasional compaction)

    #[inline(always)]
    fn clear_active(active: &mut Vec<usize>, active_head: &mut usize) {
        active.clear();
        *active_head = 0;
    }

    let mut i: usize = 0;
    while i < n1 && j < n2 {
        // Align groups
        let g1 = grp1[i];
        let g2 = grp2[j];

        if g1 < g2 {
            let gg = g1;
            while i < n1 && grp1[i] == gg {
                i += 1;
            }
            continue;
        } else if g2 < g1 {
            let gg = g2;
            while j < n2 && grp2[j] == gg {
                j += 1;
            }
            continue;
        }

        // Groups equal: process this group chunk.
        let grp = g1;

        // Group ranges [i0, i1) and [j0, j1).
        let i0 = i;
        while i < n1 && grp1[i] == grp {
            i += 1;
        }
        let i1 = i;

        let j0 = j;
        while j < n2 && grp2[j] == grp {
            j += 1;
        }
        let j1 = j;

        // Reset sweep state for this group.
        clear_active(&mut active, &mut active_head);

        // Right pointer within this group.
        let mut jr = j0;

        // Sweep left intervals in this group.
        for il in i0..i1 {
            let a_start = start1[il];
            let a_end = end1[il];

            // Add to active: all right intervals whose start < a_end + slack.
            let a_end_slack = a_end + slack;
            while jr < j1 && start2[jr] < a_end_slack {
                active.push(jr);
                jr += 1;
            }

            // Retire: any right interval that is certainly too far left (end + slack <= a_start).
            while active_head < active.len() {
                let k = active[active_head];
                if (end2[k] + slack) <= a_start {
                    active_head += 1;
                } else {
                    break;
                }
            }

            // Occasional compaction (cheap amortized).
            if active_head > 0 && active_head * 2 >= active.len() {
                active.drain(0..active_head);
                active_head = 0;
            }

            match multiple {
                OverlapType::All => {
                    for idx in active_head..active.len() {
                        let r = active[idx];
                        let b_start = start2[r];
                        let b_end = end2[r];

                        if !overlaps_with_slack(a_start, a_end, b_start, b_end, slack) {
                            continue;
                        }
                        if contained && !contained_either_direction(a_start, a_end, b_start, b_end, slack) {
                            continue;
                        }

                        out1.push(il);
                        out2.push(r);
                    }
                }
                OverlapType::First => {
                    for idx in active_head..active.len() {
                        let r = active[idx];
                        let b_start = start2[r];
                        let b_end = end2[r];

                        if !overlaps_with_slack(a_start, a_end, b_start, b_end, slack) {
                            continue;
                        }
                        if contained && !contained_either_direction(a_start, a_end, b_start, b_end, slack) {
                            continue;
                        }

                        out1.push(il);
                        out2.push(r);
                        break;
                    }
                }
                OverlapType::Last => {
                    let mut last_r: Option<usize> = None;

                    for idx in active_head..active.len() {
                        let r = active[idx];
                        let b_start = start2[r];
                        let b_end = end2[r];

                        if !overlaps_with_slack(a_start, a_end, b_start, b_end, slack) {
                            continue;
                        }
                        if contained && !contained_either_direction(a_start, a_end, b_start, b_end, slack) {
                            continue;
                        }

                        last_r = Some(r);
                    }

                    if let Some(r) = last_r {
                        out1.push(il);
                        out2.push(r);
                    }
                }
            }
        }
    }

    (out1, out2)
}

#[inline(always)]
fn contained_either_direction<T: PositionType>(a_start: T, a_end: T, b_start: T, b_end: T, slack: T) -> bool {
    // Default interpretation: keep if A contains B OR B contains A (with slack).
    contains_with_slack(a_start, a_end, b_start, b_end, slack)
        || contains_with_slack(b_start, b_end, a_start, a_end, slack)
}
