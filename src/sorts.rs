use radsort::sort_by_key;

use crate::ruranges_structs::Event;
use crate::ruranges_structs::GenericEvent;
use crate::ruranges_structs::GroupStruct;
use crate::ruranges_structs::GroupType;
use crate::ruranges_structs::Interval;
use crate::ruranges_structs::MaxEvent;
use crate::ruranges_structs::MinEvent;
use crate::ruranges_structs::PositionType;
use crate::ruranges_structs::SplicedSubsequenceInterval;
use crate::ruranges_structs::SubsequenceInterval;

pub fn build_intervals<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    sort_reverse_direction: Option<&[bool]>,
    slack: T,
) -> Vec<Interval<C, T>> {
    let mut intervals = Vec::with_capacity(chrs.len());
    match sort_reverse_direction {
        Some(reverse) => {
            for i in 0..chrs.len() {
                intervals.push(Interval {
                    group: chrs[i],
                    start: if reverse[i] {
                        -(starts[i] - slack)
                    } else {
                        starts[i] - slack
                    },
                    end: if reverse[i] {
                        -(ends[i] + slack)
                    } else {
                        ends[i] + slack
                    },
                    idx: i as u32,
                });
            }
        }
        None => {
            for i in 0..chrs.len() {
                intervals.push(Interval {
                    group: chrs[i],
                    start: starts[i] - slack,
                    end: ends[i] + slack,
                    idx: i as u32,
                });
            }
        }
    };

    intervals
}

pub fn build_subsequence_intervals<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
) -> Vec<SplicedSubsequenceInterval<G, T>> {
    let mut intervals = Vec::with_capacity(chrs.len());
    for i in 0..chrs.len() {
        intervals.push(SplicedSubsequenceInterval {
            chr: chrs[i],
            start: if strand_flags[i] {
                starts[i]
            } else {
                -starts[i]
            }, // so that negative strand intervals are sorted in the correct direction
            end: if strand_flags[i] { ends[i] } else { -ends[i] }, // we will find the absolute value when using them
            idx: i as u32,
            forward_strand: strand_flags[i],
            temp_cumsum: T::zero(),
            temp_length: T::zero(),
        });
    }

    intervals
}

pub fn build_sequence_intervals(
    chrs: &[i64],
    starts: &[i64],
    ends: &[i64],
    idxs: &[i64],
    strand_flags: &[bool],
    force_plus_strand: bool,
) -> Vec<SubsequenceInterval> {
    let mut intervals: Vec<SubsequenceInterval> = Vec::with_capacity(chrs.len());
    for i in 0..chrs.len() {
        intervals.push(SubsequenceInterval {
            group_id: chrs[i],
            start: if force_plus_strand || strand_flags[i] {
                starts[i]
            } else {
                -starts[i]
            }, // so that negative strand intervals are sorted in the correct direction
            end: if force_plus_strand || strand_flags[i] {
                ends[i]
            } else {
                -ends[i]
            }, // we will find the absolute value when using them
            idx: idxs[i],
            forward_strand: strand_flags[i],
        });
    }

    intervals
}

pub fn build_sorted_intervals<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    sort_reverse_direction: Option<&[bool]>,
    slack: T,
    sort_on_ends_too: bool,
) -> Vec<Interval<C, T>> {
    let mut intervals = build_intervals(chrs, starts, ends, sort_reverse_direction, slack);

    if sort_on_ends_too {
        sort_by_key(&mut intervals, |i| i.end);
    };
    sort_by_key(&mut intervals, |i| i.start);
    sort_by_key(&mut intervals, |i| i.group);

    intervals
}

pub fn build_sorted_subsequence_intervals<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
) -> Vec<SplicedSubsequenceInterval<G, T>> {
    let mut intervals = build_subsequence_intervals(chrs, starts, ends, strand_flags);

    sort_by_key(&mut intervals, |i| i.end);
    sort_by_key(&mut intervals, |i| i.start);
    sort_by_key(&mut intervals, |i| i.chr);

    intervals
}

pub fn build_sorted_sequence_intervals(
    chrs: &[i64],
    starts: &[i64],
    ends: &[i64],
    idxs: &[i64],
    strand_flags: &[bool],
    force_plus_strand: bool,
) -> Vec<SubsequenceInterval> {
    let mut intervals =
        build_sequence_intervals(chrs, starts, ends, idxs, strand_flags, force_plus_strand);

    sort_by_key(&mut intervals, |i| i.end);
    sort_by_key(&mut intervals, |i| i.start);
    sort_by_key(&mut intervals, |i| i.group_id);

    intervals
}

pub fn sort_order_idx<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    sort_reverse_direction: Option<&[bool]>,
) -> Vec<u32> {
    build_sorted_intervals(chrs, starts, ends, sort_reverse_direction, T::zero(), true)
        .iter()
        .map(|i| i.idx)
        .collect()
}

pub fn build_sorted_events_single_position<C: GroupType, T: PositionType>(
    chrs: &[C],
    pos: &[T],
    start: bool,
    first_set: bool,
    negative_position: bool,
    slack: T,
) -> Vec<Event<C, T>> {
    let mut events = Vec::with_capacity(2 * (chrs.len()));

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        let pos = if start {
            pos[i] - slack
        } else {
            pos[i] + slack
        };
        events.push(Event {
            chr: chrs[i],
            pos: if negative_position { -pos } else { pos },
            is_start: start,
            first_set: first_set,
            idx: i as u32,
        });
    }

    sort_by_key(&mut events, |e| (e.chr, e.pos, e.is_start));

    events
}

pub fn build_sorted_events_single_collection<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    slack: T,
) -> Vec<Event<C, T>> {
    let mut events = Vec::with_capacity(2 * (chrs.len()));

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        events.push(Event {
            chr: chrs[i],
            pos: starts[i],
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(Event {
            chr: chrs[i],
            pos: ends[i] + slack,
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    // Sort events by:
    // 1. pos (ascending)
    // 2. is_start before is_end (if pos ties)
    // (We don't strictly need to tie-break by set_id or idx, but we can.)

    sort_by_key(&mut events, |e| e.is_start);
    sort_by_key(&mut events, |e| e.pos);
    sort_by_key(&mut events, |e| e.chr);

    events
}

pub fn build_sorted_events_single_collection_separate_outputs<C: GroupType, T: PositionType>(
    chrs: &[C],
    pos: &[T],
    slack: T,
) -> Vec<MinEvent<C, T>> {
    let mut out_pos: Vec<MinEvent<C, T>> = Vec::with_capacity(chrs.len());

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        out_pos.push(MinEvent {
            chr: chrs[i],
            pos: pos[i] - slack,
            idx: i as u32,
        });
    }

    sort_by_key(&mut out_pos, |e| e.pos);
    sort_by_key(&mut out_pos, |e| e.chr);

    out_pos
}

pub fn build_sorted_groups<C: GroupType>(
    chrs: &[C],
) -> Vec<u32> {
    let mut out: Vec<GroupStruct<C>> = (0..chrs.len())
    .map(|i| GroupStruct { chr: chrs[i], idx: i as u32 })
    .collect();

    out.sort_by_key(|e| e.chr);

    // take the chromosome field, cast to u32, collect -----------------------
    out.into_iter().map(|e| e.idx).collect()
}

pub fn build_sorted_events_with_starts_ends<C: GroupType, T: PositionType>(
    chrs: &[C],
    pos: &[T],
    slack: T,
) -> Vec<MinEvent<C, T>> {
    let mut out_pos = Vec::with_capacity(chrs.len());

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        out_pos.push(MinEvent {
            chr: chrs[i],
            pos: pos[i] - slack,
            idx: i as u32,
        });
    }

    sort_by_key(&mut out_pos, |e| e.pos);
    sort_by_key(&mut out_pos, |e| e.chr);

    out_pos
}

pub fn build_sorted_events<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> Vec<GenericEvent<C, T>> {
    let mut events = Vec::with_capacity(2 * (chrs.len() + chrs2.len()));

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        events.push(GenericEvent {
            chr: chrs[i],
            pos: if slack < starts[i] {
                starts[i] - slack
            } else {
                T::zero()
            },
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(GenericEvent {
            chr: chrs[i],
            pos: ends[i].saturating_add(slack),
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    for j in 0..chrs2.len() {
        events.push(GenericEvent {
            chr: chrs2[j],
            pos: starts2[j],
            is_start: true,
            first_set: false,
            idx: j as u32,
        });
        events.push(GenericEvent {
            chr: chrs2[j],
            pos: ends2[j],
            is_start: false,
            first_set: false,
            idx: j as u32,
        });
    }

    sort_by_key(&mut events, |e| e.is_start);
    sort_by_key(&mut events, |e| e.pos);
    sort_by_key(&mut events, |e| e.chr);

    events
}

pub fn build_sorted_maxevents_with_starts_ends<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> Vec<MaxEvent<C, T>> {
    let mut events = Vec::with_capacity(2 * (chrs.len() + chrs2.len()));

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        events.push(MaxEvent {
            chr: chrs[i],
            pos: starts[i] - slack,
            start: starts[i] - slack,
            end: ends[i] + slack,
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(MaxEvent {
            chr: chrs[i],
            pos: ends[i] + slack,
            end: ends[i] + slack,
            start: starts[i] - slack,
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    for i in 0..chrs2.len() {
        events.push(MaxEvent {
            chr: chrs2[i],
            pos: starts2[i],
            start: starts2[i],
            end: ends2[i],
            is_start: true,
            first_set: false,
            idx: i as u32,
        });
        events.push(MaxEvent {
            chr: chrs2[i],
            pos: ends2[i],
            start: starts2[i],
            end: ends2[i],
            is_start: false,
            first_set: false,
            idx: i as u32,
        });
    }

    sort_by_key(&mut events, |e| e.is_start);
    sort_by_key(&mut events, |e| e.pos);
    sort_by_key(&mut events, |e| e.chr);

    events
}

pub fn build_sorted_events_idxs<C: GroupType, T: PositionType>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    chrs2: &[C],
    starts2: &[T],
    ends2: &[T],
    slack: T,
) -> Vec<Event<C, T>> {
    let mut events = Vec::with_capacity(2 * (chrs.len() + chrs2.len()));

    // Convert set1 intervals into events
    for i in 0..chrs.len() {
        events.push(Event {
            chr: chrs[i],
            pos: starts[i] - slack,
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(Event {
            chr: chrs[i],
            pos: ends[i] + slack,
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    for j in 0..chrs2.len() {
        events.push(Event {
            chr: chrs2[j],
            pos: starts2[j],
            is_start: true,
            first_set: false,
            idx: j as u32,
        });
        events.push(Event {
            chr: chrs2[j],
            pos: ends2[j],
            is_start: false,
            first_set: false,
            idx: j as u32,
        });
    }

    sort_by_key(&mut events, |e| e.is_start);
    sort_by_key(&mut events, |e| e.pos);
    sort_by_key(&mut events, |e| e.chr);

    events
}

pub fn build_sorted_events_from_intervals<C: GroupType, T: PositionType>(
    intervals1: &mut [Interval<C, T>],
    intervals2: &mut [Interval<C, T>],
) -> Vec<Event<C, T>> {
    let mut events = Vec::with_capacity(2 * (intervals1.len() + intervals2.len()));

    // Convert set1 intervals into events
    for interval in intervals1 {
        events.push(Event {
            chr: interval.group,
            pos: interval.start,
            is_start: true,
            first_set: true,
            idx: interval.idx,
        });
        events.push(Event {
            chr: interval.group,
            pos: interval.end,
            is_start: false,
            first_set: true,
            idx: interval.idx,
        });
    }

    for interval in intervals2 {
        events.push(Event {
            chr: interval.group,
            pos: interval.start,
            is_start: true,
            first_set: false,
            idx: interval.idx,
        });
        events.push(Event {
            chr: interval.group,
            pos: interval.end,
            is_start: false,
            first_set: false,
            idx: interval.idx,
        });
    }

    // Sort events by:
    // 1. pos (ascending)
    // 2. is_start before is_end (if pos ties)
    // (We don't strictly need to tie-break by set_id or idx, but we can.)
    sort_by_key(&mut events, |e| e.is_start);
    sort_by_key(&mut events, |e| e.pos);

    events
}
