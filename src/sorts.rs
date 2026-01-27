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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");

    let mut intervals = Vec::with_capacity(chrs.len());
    match sort_reverse_direction {
        Some(reverse) => {
            assert_eq!(chrs.len(), reverse.len(), "chrs and sort_reverse_direction must have same length");

            for (i, (((&chr, &start), &end), &rev)) in chrs.iter()
                .zip(starts)
                .zip(ends)
                .zip(reverse)
                .enumerate()
            {
                intervals.push(Interval {
                    group: chr,
                    start: if rev { -(start - slack) } else { start - slack },
                    end: if rev { -(end + slack) } else { end + slack },
                    idx: i as u32,
                });
            }
        }
        None => {
            for (i, ((&chr, &start), &end)) in chrs.iter()
                .zip(starts)
                .zip(ends)
                .enumerate()
            {
                intervals.push(Interval {
                    group: chr,
                    start: start - slack,
                    end: end + slack,
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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");
    assert_eq!(chrs.len(), strand_flags.len(), "chrs and strand_flags must have same length");

    let mut intervals = Vec::with_capacity(chrs.len());
    for (i, (((&chr, &start), &end), &forward)) in chrs.iter()
        .zip(starts)
        .zip(ends)
        .zip(strand_flags)
        .enumerate()
    {
        intervals.push(SplicedSubsequenceInterval {
            chr: chr,
            // Make start and end negative for intervals on negative strand
            // so they get sorted in the correct direction.
            // Later on the absolute value will be used.
            start: if forward { start } else { -start },
            end: if forward { end } else { -end },
            idx: i as u32,
            forward_strand: forward,
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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");
    assert_eq!(chrs.len(), idxs.len(), "chrs and idxs must have same length");
    assert_eq!(chrs.len(), strand_flags.len(), "chrs and strand_flags must have same length");

    let mut intervals: Vec<SubsequenceInterval> = Vec::with_capacity(chrs.len());
    for ((((&chr, &start), &end), &idx), &forward) in chrs.iter()
        .zip(starts)
        .zip(ends)
        .zip(idxs)
        .zip(strand_flags)
    {
        intervals.push(SubsequenceInterval {
            group_id: chr,
            // Make start and end negative for intervals on negative strand
            // so they get sorted in the correct direction.
            // Later on the absolute value will be used.
            start: if force_plus_strand || forward { start } else { -start },
            end: if force_plus_strand || forward { end } else { -end },
            idx: idx,
            forward_strand: forward,
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
    assert_eq!(chrs.len(), pos.len(), "chrs and pos must have same length");

    let mut events = Vec::with_capacity(chrs.len());

    // Convert set1 intervals into events
    for (i, (&chr, &p)) in chrs.iter().zip(pos).enumerate() {
        let position = if start { p - slack } else { p + slack };
        events.push(Event {
            chr: chr,
            pos: if negative_position { -position } else { position },
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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");

    let mut events = Vec::with_capacity(2 * chrs.len());

    // Convert set1 intervals into events
    for (i, ((&chr, &start), &end)) in chrs.iter().zip(starts).zip(ends).enumerate() {
        events.push(Event {
            chr: chr,
            pos: start,
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(Event {
            chr: chr,
            pos: end + slack,
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
    assert_eq!(chrs.len(), pos.len(), "chrs and pos must have same length");

    let mut out_pos: Vec<MinEvent<C, T>> = Vec::with_capacity(chrs.len());

    // Convert set1 intervals into events
    for (i, (&chr, &p)) in chrs.iter().zip(pos).enumerate() {
        out_pos.push(MinEvent {
            chr: chr,
            pos: p - slack,
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
    let mut out: Vec<GroupStruct<C>> = chrs.iter()
        .enumerate()
        .map(|(i, &chr)| GroupStruct { chr, idx: i as u32 })
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
    assert_eq!(chrs.len(), pos.len(), "chrs and pos must have same length");

    let mut out_pos = Vec::with_capacity(chrs.len());

    // Convert set1 intervals into events
    for (i, (&chr, &p)) in chrs.iter().zip(pos).enumerate() {
        out_pos.push(MinEvent {
            chr: chr,
            pos: p - slack,
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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");

    assert_eq!(chrs2.len(), starts2.len(), "chrs2 and starts2 must have same length");
    assert_eq!(chrs2.len(), ends2.len(), "chrs2 and ends2 must have same length");

    let mut events = Vec::with_capacity(2 * (chrs.len() + chrs2.len()));

    // Convert set1 intervals into events
    for (i, ((&chr, &start), &end)) in chrs.iter().zip(starts).zip(ends).enumerate() {
        events.push(GenericEvent {
            chr: chr,
            pos: if slack < start { start - slack } else { T::zero() },
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(GenericEvent {
            chr: chr,
            pos: end.saturating_add(slack),
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    for (j, ((&chr2, &start2), &end2)) in chrs2.iter().zip(starts2).zip(ends2).enumerate() {
        events.push(GenericEvent {
            chr: chr2,
            pos: start2,
            is_start: true,
            first_set: false,
            idx: j as u32,
        });
        events.push(GenericEvent {
            chr: chr2,
            pos: end2,
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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");

    assert_eq!(chrs2.len(), starts2.len(), "chrs2 and starts2 must have same length");
    assert_eq!(chrs2.len(), ends2.len(), "chrs2 and ends2 must have same length");

    let mut events = Vec::with_capacity(2 * (chrs.len() + chrs2.len()));

    // Convert set1 intervals into events
    for (i, ((&chr, &start), &end)) in chrs.iter().zip(starts).zip(ends).enumerate() {
        let start_slack = start - slack;
        let end_slack = end + slack;

        events.push(MaxEvent {
            chr: chr,
            pos: start_slack,
            start: start_slack,
            end: end_slack,
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(MaxEvent {
            chr: chr,
            pos: end_slack,
            end: end_slack,
            start: start_slack,
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    for (i, ((&chr2, &start2), &end2)) in chrs2.iter().zip(starts2).zip(ends2).enumerate() {
        events.push(MaxEvent {
            chr: chr2,
            pos: start2,
            start: start2,
            end: end2,
            is_start: true,
            first_set: false,
            idx: i as u32,
        });
        events.push(MaxEvent {
            chr: chr2,
            pos: end2,
            start: start2,
            end: end2,
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
    assert_eq!(chrs.len(), starts.len(), "chrs and starts must have same length");
    assert_eq!(chrs.len(), ends.len(), "chrs and ends must have same length");

    assert_eq!(chrs2.len(), starts2.len(), "chrs2 and starts2 must have same length");
    assert_eq!(chrs2.len(), ends2.len(), "chrs2 and ends2 must have same length");

    let mut events = Vec::with_capacity(2 * (chrs.len() + chrs2.len()));

    // Convert set1 intervals into events
    for (i, ((&chr, &start), &end)) in chrs.iter().zip(starts).zip(ends).enumerate() {
        events.push(Event {
            chr: chr,
            pos: start - slack,
            is_start: true,
            first_set: true,
            idx: i as u32,
        });
        events.push(Event {
            chr: chr,
            pos: end + slack,
            is_start: false,
            first_set: true,
            idx: i as u32,
        });
    }

    for (j, ((&chr2, &start2), &end2)) in chrs2.iter().zip(starts2).zip(ends2).enumerate() {
        events.push(Event {
            chr: chr2,
            pos: start2,
            is_start: true,
            first_set: false,
            idx: j as u32,
        });
        events.push(Event {
            chr: chr2,
            pos: end2,
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
