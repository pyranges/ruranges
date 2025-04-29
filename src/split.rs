use crate::{ruranges_structs::{GroupType, PositionType}, sorts};

pub fn sweep_line_split<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    slack: T,
    between: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>) {
    let events = sorts::build_sorted_events_single_collection(chrs, starts, ends, slack);

    // These will hold the output arrays: each emitted subinterval’s
    // (original_idx, start, end).
    let mut idxs_out = Vec::new();
    let mut starts_out = Vec::new();
    let mut ends_out = Vec::new();

    // Edge case: no intervals
    if events.is_empty() {
        return (idxs_out, starts_out, ends_out);
    }

    // State for the sweep line
    let mut current_chr = events[0].chr;
    // We initialize coverage to 0, then we will “process” each event,
    // but we need a “last_pos” to track from where we last emitted.
    let mut active_count: u32 = 0;
    let mut last_pos = events[0].pos;
    let mut last_idx = events[0].idx; // you can store whichever index you like

    // Decide whether coverage is “on” at the very first position:
    // If the first event is a start, coverage goes from 0 → 1 at that point.
    if events[0].is_start {
        active_count = 1;
    }

    // We iterate from the *second* event onward.
    // At each new event, we emit from last_pos → e.pos if either coverage was > 0 or `between = true`.
    for e_i in 1..events.len() {
        let e = &events[e_i];

        // If chromosome changes, we “jump” to a new chromosome
        // and do *not* produce an interval bridging old->new.
        if e.chr != current_chr {
            // reset
            current_chr = e.chr;
            active_count = if e.is_start { 1 } else { 0 };
            last_pos = e.pos;
            last_idx = e.idx;
            continue;
        }

        // same chromosome => we may emit from last_pos..e.pos if it's > 0 length
        // and either coverage>0 or we want the gap (between = true).
        if e.pos > last_pos {
            // If we were in coverage or want gaps, emit the subinterval.
            if active_count > 0 || between {
                idxs_out.push(last_idx);
                starts_out.push(last_pos);
                ends_out.push(e.pos);
            }
            last_pos = e.pos;
            last_idx = e.idx; // you might prefer to keep the same idx as “first covering interval”
        }

        // Now handle the event itself (this flips coverage up or down).
        if e.is_start {
            active_count += 1;
        } else {
            // is an end
            if active_count > 0 {
                active_count -= 1;
            }
        }
    }

    (idxs_out, starts_out, ends_out)
}
