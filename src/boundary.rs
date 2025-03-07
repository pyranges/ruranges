use std::time::Instant;

use crate::{ruranges_structs::PositionType, sorts};

pub fn sweep_line_boundary<T: PositionType>(
    chrs: &[u32],
    starts: &[T],
    ends: &[T],
) -> (Vec<u32>, Vec<T>, Vec<T>, Vec<u32>) {
    let mut out_indices = Vec::with_capacity(chrs.len());
    let mut out_starts = Vec::with_capacity(chrs.len());
    let mut out_ends = Vec::with_capacity(chrs.len());
    let mut counts = Vec::with_capacity(chrs.len());

    if chrs.is_empty() {
        return (out_indices, out_starts, out_ends, counts);
    };

    let events = sorts::build_sorted_events_single_collection(chrs, starts, ends, T::zero());

    let mut current_chr = events.first().unwrap().chr;
    let mut current_start = events.first().unwrap().pos;
    let final_idx = events.last().unwrap().idx;
    let final_end = events.last().unwrap().pos;
    let mut prev_pos = T::zero();
    let mut prev_idx = 0;
    let mut current_cluster_count = 0;

    for e in events {
        if e.chr != current_chr {
            current_cluster_count = 0;
            current_chr = e.chr;
            out_indices.push(prev_idx);
            out_starts.push(current_start);
            out_ends.push(prev_pos);
            counts.push(current_cluster_count);
            current_start = e.pos;
        }

        prev_pos = e.pos;
        prev_idx = e.idx;
        current_cluster_count += 1;
    }

    out_indices.push(final_idx);
    out_starts.push(current_start);
    out_ends.push(final_end);
    counts.push(current_cluster_count);

    (out_indices, out_starts, out_ends, counts)
}
