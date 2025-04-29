use crate::{
    ruranges_structs::{GroupType, PositionType, SplicedSubsequenceInterval},
    sorts::build_sorted_subsequence_intervals,
};

/// Replicates the "spliced_subseq" logic in one pass for intervals sorted by (chrom, start, end).
///
/// - chrs: chromosome (encoded) array: actually group ids
/// - starts: genomic start coordinates
/// - ends: genomic end coordinates
/// - idxs: some index array
/// - strand_flags: array of bool indicating forward strand (true) or reverse (false)
/// - start: spliced start (can be negative, to count from the 3' end)
/// - end: spliced end (can be None = unlimited, or negative => 3' offset)
/// - force_plus_strand: if true, treat **all** intervals as if forward strand
///
/// Returns tuple of (out_idxs, out_starts, out_ends).
pub fn spliced_subseq<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
    start: T,
    end: Option<T>,
    force_plus_strand: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>) {
    // If no intervals, just return.
    if chrs.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Build the vector of intervals, which is already sorted by (chr, start, end) in your code.
    let intervals = build_sorted_subsequence_intervals(chrs, starts, ends, strand_flags);

    // We'll accumulate the results here.
    let mut out_idxs = Vec::with_capacity(intervals.len());
    let mut out_starts = Vec::with_capacity(intervals.len());
    let mut out_ends = Vec::with_capacity(intervals.len());

    // A small buffer for intervals belonging to the "current chrom."
    let mut group_buf = Vec::new();

    // Keep track of the current chrom and running cumsum across intervals with that chrom.
    let mut current_chrom = intervals[0].chr;
    let mut running_sum = T::zero();

    // This closure finalizes one chrom-group: it applies negative indexing, forward/reverse logic,
    // filters out intervals with start >= end, and pushes results into output vectors.
    let finalize_group = |group: &mut [SplicedSubsequenceInterval<G, T>],
                          start: T,
                          end: Option<T>,
                          force_plus: bool,
                          out_idxs: &mut Vec<u32>,
                          out_starts: &mut Vec<T>,
                          out_ends: &mut Vec<T>| {
        if group.is_empty() {
            return;
        }

        // The total length of this chrom group is the cumsum of the last interval in the group.
        let total_length = group.last().unwrap().temp_cumsum;

        // If end is None, use total_length.
        let end_val = end.unwrap_or(total_length);

        // Convert negative start/end to their positive equivalents from the 3' end:
        let global_start = if start < T::zero() {
            total_length + start
        } else {
            start
        };
        let global_end = if end_val < T::zero() {
            total_length + end_val
        } else {
            end_val
        };

        // Adjust each interval according to the spliced subsequence logic.
        for iv in group.iter_mut() {
            let cumsum_start = iv.temp_cumsum - iv.temp_length; // spliced start of this exon
            let cumsum_end = iv.temp_cumsum; // spliced end of this exon

            // Determine if we use forward or reverse logic:
            // if force_plus == true, always do forward logic
            // otherwise, check iv.forward_strand
            let is_forward = force_plus || iv.forward_strand;

            if is_forward {
                //   start_adjust = global_start - cumsum_start
                //   if start_adjust > 0 => shift iv.start right
                //   end_adjust   = cumsum_end - global_end
                //   if end_adjust > 0 => shift iv.end left
                let start_adjust = global_start - cumsum_start;
                if start_adjust > T::zero() {
                    iv.start = iv.start + start_adjust;
                }

                let end_adjust = cumsum_end - global_end;
                if end_adjust > T::zero() {
                    iv.end = iv.end - end_adjust;
                }
            } else {
                //   start_adjust = global_start - cumsum_start
                //   if start_adjust > 0 => shift iv.end left
                //   end_adjust   = cumsum_end - global_end
                //   if end_adjust > 0 => shift iv.start right
                let start_adjust = global_start - cumsum_start;
                if start_adjust > T::zero() {
                    iv.end = iv.end - start_adjust
                }

                let end_adjust = cumsum_end - global_end;
                if end_adjust > T::zero() {
                    iv.start = iv.start + end_adjust;
                }
            }
        }

        let strand = group.first().unwrap().forward_strand;
        // Filter out intervals where start >= end, then push to results
        if !strand {
            group.reverse()
        };
        for iv in group.iter() {
            if iv.start < iv.end {
                out_idxs.push(iv.idx);
                out_starts.push(iv.start);
                out_ends.push(iv.end);
            }
        }
    };

    // Single pass over all intervals
    for mut interval in intervals {
        interval.start = interval.start.abs();
        interval.end = interval.end.abs();
        // If we've moved to a new chrom, finalize the previous group, then start fresh.
        if interval.chr != current_chrom {
            finalize_group(
                &mut group_buf,
                start,
                end,
                force_plus_strand,
                &mut out_idxs,
                &mut out_starts,
                &mut out_ends,
            );
            group_buf.clear();
            running_sum = T::zero();
            current_chrom = interval.chr;
        }

        // Prepare a copy of the interval with temp_length + cumsum.
        let mut iv_copy = interval.clone();
        iv_copy.temp_length = iv_copy.end - iv_copy.start;
        iv_copy.temp_cumsum = running_sum + iv_copy.temp_length;
        running_sum = iv_copy.temp_cumsum;

        group_buf.push(iv_copy);
    }

    // Finalize the last group
    finalize_group(
        &mut group_buf,
        start,
        end,
        force_plus_strand,
        &mut out_idxs,
        &mut out_starts,
        &mut out_ends,
    );

    (out_idxs, out_starts, out_ends)
}

pub fn spliced_subseq_per_row<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
    multi_starts: &[T],
    multi_ends: &[T],
    force_plus_strand: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>) {

    // 1) Build intervals
    let mut intervals = build_sorted_subsequence_intervals(chrs, starts, ends, strand_flags);

    if intervals.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // 2) We'll gather intervals for each group in `group_buf`.
    //    Whenever we see a new group, we'll finalize the old one for the rows that belong to it.
    let mut all_idxs   = Vec::new();
    let mut all_starts = Vec::new();
    let mut all_ends   = Vec::new();

    let mut group_buf = Vec::new();
    let mut running_sum = T::zero();

    let mut current_group = intervals[0].chr;
    // This will track the start of the current group in the `intervals` array (by index).
    let mut group_start_i = 0;

    for (i, interval) in intervals.iter_mut().enumerate() {
        // If we've moved on to a new group, finalize the previous group's intervals for
        // all rows in [group_start_i .. i].
        if interval.chr != current_group {
            // finalize old group
            finalize_rows_in_group(
                &group_buf,
                group_start_i,
                i,
                multi_starts,
                multi_ends,
                force_plus_strand,
                &mut all_idxs,
                &mut all_starts,
                &mut all_ends,
            );

            group_buf.clear();
            running_sum = T::zero();
            current_group = interval.chr;
            group_start_i = i;
        }

        // update cumsum for the current interval
        interval.temp_length = interval.end - interval.start;
        interval.temp_cumsum = running_sum + interval.temp_length;
        running_sum = interval.temp_cumsum;

        group_buf.push(interval.clone());
    }

    // finalize the last group
    finalize_rows_in_group(
        &group_buf,
        group_start_i,
        intervals.len(),
        multi_starts,
        multi_ends,
        force_plus_strand,
        &mut all_idxs,
        &mut all_starts,
        &mut all_ends,
    );

    (all_idxs, all_starts, all_ends)
}

/// A small helper function used inside `spliced_subseq_per_row`.
/// Finalizes the intervals for the group currently in `group_buf`
/// for each row index in [row_start .. row_end]. 
/// That means we call `finalize_group(...)` once per row in that range,
/// using `multi_starts[row_i]` / `multi_ends[row_i]`.
fn finalize_rows_in_group<G: GroupType, T: PositionType>(
    group_buf: &[SplicedSubsequenceInterval<G, T>],
    row_start: usize,
    row_end: usize,
    multi_starts: &[T],
    multi_ends: &[T],
    force_plus_strand: bool,
    all_idxs: &mut Vec<u32>,
    all_starts: &mut Vec<T>,
    all_ends: &mut Vec<T>,
) {
    if group_buf.is_empty() {
        return;
    }

    // total spliced length for this group
    let total_length = group_buf.last().unwrap().temp_cumsum;

    // For each row in [row_start .. row_end], use that row's (start,end).
    for row_i in row_start..row_end {
        let s = multi_starts[row_i];
        let e = multi_ends[row_i];

        let (idxs, sts, ends) = finalize_group(
            group_buf,
            s,
            e,
            total_length,
            force_plus_strand,
        );
        all_idxs.extend(idxs);
        all_starts.extend(sts);
        all_ends.extend(ends);
    }
}

fn finalize_group<G: GroupType, T: PositionType>(
    group: &[SplicedSubsequenceInterval<G, T>],
    subseq_start: T,         // e.g. multi_starts[row_i]
    subseq_end: T,   // e.g. multi_ends[row_i]
    total_length: T,
    force_plus_strand: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>) {
    // Convert None => total_length

    // Convert negative start/end into positive offsets from the 3' end:
    let global_start = if subseq_start < T::zero() {
        total_length + subseq_start
    } else {
        subseq_start
    };
    let global_end = if subseq_end < T::zero() {
        total_length + subseq_end
    } else {
        subseq_end
    };

    let mut out_idxs   = Vec::new();
    let mut out_starts = Vec::new();
    let mut out_ends   = Vec::new();

    // We iterate in the stored order. If you need to reverse for minus strand at the group level,
    // you can handle that separately. Here we apply per-exon forward/reverse logic.
    for iv in group.iter() {
        let cumsum_start = iv.temp_cumsum - iv.temp_length;
        let cumsum_end   = iv.temp_cumsum;

        // local copies so we do not mutate `iv`:
        let mut exon_start = iv.start;
        let mut exon_end   = iv.end;

        let is_forward = if force_plus_strand { true } else { iv.forward_strand };

        if is_forward {
            let start_adjust = global_start - cumsum_start;
            if start_adjust > T::zero() {
                exon_start = exon_start + start_adjust;
            }
            let end_adjust = cumsum_end - global_end;
            if end_adjust > T::zero() {
                exon_end = exon_end - end_adjust;
            }
        } else {
            // Reverse logic
            let start_adjust = global_start - cumsum_start;
            if start_adjust > T::zero() {
                exon_end = exon_end - start_adjust;
            }
            let end_adjust = cumsum_end - global_end;
            if end_adjust > T::zero() {
                exon_start = exon_start + end_adjust;
            }
        }

        // Only output if it's still a valid range:
        if exon_start < exon_end {
            out_idxs.push(iv.idx);
            out_starts.push(exon_start);
            out_ends.push(exon_end);
        }
    }

    // If you need final reversing of intervals for minus strand at the group level, handle it here
    // e.g. if !group.first().unwrap().forward_strand && !force_plus_strand { out_idxs.reverse() ... }

    (out_idxs, out_starts, out_ends)
}