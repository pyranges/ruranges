use radsort::sort_by_key;

use crate::{
    ruranges_structs::{GroupType, PositionType, SplicedSubsequenceInterval},
    sorts::build_sorted_subsequence_intervals,
};

/// (idxs, starts, ends, strands) for exactly one (start,end) slice
fn global_shift<T: PositionType>(starts: &[T], ends: &[T]) -> T {
    let mut min_coord = T::zero();
    for &v in starts { if v < min_coord { min_coord = v; } }
    for &v in ends   { if v < min_coord { min_coord = v; } }
    if min_coord < T::zero() { -min_coord } else { T::zero() }
}

/// (idxs, starts, ends, strands) for **one** (start,end) slice
pub fn spliced_subseq<G: GroupType, T: PositionType>(
    chrs:           &[G],
    starts:         &[T],
    ends:           &[T],
    strand_flags:   &[bool],
    start:          T,
    end:            Option<T>,
    force_plus_strand: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>, Vec<bool>) {

    // ────────────────────────── 1. pre-processing: apply global shift ─────
    let shift = global_shift(starts, ends);

    // Either borrow the original slices (shift == 0) or build shifted copies.
    // `tmp_storage` keeps the vectors alive for as long as we need the slices.
    let (starts_slice, ends_slice);
    let _tmp_storage: Option<(Vec<T>, Vec<T>)>;

    if shift > T::zero() {
        let mut s = Vec::with_capacity(starts.len());
        let mut e = Vec::with_capacity(ends.len());
        for i in 0..starts.len() {
            s.push(starts[i] + shift);
            e.push(ends  [i] + shift);
        }
        _tmp_storage = Some((s, e));
        let (s_ref, e_ref) = _tmp_storage.as_ref().unwrap();
        starts_slice = s_ref.as_slice();
        ends_slice   = e_ref.as_slice();
    } else {
        _tmp_storage = None;
        starts_slice = starts;
        ends_slice   = ends;
    }
    // ───────────────────────────────────────────────────────────────────────

    // ────────────── helper struct local to this function ───────────────────
    struct OutRec<T: PositionType> {
        idx:    u32,
        start:  T,
        end:    T,
        strand: bool,
    }

    // Build sorted interval vector (caller guarantees same grouping rules).
    let mut intervals = build_sorted_subsequence_intervals(
        chrs,
        starts_slice,
        ends_slice,
        strand_flags,
    );

    // Early-exit when nothing to do
    if intervals.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut out_recs: Vec<OutRec<T>> = Vec::with_capacity(intervals.len());

    let mut group_buf: Vec<SplicedSubsequenceInterval<G, T>> = Vec::new();
    let mut current_chr = intervals[0].chr;
    let mut running_sum = T::zero();

    // ───────── helper: finalise one transcript/group ───────────────────────
    let mut finalize_group = |group: &mut [SplicedSubsequenceInterval<G, T>]| {
        if group.is_empty() { return; }

        // total spliced length
        let total_len = group.last().unwrap().temp_cumsum;
        let end_val   = end.unwrap_or(total_len);

        // translate negative offsets
        let global_start = if start < T::zero() { total_len + start } else { start };
        let global_end   = if end_val < T::zero() { total_len + end_val } else { end_val };

        let group_forward = group[0].forward_strand;

        // per-exon closure so we don’t duplicate maths
        let mut process_iv = |iv: &mut SplicedSubsequenceInterval<G, T>| {
            let cumsum_start = iv.temp_cumsum - iv.temp_length;
            let cumsum_end   = iv.temp_cumsum;

            let mut st = iv.start;
            let mut en = iv.end;

            // coordinate arithmetic orientation
            let processed_forward =
                force_plus_strand || iv.forward_strand;

            if processed_forward {
                let shift = global_start - cumsum_start;
                if shift > T::zero() { st = st + shift; }
                let shift = cumsum_end - global_end;
                if shift > T::zero() { en = en - shift; }
            } else {
                let shift = global_start - cumsum_start;
                if shift > T::zero() { en = en - shift; }
                let shift = cumsum_end - global_end;
                if shift > T::zero() { st = st + shift; }
            }

            // keep only non-empty pieces
            if st < en {
                out_recs.push(OutRec {
                    idx:    iv.idx,
                    start:  st,
                    end:    en,
                    strand: iv.forward_strand == processed_forward, // (+)*(+) or (−)*(−) → '+'
                });
            }
        };

        // walk exons in transcription order
        if group_forward {
            for iv in group.iter_mut()       { process_iv(iv); }
        } else {
            for iv in group.iter_mut().rev() { process_iv(iv); }
        }
    };
    // ───────────────────────────────────────────────────────────────────────

    // single linear scan over all exons
    for mut iv in intervals.into_iter() {
        iv.start = iv.start.abs();
        iv.end   = iv.end.abs();

        // new chromosome ⇒ flush buffer
        if iv.chr != current_chr {
            finalize_group(&mut group_buf);
            group_buf.clear();
            running_sum = T::zero();
            current_chr = iv.chr;
        }

        iv.temp_length = iv.end - iv.start;
        iv.temp_cumsum = running_sum + iv.temp_length;
        running_sum    = iv.temp_cumsum;

        group_buf.push(iv);
    }
    finalize_group(&mut group_buf);

    // restore original row order
    sort_by_key(&mut out_recs, |r| r.idx);

    // ───────── explode OutRec list into parallel result vectors ────────────
    let mut out_idxs    = Vec::with_capacity(out_recs.len());
    let mut out_starts  = Vec::with_capacity(out_recs.len());
    let mut out_ends    = Vec::with_capacity(out_recs.len());
    let mut out_strands = Vec::with_capacity(out_recs.len());

    for rec in out_recs {
        out_idxs.push(rec.idx);
        out_starts.push(rec.start);
        out_ends.push(rec.end);
        out_strands.push(rec.strand);
    }

    // ─────────────────────────── 3. post-processing: undo shift ────────────
    if shift > T::zero() {
        for v in &mut out_starts { *v = *v - shift; }
        for v in &mut out_ends   { *v = *v - shift; }
    }
    // ───────────────────────────────────────────────────────────────────────

    (out_idxs, out_starts, out_ends, out_strands)
}

pub fn spliced_subseq_multi<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
    slice_starts: &[T],
    slice_ends: &[Option<T>],
    force_plus_strand: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>, Vec<bool>) {
    assert_eq!(chrs.len(), starts.len());
    assert_eq!(starts.len(), ends.len());
    assert_eq!(ends.len(), strand_flags.len());
    assert_eq!(strand_flags.len(), slice_starts.len());
    assert_eq!(slice_starts.len(), slice_ends.len());

    let shift = global_shift(starts, ends);

    let (starts_slice, ends_slice);
    let _tmp_storage: Option<(Vec<T>, Vec<T>)>;
    if shift > T::zero() {
        let mut s = Vec::with_capacity(starts.len());
        let mut e = Vec::with_capacity(ends.len());
        for i in 0..starts.len() {
            s.push(starts[i] + shift);
            e.push(ends[i] + shift);
        }
        _tmp_storage = Some((s, e));
        let (s_ref, e_ref) = _tmp_storage.as_ref().unwrap();
        starts_slice = s_ref.as_slice();
        ends_slice = e_ref.as_slice();
    } else {
        _tmp_storage = None;
        starts_slice = starts;
        ends_slice = ends;
    }

    struct OutRec<T: PositionType> {
        idx: u32,
        start: T,
        end: T,
        strand: bool,
    }

    let mut intervals =
        build_sorted_subsequence_intervals(chrs, starts_slice, ends_slice, strand_flags);

    if intervals.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut out_recs: Vec<OutRec<T>> = Vec::with_capacity(intervals.len());
    let mut group_buf: Vec<SplicedSubsequenceInterval<G, T>> = Vec::new();
    let mut current_chr = intervals[0].chr;
    let mut running_sum = T::zero();
    let mut current_slice_start: T = slice_starts[intervals[0].idx as usize];
    let mut current_slice_end: Option<T> = slice_ends[intervals[0].idx as usize];

    let mut finalize_group = |group: &mut [SplicedSubsequenceInterval<G, T>],
                              slice_start: T,
                              slice_end: Option<T>| {
        if group.is_empty() {
            return;
        }

        let total_len = group.last().unwrap().temp_cumsum;
        let end_val = slice_end.unwrap_or(total_len);

        let global_start = if slice_start < T::zero() {
            total_len + slice_start
        } else {
            slice_start
        };
        let global_end = if end_val < T::zero() {
            total_len + end_val
        } else {
            end_val
        };

        let group_forward = group[0].forward_strand;

        let mut process_iv = |iv: &mut SplicedSubsequenceInterval<G, T>| {
            let cumsum_start = iv.temp_cumsum - iv.temp_length;
            let cumsum_end = iv.temp_cumsum;

            let mut st = iv.start;
            let mut en = iv.end;

            let processed_forward = force_plus_strand || iv.forward_strand;

            if processed_forward {
                let shift = global_start - cumsum_start;
                if shift > T::zero() {
                    st = st + shift;
                }
                let shift = cumsum_end - global_end;
                if shift > T::zero() {
                    en = en - shift;
                }
            } else {
                let shift = global_start - cumsum_start;
                if shift > T::zero() {
                    en = en - shift;
                }
                let shift = cumsum_end - global_end;
                if shift > T::zero() {
                    st = st + shift;
                }
            }

            if st < en {
                out_recs.push(OutRec {
                    idx: iv.idx,
                    start: st,
                    end: en,
                    strand: iv.forward_strand == processed_forward,
                });
            }
        };

        if group_forward {
            for iv in group.iter_mut() {
                process_iv(iv);
            }
        } else {
            for iv in group.iter_mut().rev() {
                process_iv(iv);
            }
        }
    };

    for mut iv in intervals.into_iter() {
        iv.start = iv.start.abs();
        iv.end = iv.end.abs();

        if iv.chr != current_chr {
            finalize_group(&mut group_buf, current_slice_start, current_slice_end);
            group_buf.clear();
            running_sum = T::zero();
            current_chr = iv.chr;
            current_slice_start = slice_starts[iv.idx as usize];
            current_slice_end = slice_ends[iv.idx as usize];
        }

        iv.temp_length = iv.end - iv.start;
        iv.temp_cumsum = running_sum + iv.temp_length;
        running_sum = iv.temp_cumsum;

        group_buf.push(iv);
    }
    finalize_group(&mut group_buf, current_slice_start, current_slice_end);

    sort_by_key(&mut out_recs, |r| r.idx);

    let mut out_idxs = Vec::with_capacity(out_recs.len());
    let mut out_starts = Vec::with_capacity(out_recs.len());
    let mut out_ends = Vec::with_capacity(out_recs.len());
    let mut out_strands = Vec::with_capacity(out_recs.len());

    for rec in out_recs {
        out_idxs.push(rec.idx);
        out_starts.push(rec.start);
        out_ends.push(rec.end);
        out_strands.push(rec.strand);
    }

    if shift > T::zero() {
        for v in &mut out_starts {
            *v = *v - shift;
        }
        for v in &mut out_ends {
            *v = *v - shift;
        }
    }

    (out_idxs, out_starts, out_ends, out_strands)
}
