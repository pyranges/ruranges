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


// ────────────────────────────────────────────────────────────────────────────
// multi-row variant: one pass over the exons, many slices per transcript
// ────────────────────────────────────────────────────────────────────────────

/// Returns (idxs, starts, ends, strands) for *all* (start,end) pairs
pub fn spliced_subseq_per_row<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
    multi_starts: &[T],
    multi_ends: &[T],
    query_forward: &[bool],
    force_plus_strand: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>, Vec<bool>) {
    let mut intervals = build_sorted_subsequence_intervals(chrs, starts, ends, strand_flags);
    if intervals.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut all_idxs    = Vec::new();
    let mut all_starts  = Vec::new();
    let mut all_ends    = Vec::new();
    let mut all_strands = Vec::new();

    let mut group_buf: Vec<SplicedSubsequenceInterval<G, T>> = Vec::new();
    let mut running_sum = T::zero();

    let mut current_chr  = intervals[0].chr;
    let mut group_start  = 0;          // index into `intervals`

    for (i, iv) in intervals.iter_mut().enumerate() {
        if iv.chr != current_chr {
            finalize_rows_in_group(
                &group_buf,
                group_start,
                i,
                multi_starts,
                multi_ends,
                query_forward,
                force_plus_strand,
                &mut all_idxs,
                &mut all_starts,
                &mut all_ends,
                &mut all_strands,
            );
            group_buf.clear();
            running_sum  = T::zero();
            current_chr  = iv.chr;
            group_start  = i;
        }

        iv.temp_length = iv.end - iv.start;
        iv.temp_cumsum = running_sum + iv.temp_length;
        running_sum    = iv.temp_cumsum;

        group_buf.push(iv.clone());
    }

    finalize_rows_in_group(
        &group_buf,
        group_start,
        intervals.len(),
        multi_starts,
        multi_ends,
        query_forward,
        force_plus_strand,
        &mut all_idxs,
        &mut all_starts,
        &mut all_ends,
        &mut all_strands,
    );

    (all_idxs, all_starts, all_ends, all_strands)
}

// ───────────────────────────────── helper for the multi-row case ─────────────

// ────────────────────────────────────────────────────────────────────────────
// helper that finalises one transcript (group) for many query rows
// ────────────────────────────────────────────────────────────────────────────
fn finalize_rows_in_group<G: GroupType, T: PositionType>(
    group_buf:         &[SplicedSubsequenceInterval<G, T>],
    row_start:         usize,
    row_end:           usize,
    multi_starts:      &[T],
    multi_ends:        &[T],
    query_forward:     &[bool],
    force_plus_strand: bool,
    all_idxs:          &mut Vec<u32>,
    all_starts:        &mut Vec<T>,
    all_ends:          &mut Vec<T>,
    all_strands:       &mut Vec<bool>,
) {
    if group_buf.is_empty() { return; }

    let total_len = group_buf.last().unwrap().temp_cumsum;

    for row in row_start..row_end {
        let (idx, st, en, strand) = finalize_group(
            group_buf,
            multi_starts[row],
            multi_ends  [row],
            query_forward[row],          // ← pass the flag down
            total_len,
            force_plus_strand,
        );
        all_idxs   .extend(idx);
        all_starts .extend(st);
        all_ends   .extend(en);
        all_strands.extend(strand);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// core routine: projects one (start,end) onto one transcript
// ────────────────────────────────────────────────────────────────────────────
fn finalize_group<G: GroupType, T: PositionType>(
    group:              &[SplicedSubsequenceInterval<G, T>],
    subseq_start:       T,
    subseq_end:         T,
    query_forward:      bool,             // ← new argument
    total_len:          T,
    force_plus_strand:  bool,
) -> (Vec<u32>, Vec<T>, Vec<T>, Vec<bool>) {

    // translate negative offsets
    let global_start = if subseq_start < T::zero() { total_len + subseq_start }
                       else { subseq_start };
    let global_end   = if subseq_end   < T::zero() { total_len + subseq_end   }
                       else { subseq_end   };

    let mut idxs    = Vec::new();
    let mut sts     = Vec::new();
    let mut ens     = Vec::new();
    let mut strands = Vec::new();

    let transcript_forward = group[0].forward_strand;

    // closure reused for each exon that the slice intersects
    let mut process_iv = |iv: &SplicedSubsequenceInterval<G, T>| {
        let cumsum_start = iv.temp_cumsum - iv.temp_length;
        let cumsum_end   = iv.temp_cumsum;

        // which strand do we perform coordinate arithmetic in?
        let processed_forward =
            if force_plus_strand { true } else { iv.forward_strand };

        let mut st = iv.start;
        let mut en = iv.end;

        if processed_forward {
            // moving from left-to-right
            let shift = global_start - cumsum_start;
            if shift > T::zero() { st = st + shift; }
            let shift = cumsum_end - global_end;
            if shift > T::zero() { en = en - shift; }
        } else {
            // moving from right-to-left
            let shift = global_start - cumsum_start;
            if shift > T::zero() { en = en - shift; }
            let shift = cumsum_end - global_end;
            if shift > T::zero() { st = st + shift; }
        }

        if st < en {
            idxs.push(iv.idx);
            sts .push(st);
            ens .push(en);

            // (+,+) or (−,−) → '+',  (+,−) or (−,+) → '−'
            strands.push(iv.forward_strand == query_forward);
        }
    };

    // walk exons in transcription order
    if transcript_forward {
        for iv in group.iter()       { process_iv(iv); }
    } else {
        for iv in group.iter().rev() { process_iv(iv); }
    }

    (idxs, sts, ens, strands)
}