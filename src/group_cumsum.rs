use radsort::sort_by_key;

use crate::{ruranges_structs::{GroupType, MinInterval, PositionType}, sorts::build_subsequence_intervals};


pub fn sweep_line_cumsum<G, T>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
    sort: bool,
) -> (Vec<u32>, Vec<T>, Vec<T>)
where
    G: GroupType,
    T: PositionType,
{
    let mut ivals = build_subsequence_intervals(chrs, starts, ends, strand_flags);

    sort_by_key(&mut ivals, |iv| (iv.chr, iv.start));

    let mut results= Vec::with_capacity(chrs.len());

    if ivals.is_empty() {
        return (Vec::with_capacity(chrs.len()),Vec::with_capacity(chrs.len()), Vec::with_capacity(chrs.len()));
    }

    let mut current_chr   = ivals[0].chr;
    let mut running_total = T::zero();

    for iv in ivals {
        if iv.chr != current_chr {
            running_total = T::zero();
            current_chr   = iv.chr;
        }

        let len = if iv.end >= iv.start { iv.end - iv.start } else { iv.start - iv.end };

        let s = running_total;
        let e = running_total + len;

        results.push(MinInterval {idx: iv.idx, start: s, end: e});
        running_total = e;
    }

    if sort {
        sort_by_key(&mut results, |i| i.idx);
    }

    let mut out_idxs    = Vec::with_capacity(results.len());
    let mut out_starts  = Vec::with_capacity(results.len());
    let mut out_ends = Vec::with_capacity(results.len());

    for rec in results {
        out_idxs.push(rec.idx);
        out_starts.push(rec.start);
        out_ends.push(rec.end);
    }

    (out_idxs, out_starts, out_ends)
}
