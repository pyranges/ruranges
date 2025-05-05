use radsort::sort_by_key;

use crate::{ruranges_structs::{GroupType, PositionType}, sorts::build_subsequence_intervals};


pub fn sweep_line_cumsum<G, T>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    strand_flags: &[bool],
) -> (Vec<u32>, Vec<T>, Vec<T>)
where
    G: GroupType,
    T: PositionType,
{
    let mut ivals = build_subsequence_intervals(chrs, starts, ends, strand_flags);

    sort_by_key(&mut ivals, |iv| (iv.chr, iv.start));

    let mut idx_out       = Vec::with_capacity(chrs.len());
    let mut cumsum_start  = Vec::with_capacity(chrs.len());
    let mut cumsum_end    = Vec::with_capacity(chrs.len());

    if ivals.is_empty() {
        return (idx_out, cumsum_start, cumsum_end);
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

        idx_out.push(iv.idx);
        cumsum_start.push(s);
        cumsum_end.push(e);
        running_total = e;
    }

    (idx_out, cumsum_start, cumsum_end)
}
