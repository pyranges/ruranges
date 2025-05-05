use std::collections::HashMap;

use crate::ruranges_structs::{GroupType, PositionType};

pub fn outside_bounds<G: GroupType, T: PositionType>(
    groups:      &[G],
    starts:      &[T],
    ends:        &[T],
    chrom_lens:  &[T],
    clip:        bool,
    only_right:  bool,
) -> Result<(Vec<u32>, Vec<T>, Vec<T>), String> {

    if starts.len() != ends.len()
        || groups.len() != starts.len()
        || chrom_lens.len() != starts.len()
    {
        return Err("All input slices must have the same length".into());
    }

    let n = starts.len();
    let mut idx        = Vec::with_capacity(n);
    let mut out_starts = Vec::with_capacity(n);
    let mut out_ends   = Vec::with_capacity(n);

    for i in 0..n {
        let size        = chrom_lens[i];
        let orig_start  = starts[i];
        let orig_end    = ends[i];

        if !clip {
            // ===== Removal mode =========================================
            let skip = if only_right {
                orig_end > size
            } else {
                orig_end > size || orig_start < T::zero()
            };
            if skip { continue; }

            idx.push(i);
            out_starts.push(orig_start);
            out_ends.push(orig_end);
        } else {
            // ===== Clipping mode ========================================
            if only_right {
                // whole interval right of the chromosome
                if orig_start >= size { continue; }

                let clipped_end = if orig_end > size { size } else { orig_end };

                idx.push(i);
                out_starts.push(orig_start);
                out_ends.push(clipped_end);
            } else {
                // clip on both sides
                if orig_start >= size || orig_end <= T::zero() { continue; }

                let clipped_start = if orig_start < T::zero() { T::zero() } else { orig_start };
                let clipped_end   = if orig_end   > size      { size      } else { orig_end };

                idx.push(i);
                out_starts.push(clipped_start);
                out_ends.push(clipped_end);
            }
        }
    }

    let idx_u32: Vec<u32> = idx.into_iter().map(|x| x as u32).collect();

    Ok((idx_u32, out_starts, out_ends))
}
