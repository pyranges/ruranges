use crate::ruranges_structs::{GroupType, PositionType};

/// Returns tiled intervals along with the original row index and the tile overlap as a fraction of tile size.
///
/// For each interval defined by `starts[i]` and `ends[i]`, the function splits the genome into
/// fixed-size tiles of length `tile_size` (e.g., [tile_start, tile_start + tile_size)) and computes
/// the fraction of each tile that overlaps the original interval.
///
/// # Examples
///
/// - For an interval 99–100 with tile size 100, the tile [0,100) gets an overlap fraction of 0.01.
/// - For an interval 100–250 with tile size 100:
///     - The tile [100,200) gets an overlap fraction of 1.0,
///     - The tile [200,300) gets an overlap fraction of 0.5.
pub fn tile<T>(
    starts: &[T],
    ends: &[T],
    negative_strand: &[bool],
    tile_size: T,
) -> (Vec<T>, Vec<T>, Vec<usize>, Vec<f64>) where T: PositionType {
    assert_eq!(starts.len(), ends.len());
    assert_eq!(starts.len(), negative_strand.len());

    let mut out_starts = Vec::new();
    let mut out_ends = Vec::new();
    let mut out_indices = Vec::new();
    let mut out_overlaps = Vec::new();
    let denom = tile_size.to_f64().unwrap();

    for (i, ((&s, &e), &is_neg)) in starts
        .iter()
        .zip(ends.iter())
        .zip(negative_strand.iter())
        .enumerate()
    {
        // Skip invalid intervals.
        if e <= s {
            continue;
        }

        if !is_neg {
            // === Forward direction (same as original) === //

            // Determine the first tile boundary that is <= s.
            let mut tile_start = if s >= T::zero() {
                (s / tile_size) * tile_size
            } else {
                let mut multiple = s / tile_size;
                if s % tile_size != T::zero() {
                    multiple = multiple - T::one();
                }
                multiple * tile_size
            };

            // Process each tile that may overlap [s, e).
            while tile_start < e {
                let tile_end = tile_start + tile_size;
                if tile_end > s && tile_start < e {
                    // Calculate overlap fraction
                    let num: f64 = (tile_end.min(e) - tile_start.max(s)).to_f64().unwrap();
                    let denom: f64 = tile_size.to_f64().unwrap();
                    let overlap_fraction = num / denom;
                    out_starts.push(tile_start);
                    out_ends.push(tile_end);
                    out_indices.push(i);
                    out_overlaps.push(overlap_fraction);
                }
                tile_start = tile_start + tile_size;
            }
        } else {
            // === Reverse direction === //

            // We want to find the first tile boundary >= e.
            // Because e could be negative or positive, we handle it similarly to the forward code,
            // but in reverse.
            //
            // Example logic:
            //   if e = 787 and tile_size = 100,
            //   the first boundary >= 787 is 800
            //
            // For negative e, we do a similar approach but be mindful of rounding.
            let mut tile_end = if e > T::zero() {
                // Round up to nearest multiple
                let div = (e - T::one()) / tile_size; // subtract 1 so that exact multiple doesn't push us one step further
                (div + T::one()) * tile_size
            } else {
                // e is negative or 0
                let mut multiple = e / tile_size;
                if e % tile_size != T::zero() {
                    multiple = multiple - T::zero(); // go one step "earlier" in negative direction
                }
                multiple * tile_size
            };

            // Walk backward until the tile_end <= s
            while tile_end > s {
                let tile_start = tile_end - tile_size;
                // Still check for overlap with [s, e).
                if tile_start < e && tile_end > s {
                    let num= (tile_end.min(e) - tile_start.max(s)).to_f64().unwrap();
                    let overlap_fraction = num / denom;
                    // We keep intervals with the smaller coordinate as start:
                    out_starts.push(tile_start);
                    out_ends.push(tile_end);
                    out_indices.push(i);
                    out_overlaps.push(overlap_fraction);
                }
                tile_end = tile_end - tile_size;
            }
        }
    }

    (out_starts, out_ends, out_indices, out_overlaps)
}


pub fn window<T>(
    starts: &[T],
    ends: &[T],
    negative_strand: &[bool],
    window_size: T,
) -> (Vec<T>, Vec<T>, Vec<usize>) where T: PositionType {
    assert_eq!(starts.len(), ends.len());
    assert_eq!(starts.len(), negative_strand.len());

    let mut out_starts = Vec::new();
    let mut out_ends = Vec::new();
    let mut out_indices = Vec::new();

    for (i, ((&s, &e), &is_neg)) in starts
        .iter()
        .zip(ends.iter())
        .zip(negative_strand.iter())
        .enumerate()
    {
        if e <= s {
            continue;
        }

        if !is_neg {
            let mut cur_start = s;
            while cur_start < e {
                let cur_end = (cur_start + window_size).min(e);
                out_starts.push(cur_start);
                out_ends.push(cur_end);
                out_indices.push(i);
                cur_start = cur_start + window_size;
            }
        } else {
            let mut cur_end = e;
            while cur_end > s {
                let cur_start = (cur_end - window_size).max(s);
                out_starts.push(cur_start);
                out_ends.push(cur_end);
                out_indices.push(i);
                cur_end = cur_end - window_size;
            }
        }
    }

    (out_starts, out_ends, out_indices)
}
