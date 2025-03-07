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
pub fn tile(
    starts: &[i64],
    ends: &[i64],
    negative_strand: &[bool],
    tile_size: i64,
) -> (Vec<i64>, Vec<i64>, Vec<usize>, Vec<f64>) {
    assert_eq!(starts.len(), ends.len());
    assert_eq!(starts.len(), negative_strand.len());

    let mut out_starts = Vec::new();
    let mut out_ends = Vec::new();
    let mut out_indices = Vec::new();
    let mut out_overlaps = Vec::new();

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
            let mut tile_start = if s >= 0 {
                (s / tile_size) * tile_size
            } else {
                let mut multiple = s / tile_size;
                if s % tile_size != 0 {
                    multiple -= 1;
                }
                multiple * tile_size
            };

            // Process each tile that may overlap [s, e).
            while tile_start < e {
                let tile_end = tile_start + tile_size;
                if tile_end > s && tile_start < e {
                    // Calculate overlap fraction
                    let overlap_fraction =
                        (tile_end.min(e) - tile_start.max(s)) as f64 / tile_size as f64;
                    out_starts.push(tile_start);
                    out_ends.push(tile_end);
                    out_indices.push(i);
                    out_overlaps.push(overlap_fraction);
                }
                tile_start += tile_size;
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
            let mut tile_end = if e > 0 {
                // Round up to nearest multiple
                let div = (e - 1) / tile_size; // subtract 1 so that exact multiple doesn't push us one step further
                (div + 1) * tile_size
            } else {
                // e is negative or 0
                let mut multiple = e / tile_size;
                if e % tile_size != 0 {
                    multiple -= 1; // go one step "earlier" in negative direction
                }
                multiple * tile_size
            };

            // Walk backward until the tile_end <= s
            while tile_end > s {
                let tile_start = tile_end - tile_size;
                // Still check for overlap with [s, e).
                if tile_start < e && tile_end > s {
                    let overlap_fraction =
                        (tile_end.min(e) - tile_start.max(s)) as f64 / tile_size as f64;
                    // We keep intervals with the smaller coordinate as start:
                    out_starts.push(tile_start);
                    out_ends.push(tile_end);
                    out_indices.push(i);
                    out_overlaps.push(overlap_fraction);
                }
                tile_end -= tile_size;
            }
        }
    }

    (out_starts, out_ends, out_indices, out_overlaps)
}

pub fn window(
    starts: &[i64],
    ends: &[i64],
    negative_strand: &[bool],
    window_size: i64,
) -> (Vec<i64>, Vec<i64>, Vec<usize>) {
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
            // === Forward direction (same as original) === //
            let mut cur_start = s;
            while cur_start < e {
                let cur_end = (cur_start + window_size).min(e);
                out_starts.push(cur_start);
                out_ends.push(cur_end);
                out_indices.push(i);
                cur_start += window_size;
            }
        } else {
            // === Reverse direction === //
            // For negative strand, we go from `e` down to `s`, stepping by `window_size`.
            let mut cur_end = e;
            while cur_end > s {
                let cur_start = (cur_end - window_size).max(s);
                out_starts.push(cur_start);
                out_ends.push(cur_end);
                out_indices.push(i);
                cur_end -= window_size;
            }
        }
    }

    (out_starts, out_ends, out_indices)
}
