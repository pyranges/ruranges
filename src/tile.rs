use crate::ruranges_structs::{GroupType, PositionType};

pub fn tile_grouped<T, C>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    negative_strand: &[bool],
    tile_size: T,
) -> (Vec<T>, Vec<T>, Vec<usize>, Vec<f64>)
where
    T: PositionType,          // signed integer-like
    C: GroupType + PartialEq, // unsigned integer-like; equality for boundaries
{
    assert_eq!(starts.len(), ends.len());
    assert_eq!(starts.len(), negative_strand.len());
    assert_eq!(starts.len(), chrs.len());

    let n = starts.len();
    let mut out_starts = Vec::new();
    let mut out_ends = Vec::new();
    let mut out_indices = Vec::new();
    let mut out_overlaps = Vec::new();

    if n == 0 {
        return (out_starts, out_ends, out_indices, out_overlaps);
    }

    let denom = tile_size.to_f64().unwrap();

    // Walk groups of equal `chrs` (assumed sorted so equal keys are contiguous)
    let mut g_start = 0usize;
    while g_start < n {
        let mut g_end = g_start + 1;
        while g_end < n && chrs[g_end] == chrs[g_start] {
            g_end += 1;
        }

        // Process this group exactly like the original function (no cross-group state)
        for i in g_start..g_end {
            let s = starts[i];
            let e = ends[i];

            // Skip invalid intervals.
            if e <= s {
                continue;
            }

            if !negative_strand[i] {
                // === Forward direction (same as original) ===

                // First tile boundary <= s (works for negatives too)
                let mut tile_start = if s >= T::zero() {
                    (s / tile_size) * tile_size
                } else {
                    let mut multiple = s / tile_size;
                    if s % tile_size != T::zero() {
                        multiple = multiple - T::one(); // round toward -inf
                    }
                    multiple * tile_size
                };

                // Step forward over tiles and keep overlaps with [s, e)
                while tile_start < e {
                    let tile_end = tile_start + tile_size;
                    if tile_end > s && tile_start < e {
                        let num: f64 = (tile_end.min(e) - tile_start.max(s)).to_f64().unwrap();
                        let overlap_fraction = num / denom;

                        out_starts.push(tile_start);
                        out_ends.push(tile_end);
                        out_indices.push(i);
                        out_overlaps.push(overlap_fraction);
                    }
                    tile_start = tile_start + tile_size;
                }
            } else {
                // === Reverse direction (emit right→left like original) ===

                // First tile boundary >= e
                let mut tile_end = if e > T::zero() {
                    // ceil(e / tile_size) * tile_size without using floating-point
                    let div = (e - T::one()) / tile_size; // ensure exact multiples stay at e
                    (div + T::one()) * tile_size
                } else {
                    // e <= 0
                    let mut multiple = e / tile_size;
                    if e % tile_size != T::zero() {
                        multiple = multiple - T::one(); // fix: was - T::zero()
                    }
                    multiple * tile_size
                };

                // Walk backward over tiles and keep overlaps with [s, e)
                while tile_end > s {
                    let tile_start = tile_end - tile_size;
                    if tile_start < e && tile_end > s {
                        let num: f64 = (tile_end.min(e) - tile_start.max(s)).to_f64().unwrap();
                        let overlap_fraction = num / denom;

                        out_starts.push(tile_start);
                        out_ends.push(tile_end);
                        out_indices.push(i);
                        out_overlaps.push(overlap_fraction);
                    }
                    tile_end = tile_end - tile_size;
                }
            }
        }

        g_start = g_end;
    }

    (out_starts, out_ends, out_indices, out_overlaps)
}


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


use std::cmp::min;

pub fn window_grouped<T, C>(
    chrs: &[C],
    starts: &[T],
    ends: &[T],
    negative_strand: &[bool],
    window_size: T,
) -> (Vec<T>, Vec<T>, Vec<usize>)
where
    T: PositionType,          // PrimInt + Signed + Zero + etc.
    C: GroupType + PartialEq, // PrimInt + Zero + equality to find boundaries
{
    assert_eq!(starts.len(), ends.len());
    assert_eq!(starts.len(), negative_strand.len());
    assert_eq!(starts.len(), chrs.len());
    assert!(window_size > T::zero());

    let n = starts.len();
    let mut out_starts = Vec::new();
    let mut out_ends = Vec::new();
    let mut out_indices = Vec::new();

    if n == 0 {
        return (out_starts, out_ends, out_indices);
    }

    let mut g_start = 0usize;
    while g_start < n {
        // ----- find end of current group (maximal run of equal chrs) -----
        let mut g_end = g_start + 1;
        while g_end < n && chrs[g_end] == chrs[g_start] {
            g_end += 1;
        }

        // ----- per-group state -----
        // PLUS: carry how much we've filled of the current left->right window
        let mut carry_plus = T::zero();

        // MINUS: how many bases we need at the LEFT of the next minus interval
        // to complete the current RIGHT-anchored window across this group
        let mut total_minus_len = T::zero();
        for i in g_start..g_end {
            if negative_strand[i] {
                let len = ends[i] - starts[i];
                if len > T::zero() {
                    total_minus_len = total_minus_len + len;
                }
            }
        }
        let mut minus_needed = if total_minus_len.is_zero() {
            T::zero()
        } else {
            total_minus_len % window_size
        };

        // ----- process intervals in the group -----
        for i in g_start..g_end {
            let s = starts[i];
            let e = ends[i];
            if e <= s {
                continue;
            }

            if !negative_strand[i] {
                // ================= PLUS strand =================
                let mut cur = s;
                let mut remaining = e - s;

                // 1) If we have a carry, complete that pending window first
                if !carry_plus.is_zero() {
                    let need = window_size - carry_plus;
                    let take = min(need, remaining);
                    if take > T::zero() {
                        let seg_end = cur + take;
                        out_starts.push(cur);
                        out_ends.push(seg_end);
                        out_indices.push(i);

                        cur = seg_end;
                        remaining = remaining - take;
                        carry_plus = carry_plus + take;

                        if carry_plus == window_size {
                            carry_plus = T::zero(); // phase boundary reached
                        }
                        if remaining.is_zero() {
                            continue;
                        }
                    }
                }

                // 2) Full windows
                while remaining >= window_size {
                    let seg_end = cur + window_size;
                    out_starts.push(cur);
                    out_ends.push(seg_end);
                    out_indices.push(i);

                    cur = seg_end;
                    remaining = remaining - window_size;
                }

                // 3) Tail becomes next carry
                if remaining > T::zero() {
                    let seg_end = e; // cur + remaining
                    out_starts.push(cur);
                    out_ends.push(seg_end);
                    out_indices.push(i);
                    carry_plus = remaining; // read at start of next interval in this group
                } else {
                    carry_plus = T::zero();
                }
            } else {
                // ================= MINUS strand =================
                // We’ll *collect* segments left→right, then emit them right→left
                // to match legacy ordering (rightmost segment first).
                let mut cur = s;
                let mut remaining = e - s;
                let mut segs: Vec<(T, T)> = Vec::new();

                // 1) Consume initial needed-at-left to align right-anchored phase
                if minus_needed > T::zero() {
                    let take = min(minus_needed, remaining);
                    let seg_end = cur + take;
                    segs.push((cur, seg_end));

                    cur = seg_end;
                    remaining = remaining - take;
                    minus_needed = minus_needed - take;

                    if remaining.is_zero() {
                        // emit collected (just the leftmost piece) in reverse (trivial here)
                        for (st, en) in segs.into_iter().rev() {
                            out_starts.push(st);
                            out_ends.push(en);
                            out_indices.push(i);
                        }
                        continue; // still need more from next minus interval
                    }
                }

                // 2) Full windows (collect left→right)
                while remaining >= window_size {
                    let seg_end = cur + window_size;
                    segs.push((cur, seg_end));

                    cur = seg_end;
                    remaining = remaining - window_size;
                }

                // 3) Tail at RIGHT; set need for LEFT of next minus interval
                if remaining > T::zero() {
                    let seg_end = e; // cur + remaining
                    segs.push((cur, seg_end));

                    let tail = remaining;         // 0 < tail < window_size here
                    let rem = tail % window_size; // == tail
                    minus_needed = if rem.is_zero() { T::zero() } else { window_size - rem };
                } else {
                    minus_needed = T::zero();
                }

                // Emit minus-interval segments in reverse order (right→left)
                for (st, en) in segs.into_iter().rev() {
                    out_starts.push(st);
                    out_ends.push(en);
                    out_indices.push(i);
                }
            }
        }

        g_start = g_end;
    }

    (out_starts, out_ends, out_indices)
}