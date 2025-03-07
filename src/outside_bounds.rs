use std::collections::HashMap;

pub fn outside_bounds(
    groups: &[u32],
    starts: &[i64],
    ends: &[i64],
    chromsizes: &HashMap<u32, i64>,
    clip: bool,
    only_right: bool,
) -> Result<(Vec<usize>, Vec<i64>, Vec<i64>), String> {
    if starts.len() != ends.len() || groups.len() != starts.len() {
        return Err("Input slices must all have the same length.".to_string());
    }
    let n = starts.len();
    let mut indices = Vec::new();
    let mut new_starts = Vec::new();
    let mut new_ends = Vec::new();

    for i in 0..n {
        let chrom = groups[i];
        // Look up the chromosome size; error if missing.
        let size = chromsizes
            .get(&chrom)
            .ok_or_else(|| format!("Chromosome {} not found in chromsizes", chrom))?;
        let orig_start = starts[i];
        let orig_end = ends[i];

        if !clip {
            // Removal mode.
            if only_right {
                if orig_end > *size {
                    continue; // skip interval
                }
            } else {
                if orig_end > *size || orig_start < 0 {
                    continue; // skip interval
                }
            }
            indices.push(i);
            new_starts.push(orig_start);
            new_ends.push(orig_end);
        } else {
            // Clipping mode.
            if only_right {
                // If the entire interval is completely right-of-bound, skip it.
                if orig_start >= *size {
                    continue;
                }
                let clipped_end = if orig_end > *size { *size } else { orig_end };
                indices.push(i);
                new_starts.push(orig_start);
                new_ends.push(clipped_end);
            } else {
                // Clip both sides.
                if orig_start >= *size || orig_end <= 0 {
                    continue;
                }
                let clipped_start = if orig_start < 0 { 0 } else { orig_start };
                let clipped_end = if orig_end > *size { *size } else { orig_end };
                indices.push(i);
                new_starts.push(clipped_start);
                new_ends.push(clipped_end);
            }
        }
    }
    Ok((indices, new_starts, new_ends))
}
