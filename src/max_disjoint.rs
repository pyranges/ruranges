use crate::{ruranges_structs::{GroupType, PositionType}, sorts::build_sorted_intervals};

/// Returns a subset of indexes corresponding to a maximal set of disjoint intervals.
/// `groups`, `starts`, and `ends` must have the same length.
/// `slack` controls how far apart intervals must be to be considered non-overlapping.
pub fn max_disjoint<G:GroupType, T: PositionType>(
    groups: &[G],
    starts: &[T],
    ends: &[T],
    slack: T,
) -> Vec<u32> {
    // Ensure the input slices all have the same length.
    assert_eq!(groups.len(), starts.len());
    assert_eq!(starts.len(), ends.len());

    // Build and sort intervals (presumably by end time)
    let intervals = build_sorted_intervals(groups, starts, ends, None, slack, true);

    if intervals.is_empty() {
        return vec![];
    }

    let mut output: Vec<u32> = Vec::new();

    // Always include the first interval.
    output.push(intervals[0].idx as u32);

    // Track the end of the last accepted interval.
    let mut last_end = intervals[0].end;

    // Iterate through the rest of the intervals.
    for interval in intervals.iter().skip(1) {
        // If the current interval starts after (last_end + slack),
        // it does not overlap, so include it.
        if interval.start > last_end + slack {
            output.push(interval.idx as u32);
            last_end = interval.end;
        }
    }

    output
}
