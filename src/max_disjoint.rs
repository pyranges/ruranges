use radsort::sort;

use crate::{ruranges_structs::{GroupType, PositionType}, sorts::build_sorted_intervals};

pub fn max_disjoint<G, T>(
    groups: &[G],
    starts: &[T],
    ends: &[T],
    slack: T,
) -> Vec<u32>
where
    G: GroupType,
    T: PositionType,
{
    // Ensure the input slices all have the same length.
    assert_eq!(groups.len(), starts.len());
    assert_eq!(starts.len(), ends.len());

    // Build and sort intervals (group ➜ start ➜ end).
    let intervals = build_sorted_intervals(groups, starts, ends, None, slack, true);

    if intervals.is_empty() {
        return Vec::new();
    }

    // Output indices of the chosen, mutually disjoint intervals.
    let mut output: Vec<u32> = Vec::with_capacity(intervals.len());

    // Always accept the first interval of the first group.
    let mut current_group = intervals[0].group;
    let mut last_end      = intervals[0].end;
    output.push(intervals[0].idx as u32);

    // Walk through the remaining intervals.
    for interval in intervals.iter().skip(1) {

        // NEW: different groups are automatically disjoint – start a fresh tracker.
        if interval.group != current_group {
            current_group = interval.group;
            last_end      = interval.end;
            output.push(interval.idx as u32);
            continue;
        }

        // Same group: test true overlap.
        if interval.start > last_end + slack {
            last_end = interval.end;
            output.push(interval.idx as u32);
        }
    }


    sort(&mut output);
    output
}