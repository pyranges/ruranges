use crate::{ruranges_structs::{GroupType, PositionType}, sorts};

pub fn sweep_line_cluster<G: GroupType, T: PositionType>(
    chrs: &[G],
    starts: &[T],
    ends: &[T],
    slack: T,
) -> (Vec<u32>, Vec<u32>) {
    let mut indices = Vec::with_capacity(chrs.len());
    let mut cluster_ids = Vec::with_capacity(chrs.len());

    if chrs.is_empty() {
        return (cluster_ids, indices);
    };

    let events = sorts::build_sorted_events_single_collection(chrs, starts, ends, slack);

    let mut current_chr = events.first().unwrap().chr;
    let mut current_cluster = 0;
    let mut active_intervals = 0;

    for e in events {
        if e.chr != current_chr {
            current_cluster += 1;
            active_intervals = 0;
            current_chr = e.chr;
        }

        if e.is_start {
            indices.push(e.idx);
            cluster_ids.push(current_cluster);
            active_intervals += 1;
        } else {
            active_intervals -= 1;
            if active_intervals == 0 {
                current_cluster += 1;
            }
        }
    }

    (cluster_ids, indices)
}
