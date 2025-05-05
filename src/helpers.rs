use rustc_hash::FxHashSet;

use crate::ruranges_structs::OverlapPair;





pub fn keep_first_by_idx(pairs: &mut Vec<OverlapPair>) {
    let mut seen_idx = FxHashSet::default();
    pairs.retain(|pair| seen_idx.insert(pair.idx));
}