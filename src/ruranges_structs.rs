use num_traits::{PrimInt, Signed, Zero};
use numpy::Element; // You'll need the num-traits crate
use std::hash::Hash;

pub trait PositionType: PrimInt + Signed + Hash + Copy + radsort::Key + Element + Zero {}
impl<T> PositionType for T where T: PrimInt + Signed + Hash + Copy + radsort::Key + Element + Zero {}
pub trait GroupType: PrimInt + Hash + Copy + radsort::Key + Zero {}
impl<T> GroupType for T where T: PrimInt + Hash + Copy + radsort::Key + Zero {}

pub struct GenomicData<C: GroupType, P: PositionType> {
    pub chroms: Vec<C>,
    pub starts: Vec<P>,
    pub ends: Vec<P>,
    pub strands: Option<Vec<bool>>,
}

#[derive(Debug, Clone)]
pub struct Interval<C: GroupType, T: PositionType> {
    pub group: C,
    pub start: T,
    pub end: T,
    pub idx: u32,
}

#[derive(Debug, Clone, Hash)]
pub struct EventUsize {
    pub chr: i64,
    pub pos: i64,
    pub is_start: bool,
    pub first_set: bool,
    pub idx: usize,
}
/// An "event" in the sweep line:
/// - `pos`: the coordinate (start or end of an interval)
/// - `is_start`: true if it's a start event, false if it's an end event
/// - `set_id`: which set does this interval belong to? (1 or 2)
/// - `idx`: the interval's ID/index
#[derive(Debug, Clone, Hash)]
pub struct Event<C: GroupType, T: PositionType> {
    pub chr: C,
    pub pos: T,
    pub is_start: bool,
    pub first_set: bool,
    pub idx: u32,
}

#[derive(Debug, Clone, Hash)]
pub struct MaxEvent<C: GroupType, T: PositionType> {
    pub chr: C,
    pub pos: T,
    pub start: T,
    pub end: T,
    pub is_start: bool,
    pub first_set: bool,
    pub idx: u32,
}

#[derive(Debug, Clone, Hash)]
pub struct MinEvent<C: GroupType, T: PositionType> {
    pub chr: C,
    pub pos: T,
    pub idx: u32,
}

#[derive(Debug, Clone, Hash)]
pub struct OverlapPair {
    pub idx: u32,
    pub idx2: u32,
}

#[derive(Debug, Clone, Hash)]
pub struct Nearest<T: PositionType> {
    pub distance: T,
    pub idx: u32,
    pub idx2: u32,
}

#[derive(Debug, Clone)]
pub struct SplicedSubsequenceInterval<T: PositionType> {
    /// Encoded chromosome (or chrom+strand+gene) ID.
    pub chr: u32,

    /// The genomic start coordinate.
    pub start: T,

    /// The genomic end coordinate.
    pub end: T,

    pub idx: u32,

    pub forward_strand: bool,

    /// Temporary: length = (end - start).
    pub temp_length: T,

    /// Temporary: cumulative sum of lengths within this chrom group.
    pub temp_cumsum: T,
}

/// A simple struct to hold each interval's data for "subsequence" logic.
#[derive(Clone)]
pub struct SubsequenceInterval {
    pub group_id: i64,        // grouping ID
    pub start: i64,           // genomic start
    pub end: i64,             // genomic end
    pub idx: i64,             // e.g. row index or something else
    pub forward_strand: bool, // true => + strand, false => - strand
}

pub struct GenericEvent<C: GroupType, T: PositionType> {
    pub chr: C,
    pub pos: T,
    pub is_start: bool,
    pub first_set: bool,
    pub idx: u32,
}
