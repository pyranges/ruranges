use std::collections::HashMap;
use std::f32::consts::E;
use std::str::FromStr;
use std::time::Instant;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::boundary::sweep_line_boundary;
use crate::cluster::sweep_line_cluster;
use crate::complement::sweep_line_non_overlaps;
use crate::complement_single::sweep_line_complement;
use crate::extend::{extend, extend_grp};
use crate::max_disjoint::max_disjoint;
use crate::merge::sweep_line_merge;
use crate::nearest::nearest;
use crate::outside_bounds::outside_bounds;
// use crate::nearest::nearest;
use crate::overlaps::{self, count_overlaps};
use crate::ruranges_structs::{OverlapPair, PositionType};
use crate::spliced_subsequence::{spliced_subseq, spliced_subseq_per_row};
use crate::split::sweep_line_split;
use crate::subtract::sweep_line_subtract;
use crate::tile::{tile, window};
use crate::{outside_bounds, sorts};

// use crate::bindings::polars_bindings::{
//     self, chromsweep_polars, cluster_polars, sweep_line_overlaps_set1_polars,
// };



#[pyfunction]
pub fn chromsweep_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    chrs2: PyReadonlyArray1<u32>,
    starts2: PyReadonlyArray1<i64>,
    ends2: PyReadonlyArray1<i64>,
    slack: i64,
    overlap_type: &str,
    contained: bool,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let chrs_slice2 = chrs2.as_slice()?;
    let starts_slice2 = starts2.as_slice()?;
    let ends_slice2 = ends2.as_slice()?;

    let (idx1, idx2) = overlaps::chromsweep_full(
        chrs_slice,
        starts_slice,
        ends_slice,
        chrs_slice2,
        starts_slice2,
        ends_slice2,
        slack,
        overlap_type,
        contained,
    );
    let res = Ok((
        idx1.into_pyarray(py).to_owned().into(),
        idx2.into_pyarray(py).to_owned().into(),
    ));
    res
}


#[pyfunction]
#[pyo3(signature = (*, chrs, starts, ends, chrs2, starts2, ends2, slack=0, k=1, include_overlaps=true, direction="any"))]
pub fn nearest_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    chrs2: PyReadonlyArray1<u32>,
    starts2: PyReadonlyArray1<i64>,
    ends2: PyReadonlyArray1<i64>,
    slack: i64,
    k: usize,
    include_overlaps: bool,
    direction: &str,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<i64>>)> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let chrs_slice2 = chrs2.as_slice()?;
    let starts_slice2 = starts2.as_slice()?;
    let ends_slice2 = ends2.as_slice()?;

    let result = nearest(
        chrs_slice,
        starts_slice,
        ends_slice,
        chrs_slice2,
        starts_slice2,
        ends_slice2,
        slack,
        k,
        include_overlaps,
        direction,
    );
    let res = Ok((
        result.0.into_pyarray(py).to_owned().into(),
        result.1.into_pyarray(py).to_owned().into(),
        result.2.into_pyarray(py).to_owned().into(),
    ));
    res
}

#[pyfunction]
pub fn subtract_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    chrs2: PyReadonlyArray1<u32>,
    starts2: PyReadonlyArray1<i64>,
    ends2: PyReadonlyArray1<i64>,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let chrs_slice2 = chrs2.as_slice()?;
    let starts_slice2 = starts2.as_slice()?;
    let ends_slice2 = ends2.as_slice()?;

    let result = sweep_line_subtract(
        chrs_slice,
        starts_slice,
        ends_slice,
        chrs_slice2,
        starts_slice2,
        ends_slice2,
    );
    Ok((
        result.0.into_pyarray(py).to_owned().into(),
        result.1.into_pyarray(py).to_owned().into(),
        result.2.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, sort_reverse_direction=None))]
pub fn sort_intervals_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    sort_reverse_direction: Option<PyReadonlyArray1<bool>>,
    py: Python,
) -> PyResult<Py<PyArray1<u32>>> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let sort_reverse_direction_slice = match &sort_reverse_direction {
        Some(reverse) => Some(reverse.as_slice()?),
        None => None,
    };

    let indexes = sorts::sort_order_idx(
        chrs_slice,
        starts_slice,
        ends_slice,
        sort_reverse_direction_slice,
    );
    Ok(indexes.into_pyarray(py).to_owned().into())
}

// #[pyfunction]
// pub fn nearest_intervals_unique_k_numpy(
//     chrs: PyReadonlyArray1<i64>,
//     starts: PyReadonlyArray1<i64>,
//     ends: PyReadonlyArray1<i64>,
//     idxs: PyReadonlyArray1<i64>,
//     chrs2: PyReadonlyArray1<i64>,
//     starts2: PyReadonlyArray1<i64>,
//     ends2: PyReadonlyArray1<i64>,
//     idxs2: PyReadonlyArray1<i64>,
//     k: usize,
//     overlaps: bool,
//     direction: &str,
//     py: Python,
// ) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
//     let dir: Direction = direction.parse().expect("Invalid direction, must be forward, backwards, or any.");
//
//     let chrs_slice = chrs.as_slice()?;
//     let starts_slice = starts.as_slice()?;
//     let ends_slice = ends.as_slice()?;
//     let idxs_slice = idxs.as_slice()?;
//     let chrs_slice2 = chrs2.as_slice()?;
//     let starts_slice2 = starts2.as_slice()?;
//     let ends_slice2 = ends2.as_slice()?;
//     let idxs_slice2 = idxs2.as_slice()?;
//
//     let mut right_result = Some((Vec::new(), Vec::new(), Vec::new()));
//     let mut left_result = Some((Vec::new(), Vec::new(), Vec::new()));
//     let mut overlap_result: Option<(Vec<i64>, Vec<i64>)> = None;
//
//     rayon::scope(|s| {
//         let right_ref = &mut right_result;
//         let left_ref = &mut left_result;
//         let overlap_ref = &mut overlap_result;
//         if dir != Direction::Forward {
//             s.spawn(|_| {
//                 let tmp = sweep_line_k_nearest(
//                     chrs_slice,
//                     ends_slice,
//                     idxs_slice,
//                     chrs_slice2,
//                     starts_slice2,
//                     idxs_slice2,
//                     false,
//                     k,
//                 );
//                 *left_ref = Some(tmp);
//             });
//         }
//         if dir != Direction::Backward {
//             s.spawn(|_| {
//                 let tmp = sweep_line_k_nearest(
//                     chrs_slice,
//                     starts_slice,
//                     idxs_slice,
//                     chrs_slice2,
//                     ends_slice2,
//                     idxs_slice2,
//                     true,
//                     k,
//                 );
//                 *right_ref = Some(tmp);
//             });
//         }
//         if overlaps {
//             s.spawn(|_| {
//                         let tmp_overlap = sweep_line_overlaps(
//                             chrs_slice,
//                             starts_slice,
//                             ends_slice,
//                             idxs_slice,
//                             chrs_slice2,
//                             starts_slice2,
//                             ends_slice2,
//                             idxs_slice2,
//                             0,
//                         );
//                         *overlap_ref = Some(tmp_overlap);
//                     });
//             }
//             });
//     let (r_idxs1, r_idxs2, r_dists) = right_result.unwrap();
//     let (l_idxs1, l_idxs2, l_dists) = left_result.unwrap();
//
//     let (_out_idx1, _out_idx2, _out_dists) = pick_k_distances_combined(
//         &l_idxs1,
//         &l_idxs2,
//         &l_dists,
//         &r_idxs1,
//         &r_idxs2,
//         &r_dists,
//         k,
//     );
//
//     let (out_idx1, out_idx2, out_dists) = if overlaps {
//         let (o_idxs1, o_idxs2) = overlap_result.unwrap();
//         let dists = vec![0; o_idxs1.len()];
//         pick_k_distances_combined(
//             &_out_idx1,
//             &_out_idx2,
//             &_out_dists,
//             &o_idxs1,
//             &o_idxs2,
//             &dists,
//             k,
//         )
//     } else {
//         (_out_idx1, _out_idx2, _out_dists)
//     };
//
//     Ok((
//         out_idx1.into_pyarray(py).to_owned().into(),
//         out_idx2.into_pyarray(py).to_owned().into(),
//         out_dists.into_pyarray(py).to_owned().into(),
//     ))
// }

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, slack=0))]
pub fn cluster_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    slack: i64,
    py: Python,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
    let (cluster_ids, indices) = sweep_line_cluster(
        chrs.as_slice()?,
        starts.as_slice()?,
        ends.as_slice()?,
        slack,
    );
    Ok((
        cluster_ids.into_pyarray(py).to_owned().into(),
        indices.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (starts, ends, negative_strand, tile_size))]
pub fn tile_numpy(
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    negative_strand: PyReadonlyArray1<bool>,
    tile_size: i64,
    py: Python,
) -> PyResult<(
    Py<PyArray1<usize>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f64>>,
)> {
    let (starts, ends, indices, overlap_fraction) = tile(
        starts.as_slice()?,
        ends.as_slice()?,
        negative_strand.as_slice()?,
        tile_size,
    );
    Ok((
        indices.into_pyarray(py).to_owned().into(),
        starts.into_pyarray(py).to_owned().into(),
        ends.into_pyarray(py).to_owned().into(),
        overlap_fraction.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (starts, ends, negative_strand, window_size))]
pub fn window_numpy(
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    negative_strand: PyReadonlyArray1<bool>,
    window_size: i64,
    py: Python,
) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let (starts, ends, indices) = window(
        starts.as_slice()?,
        ends.as_slice()?,
        negative_strand.as_slice()?,
        window_size,
    );
    Ok((
        indices.into_pyarray(py).to_owned().into(),
        starts.into_pyarray(py).to_owned().into(),
        ends.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, slack=0))]
pub fn merge_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    slack: i64,
    py: Python,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<u32>>,
)> {
    let (indices, starts, ends, counts) = sweep_line_merge(
        chrs.as_slice()?,
        starts.as_slice()?,
        ends.as_slice()?,
        slack,
    );
    Ok((
        indices.into_pyarray(py).to_owned().into(),
        starts.into_pyarray(py).to_owned().into(),
        ends.into_pyarray(py).to_owned().into(),
        counts.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (groups, starts, ends, negative_strand, ext, ext_3, ext_5))]
pub fn extend_numpy(
    groups: Option<PyReadonlyArray1<u32>>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    negative_strand: PyReadonlyArray1<bool>,
    ext: Option<i64>,
    ext_3: Option<i64>,
    ext_5: Option<i64>,
    py: Python,
) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let (starts, ends) = match groups {
        Some(groups) => extend_grp(
            groups.as_slice()?,
            starts.as_slice()?,
            ends.as_slice()?,
            negative_strand.as_slice()?,
            ext,
            ext_3,
            ext_5,
        ),
        None => extend(
            starts.as_slice()?,
            ends.as_slice()?,
            negative_strand.as_slice()?,
            ext,
            ext_3,
            ext_5,
        ),
    };
    Ok((
        starts.into_pyarray(py).to_owned().into(),
        ends.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, slack=0))]
pub fn max_disjoint_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    slack: i64,
    py: Python,
) -> PyResult<Py<PyArray1<u32>>> {
    let indices = max_disjoint(
        chrs.as_slice()?,
        starts.as_slice()?,
        ends.as_slice()?,
        slack,
    );
    Ok(indices.into_pyarray(py).to_owned().into())
}

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, slack=0, between=false))]
pub fn split_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    slack: i64,
    between: bool,
    py: Python,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let (indices, starts, ends) = sweep_line_split(
        chrs.as_slice()?,
        starts.as_slice()?,
        ends.as_slice()?,
        slack,
        between,
    );
    Ok((
        indices.into_pyarray(py).to_owned().into(),
        starts.into_pyarray(py).to_owned().into(),
        ends.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, strand_flags, start, end = None, force_plus_strand = false))]
pub fn spliced_subsequence_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    strand_flags: PyReadonlyArray1<bool>,
    start: i64,
    end: Option<i64>,
    force_plus_strand: bool,
    py: Python,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let (outidx, outstarts, outends) = spliced_subseq(
        chrs.as_slice()?,
        starts.as_slice()?,
        ends.as_slice()?,
        strand_flags.as_slice()?,
        start,
        end,
        force_plus_strand,
    );
    Ok((
        outidx.into_pyarray(py).to_owned().into(),
        outstarts.into_pyarray(py).to_owned().into(),
        outends.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (chrs, starts, ends, strand_flags, starts_subseq, ends_subseq, force_plus_strand = false))]
pub fn spliced_subsequence_per_row_numpy(
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    strand_flags: PyReadonlyArray1<bool>,
    starts_subseq: PyReadonlyArray1<i64>,
    ends_subseq: PyReadonlyArray1<i64>,
    force_plus_strand: bool,
    py: Python,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let (outidx, outstarts, outends) = spliced_subseq_per_row(
        chrs.as_slice()?,
        starts.as_slice()?,
        ends.as_slice()?,
        strand_flags.as_slice()?,
        starts_subseq.as_slice()?,
        ends_subseq.as_slice()?,
        force_plus_strand,
    );
    Ok((
        outidx.into_pyarray(py).to_owned().into(),
        outstarts.into_pyarray(py).to_owned().into(),
        outends.into_pyarray(py).to_owned().into(),
    ))
}


#[pyfunction]
pub fn complement_overlaps_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    chrs2: PyReadonlyArray1<u32>,
    starts2: PyReadonlyArray1<i64>,
    ends2: PyReadonlyArray1<i64>,
    slack: i64,
) -> PyResult<Py<PyArray1<u32>>> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let chrs_slice2 = chrs2.as_slice()?;
    let starts_slice2 = starts2.as_slice()?;
    let ends_slice2 = ends2.as_slice()?;

    let result = sweep_line_non_overlaps(
        chrs_slice,
        starts_slice,
        ends_slice,
        chrs_slice2,
        starts_slice2,
        ends_slice2,
        slack,
    );
    Ok(result.into_pyarray(py).to_owned().into())
}

#[pyfunction]
pub fn complement_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    slack: i64,
    chrom_len_ids: PyReadonlyArray1<u32>,
    chrom_lens: PyReadonlyArray1<i64>,
    include_first_interval: bool,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<u32>>,
)> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;

    let keys = chrom_len_ids.as_slice()?;
    let vals = chrom_lens.as_slice()?;

    if keys.len() != vals.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys array and values array must have the same length",
        ));
    }
    let mut lens_map = FxHashMap::default();
    for (&k, &v) in keys.iter().zip(vals.iter()) {
        lens_map.insert(k, v);
    }

    let (outchrs, outstarts, outends, outidxs) = sweep_line_complement(
        chrs_slice,
        starts_slice,
        ends_slice,
        slack,
        &lens_map,
        include_first_interval,
    );
    Ok((
        outchrs.into_pyarray(py).to_owned().into(),
        outstarts.into_pyarray(py).to_owned().into(),
        outends.into_pyarray(py).to_owned().into(),
        outidxs.into_pyarray(py).to_owned().into(),
    ))
}

#[pyfunction]
pub fn count_overlaps_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    chrs2: PyReadonlyArray1<u32>,
    starts2: PyReadonlyArray1<i64>,
    ends2: PyReadonlyArray1<i64>,
    slack: i64,
) -> PyResult<Py<PyArray1<u32>>> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let chrs_slice2 = chrs2.as_slice()?;
    let starts_slice2 = starts2.as_slice()?;
    let ends_slice2 = ends2.as_slice()?;

    let result = count_overlaps(
        chrs_slice,
        starts_slice,
        ends_slice,
        chrs_slice2,
        starts_slice2,
        ends_slice2,
        slack,
    );
    Ok(result.into_pyarray(py).to_owned().into())
}

#[pyfunction]
pub fn boundary_numpy(
    py: Python,
    chrs: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<u32>>,
)> {
    let chrs_slice = chrs.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;

    let (outidxs, outstarts, outends, counts) =
        sweep_line_boundary(chrs_slice, starts_slice, ends_slice);
    Ok((
        outidxs.into_pyarray(py).to_owned().into(),
        outstarts.into_pyarray(py).to_owned().into(),
        outends.into_pyarray(py).to_owned().into(),
        counts.into_pyarray(py).to_owned().into(),
    ))
}

#[derive(Debug, PartialEq)]
enum Direction {
    Forward,
    Backward,
    Any,
}

impl FromStr for Direction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "forward" => Ok(Direction::Forward),
            "backward" => Ok(Direction::Backward),
            "any" => Ok(Direction::Any),
            _ => Err(format!("Invalid direction: {}", s)),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (groups, starts, ends, chrom_ids, chrom_length, clip=false, only_right=false))]
pub fn genome_bounds_numpy(
    groups: PyReadonlyArray1<u32>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
    chrom_ids: PyReadonlyArray1<u32>,
    chrom_length: PyReadonlyArray1<i64>,
    clip: bool,
    only_right: bool,
    py: Python,
) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let groups_slice = groups.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;

    let ids = chrom_ids.as_array();
    let lengths = chrom_length.as_array();

    if ids.len() != lengths.len() {
        return Err(PyValueError::new_err(
            "chrom_ids and chrom_length must have the same length",
        ));
    }

    let map = ids
        .iter()
        .zip(lengths.iter())
        .map(|(&id, &len)| (id, len))
        .collect::<HashMap<_, _>>();

    let (indices, new_starts, new_ends) = outside_bounds(
        groups_slice,
        starts_slice,
        ends_slice,
        &map,
        clip,
        only_right,
    )
    .map_err(|e| PyValueError::new_err(e))?;

    // Convert the results back to NumPy arrays.
    let indices_array = PyArray1::from_vec(py, indices);
    let new_starts_array = PyArray1::from_vec(py, new_starts);
    let new_ends_array = PyArray1::from_vec(py, new_ends);

    Ok((
        indices_array.to_owned().into(),
        new_starts_array.to_owned().into(),
        new_ends_array.to_owned().into(),
    ))
}

#[pymodule]
#[pyo3(name = "_ruranges")]
fn _ruranges(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chromsweep_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(extend_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(window_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(tile_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy, m)?)?;
    // m.add_function(wrap_pyfunction!(nearest_intervals_unique_k_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(boundary_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy, m)?)?;
    //     m.add_function(wrap_pyfunction!(subsequence_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(spliced_subsequence_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(spliced_subsequence_per_row_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(split_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(genome_bounds_numpy, m)?)?;

    //m.add_function(wrap_pyfunction!(sweep_line_overlaps_set1_polars, m)?)?;
    //m.add_function(wrap_pyfunction!(cluster_polars, m)?)?;
    //m.add_function(wrap_pyfunction!(chromsweep_polars, m)?)?;

    // m.add_function(wrap_pyfunction!(nearest_next_intervals_numpy, m)?)?;
    // m.add_function(wrap_pyfunction!(nearest_previous_intervals_numpy, m)?)?;
    Ok(())
}
