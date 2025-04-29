use std::collections::HashMap;
use std::str::FromStr;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use bindings::numpy_bindings::overlaps_numpy::*;
use bindings::numpy_bindings::nearest_numpy::*;
use bindings::numpy_bindings::subtract_numpy::*;
use bindings::numpy_bindings::complement_overlaps_numpy::*;
use bindings::numpy_bindings::count_overlaps_numpy::*;
use bindings::numpy_bindings::sort_intervals_numpy::*;
use bindings::numpy_bindings::cluster_numpy::*;
use bindings::numpy_bindings::merge_numpy::*;
use bindings::numpy_bindings::window_numpy::*;
use bindings::numpy_bindings::tile_numpy::*;
use bindings::numpy_bindings::max_disjoint_numpy::*;
use bindings::numpy_bindings::extend_numpy::*;
use bindings::numpy_bindings::complement_numpy::*;

use crate::boundary::sweep_line_boundary;
use crate::spliced_subsequence::{spliced_subseq, spliced_subseq_per_row};
use crate::split::sweep_line_split;
use crate::{bindings, outside_bounds};

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

    let (indices, new_starts, new_ends) = outside_bounds::outside_bounds(
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
#[pyo3(name = "ruranges")]
fn ruranges(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(chromsweep_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(nearest_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(subtract_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(cluster_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(merge_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(complement_numpy_u64_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u32_i16, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u16_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u16_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u16_i16, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u8_i64, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u8_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u8_i16, m)?)?;

    m.add_function(wrap_pyfunction!(window_numpy_i64, m)?)?;
    m.add_function(wrap_pyfunction!(window_numpy_i32, m)?)?;
    m.add_function(wrap_pyfunction!(window_numpy_i16, m)?)?;

    m.add_function(wrap_pyfunction!(tile_numpy_i64, m)?)?;
    m.add_function(wrap_pyfunction!(tile_numpy_i32, m)?)?;
    m.add_function(wrap_pyfunction!(tile_numpy_i16, m)?)?;

    m.add_function(wrap_pyfunction!(extend_numpy_i64, m)?)?;
    m.add_function(wrap_pyfunction!(extend_numpy_i32, m)?)?;
    m.add_function(wrap_pyfunction!(extend_numpy_i16, m)?)?;

    m.add_function(wrap_pyfunction!(boundary_numpy, m)?)?;

    m.add_function(wrap_pyfunction!(spliced_subsequence_numpy, m)?)?;

    m.add_function(wrap_pyfunction!(spliced_subsequence_per_row_numpy, m)?)?;

    m.add_function(wrap_pyfunction!(split_numpy, m)?)?;

    m.add_function(wrap_pyfunction!(genome_bounds_numpy, m)?)?;

    Ok(())
}
