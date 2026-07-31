#[cfg(not(feature = "backend-numpy"))]
compile_error!("ruranges-py requires the `backend-numpy` feature.");

#[cfg(feature = "backend-numpy")]
pub mod numpy_bindings;

use std::collections::HashMap;
use std::str::FromStr;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::numpy_bindings::boundary_numpy::*;
use crate::numpy_bindings::cluster_numpy::*;
use crate::numpy_bindings::complement_numpy::*;
use crate::numpy_bindings::complement_overlaps_numpy::*;
use crate::numpy_bindings::count_overlaps_numpy::*;
use crate::numpy_bindings::extend_numpy::*;
use crate::numpy_bindings::genome_bounds_numpy::*;
use crate::numpy_bindings::group_cumsum_numpy::*;
use crate::numpy_bindings::map_to_global_numpy::*;
use crate::numpy_bindings::max_disjoint_numpy::*;
use crate::numpy_bindings::merge_numpy::*;
use crate::numpy_bindings::nearest_numpy::*;
use crate::numpy_bindings::overlaps_numpy::*;
use crate::numpy_bindings::ranks_numpy::*;
use crate::numpy_bindings::sort_intervals_numpy::*;
use crate::numpy_bindings::spliced_subsequence_numpy::*;
use crate::numpy_bindings::split_numpy::*;
use crate::numpy_bindings::subtract_numpy::*;
use crate::numpy_bindings::tile_numpy::*;
use crate::numpy_bindings::window_numpy::*;

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

#[pymodule]
#[pyo3(name = "ruranges")]
fn ruranges(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(map_to_global_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_global_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(chromsweep_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(chromsweep_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(nearest_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(subtract_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_overlaps_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(natural_rank_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(lexical_rank_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fold_ranks_numpy, m)?)?;

    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(sort_intervals_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_groups_numpy_u32, m)?)?;

    m.add_function(wrap_pyfunction!(cluster_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(merge_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(merge_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(complement_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(complement_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(window_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(window_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(tile_numpy_i32, m)?)?;
    m.add_function(wrap_pyfunction!(tile_numpy_i64, m)?)?;

    m.add_function(wrap_pyfunction!(boundary_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(boundary_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(spliced_subsequence_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(spliced_subsequence_numpy_u32_i64, m)?)?;
    m.add_function(wrap_pyfunction!(
        spliced_subsequence_multi_numpy_u32_i32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spliced_subsequence_multi_numpy_u32_i64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(extend_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(extend_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(split_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(split_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(genome_bounds_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(genome_bounds_numpy_u32_i64, m)?)?;

    m.add_function(wrap_pyfunction!(group_cumsum_numpy_u32_i32, m)?)?;
    m.add_function(wrap_pyfunction!(group_cumsum_numpy_u32_i64, m)?)?;

    Ok(())
}
