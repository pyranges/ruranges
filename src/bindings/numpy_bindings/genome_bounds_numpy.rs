use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyReadonlyArray1, PyArray1};

use crate::outside_bounds::outside_bounds;

macro_rules! define_genome_bounds_numpy {
    ($fname:ident, $grp_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            groups,
            starts,
            ends,
            chrom_ids,
            chrom_length,
            clip = false,
            only_right = false
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            groups:        PyReadonlyArray1<$grp_ty>,
            starts:        PyReadonlyArray1<$pos_ty>,
            ends:          PyReadonlyArray1<$pos_ty>,
            chrom_ids:     PyReadonlyArray1<$grp_ty>,
            chrom_length:  PyReadonlyArray1<$pos_ty>,
            clip:          bool,
            only_right:    bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<usize>>,    // indices
            Py<PyArray1<$pos_ty>>,  // new starts
            Py<PyArray1<$pos_ty>>,  // new ends
        )> {
            use std::collections::HashMap;
            use pyo3::exceptions::PyValueError;

            let ids  = chrom_ids.as_slice()?;
            let lens = chrom_length.as_slice()?;
            if ids.len() != lens.len() {
                return Err(PyValueError::new_err(
                    "chrom_ids and chrom_length must have the same length",
                ));
            }

            let chrom_map: HashMap<$grp_ty, $pos_ty> =
                ids.iter().copied().zip(lens.iter().copied()).collect();

            let (idx, new_starts, new_ends) = outside_bounds(
                groups.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                &chrom_map,
                clip,
                only_right,
            )
            .map_err(|e| PyValueError::new_err(e))?;

            Ok((
                PyArray1::from_vec(py, idx)        .to_owned().into(),
                PyArray1::from_vec(py, new_starts) .to_owned().into(),
                PyArray1::from_vec(py, new_ends)   .to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_genome_bounds_numpy!(genome_bounds_numpy_u64_i64, u64, i64);
define_genome_bounds_numpy!(genome_bounds_numpy_u32_i64, u32, i64);
define_genome_bounds_numpy!(genome_bounds_numpy_u32_i32, u32, i32);
define_genome_bounds_numpy!(genome_bounds_numpy_u32_i16, u32, i16);
define_genome_bounds_numpy!(genome_bounds_numpy_u16_i64, u16, i64);
define_genome_bounds_numpy!(genome_bounds_numpy_u16_i32, u16, i32);
define_genome_bounds_numpy!(genome_bounds_numpy_u16_i16, u16, i16);
define_genome_bounds_numpy!(genome_bounds_numpy_u8_i64,  u8,  i64);
define_genome_bounds_numpy!(genome_bounds_numpy_u8_i32,  u8,  i32);
define_genome_bounds_numpy!(genome_bounds_numpy_u8_i16,  u8,  i16);