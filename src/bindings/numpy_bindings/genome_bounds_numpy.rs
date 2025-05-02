
use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::outside_bounds::outside_bounds;

macro_rules! define_genome_bounds_numpy {
    ($fname:ident, $grp_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature=(
            groups,
            starts,
            ends,
            chrom_lengths,     //  <-- single vector, same length as rows
            clip = false,
            only_right = false
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            groups:         PyReadonlyArray1<$grp_ty>,
            starts:         PyReadonlyArray1<$pos_ty>,
            ends:           PyReadonlyArray1<$pos_ty>,
            chrom_lengths:  PyReadonlyArray1<$pos_ty>,
            clip:           bool,
            only_right:     bool,
            py:             Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,    // kept identical return signature
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
        )> {
            use pyo3::exceptions::PyValueError;

            // Fast length consistency check while we still hold the gil.
            let n = starts.len()?;
            if ends.len()? != n || groups.len()? != n || chrom_lengths.len()? != n {
                return Err(PyValueError::new_err(
                    "`groups`, `starts`, `ends`, and `chrom_lengths` must all have the same length",
                ));
            }

            let (idx, new_starts, new_ends) = outside_bounds(
                groups.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                chrom_lengths.as_slice()?,
                clip,
                only_right,
            )
            .map_err(PyValueError::new_err)?;

            // Convert the three Vecs back to NumPy arrays.
            Ok((
                idx.into_pyarray(py).to_owned().into(),
                new_starts.into_pyarray(py).to_owned().into(),
                new_ends.into_pyarray(py).to_owned().into(),
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