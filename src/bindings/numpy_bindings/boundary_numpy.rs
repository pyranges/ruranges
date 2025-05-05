use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::boundary::sweep_line_boundary;

macro_rules! define_boundary_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python<'_>,
            chrs:   PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends:   PyReadonlyArray1<$pos_ty>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,      // indices
            Py<PyArray1<$pos_ty>>,  // boundary starts
            Py<PyArray1<$pos_ty>>,  // boundary ends
            Py<PyArray1<u32>>,      // counts
        )> {
            let (idx, b_starts, b_ends, counts) = sweep_line_boundary(
                chrs.as_slice()?, starts.as_slice()?, ends.as_slice()?,
            );
            Ok((
                idx     .into_pyarray(py).to_owned().into(),
                b_starts.into_pyarray(py).to_owned().into(),
                b_ends  .into_pyarray(py).to_owned().into(),
                counts  .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_boundary_numpy!(boundary_numpy_u64_i64, u64, i64);
define_boundary_numpy!(boundary_numpy_u32_i64, u32, i64);
define_boundary_numpy!(boundary_numpy_u32_i32, u32, i32);
define_boundary_numpy!(boundary_numpy_u32_i16, u32, i16);
define_boundary_numpy!(boundary_numpy_u16_i64, u16, i64);
define_boundary_numpy!(boundary_numpy_u16_i32, u16, i32);
define_boundary_numpy!(boundary_numpy_u16_i16, u16, i16);
define_boundary_numpy!(boundary_numpy_u8_i64,  u8,  i64);
define_boundary_numpy!(boundary_numpy_u8_i32,  u8,  i32);
define_boundary_numpy!(boundary_numpy_u8_i16,  u8,  i16);