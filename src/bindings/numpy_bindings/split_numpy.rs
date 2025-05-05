use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::split::sweep_line_split;

macro_rules! define_split_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, slack = 0, between = false))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs:   PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends:   PyReadonlyArray1<$pos_ty>,
            slack:  $pos_ty,
            between: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,      // indices
            Py<PyArray1<$pos_ty>>,  // split starts
            Py<PyArray1<$pos_ty>>,  // split ends
        )> {
            let (idx, s_starts, s_ends) = sweep_line_split(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                slack,
                between,
            );
            Ok((
                idx      .into_pyarray(py).to_owned().into(),
                s_starts .into_pyarray(py).to_owned().into(),
                s_ends   .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_split_numpy!(split_numpy_u64_i64, u64, i64);
define_split_numpy!(split_numpy_u32_i64, u32, i64);
define_split_numpy!(split_numpy_u32_i32, u32, i32);
define_split_numpy!(split_numpy_u32_i16, u32, i16);
define_split_numpy!(split_numpy_u16_i64, u16, i64);
define_split_numpy!(split_numpy_u16_i32, u16, i32);
define_split_numpy!(split_numpy_u16_i16, u16, i16);
define_split_numpy!(split_numpy_u8_i64,  u8,  i64);
define_split_numpy!(split_numpy_u8_i32,  u8,  i32);
define_split_numpy!(split_numpy_u8_i16,  u8,  i16);