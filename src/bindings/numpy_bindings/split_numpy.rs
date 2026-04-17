use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ruranges_core::split::sweep_line_split;

macro_rules! define_split_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, slack = 0, between = false))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            between: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
        )> {
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;

            let (idx, s_starts, s_ends) =
                py.allow_threads(|| sweep_line_split(chrs_s, starts_s, ends_s, slack, between));

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                s_starts.into_pyarray(py).to_owned().into(),
                s_ends.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_split_numpy!(split_numpy_u32_i32, u32, i32);
define_split_numpy!(split_numpy_u32_i64, u32, i64);
