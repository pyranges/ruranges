use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ruranges_core::boundary::sweep_line_boundary;

macro_rules! define_boundary_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python<'_>,
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<u32>>,
        )> {
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;

            let (idx, b_starts, b_ends, counts) =
                py.allow_threads(|| sweep_line_boundary(chrs_s, starts_s, ends_s));

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                b_starts.into_pyarray(py).to_owned().into(),
                b_ends.into_pyarray(py).to_owned().into(),
                counts.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_boundary_numpy!(boundary_numpy_u32_i32, u32, i32);
define_boundary_numpy!(boundary_numpy_u32_i64, u32, i64);
