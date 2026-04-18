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
            Py<PyArray1<u32>>,     // indices
            Py<PyArray1<$pos_ty>>, // boundary starts
            Py<PyArray1<$pos_ty>>, // boundary ends
            Py<PyArray1<u32>>,     // counts
        )> {
            let chrs = chrs.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let (idx, b_starts, b_ends, counts) =
                py.allow_threads(|| sweep_line_boundary(chrs, starts, ends));
            Ok((
                idx.into_pyarray(py).to_owned().into(),
                b_starts.into_pyarray(py).to_owned().into(),
                b_ends.into_pyarray(py).to_owned().into(),
                counts.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_boundary_numpy!(boundary_numpy_u32_i32, u32, i32);
define_boundary_numpy!(boundary_numpy_u32_i64, u32, i64);
