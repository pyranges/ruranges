use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::tile::window;

macro_rules! define_window_numpy {
    ($fname:ident, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (starts, ends, negative_strand, window_size))]
        pub fn $fname(
            starts:          PyReadonlyArray1<$pos_ty>,
            ends:            PyReadonlyArray1<$pos_ty>,
            negative_strand: PyReadonlyArray1<bool>,
            window_size:     $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<usize>>,   // indices
            Py<PyArray1<$pos_ty>>, // windowed starts
            Py<PyArray1<$pos_ty>>, // windowed ends
        )> {
            // NB: backend returns (starts, ends, indices)
            let (w_starts, w_ends, idx) = window(
                starts.as_slice()?,
                ends.as_slice()?,
                negative_strand.as_slice()?,
                window_size,
            );

            Ok((
                idx      .into_pyarray(py).to_owned().into(),
                w_starts .into_pyarray(py).to_owned().into(),
                w_ends   .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_window_numpy!(window_numpy_i64, i64);
define_window_numpy!(window_numpy_i32, i32);
define_window_numpy!(window_numpy_i16, i16);