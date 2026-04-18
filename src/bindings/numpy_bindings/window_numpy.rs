use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::tile::window_grouped;

macro_rules! define_window_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, negative_strand, window_size))]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            negative_strand: PyReadonlyArray1<bool>,
            window_size: $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<usize>>,   // indices
            Py<PyArray1<$pos_ty>>, // windowed starts
            Py<PyArray1<$pos_ty>>, // windowed ends
        )> {
            let chrs = chrs.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let negative_strand = negative_strand.as_slice()?;
            // NB: backend returns (starts, ends, indices)
            let (w_starts, w_ends, idx) =
                py.allow_threads(|| window_grouped(chrs, starts, ends, negative_strand, window_size));

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                w_starts.into_pyarray(py).to_owned().into(),
                w_ends.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_window_numpy!(window_numpy_u32_i32, u32, i32);
define_window_numpy!(window_numpy_u32_i64, u32, i64);
