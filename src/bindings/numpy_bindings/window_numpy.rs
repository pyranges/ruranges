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
            Py<PyArray1<usize>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
        )> {
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            let neg_s = negative_strand.as_slice()?;

            let (w_starts, w_ends, idx) =
                py.allow_threads(|| window_grouped(chrs_s, starts_s, ends_s, neg_s, window_size));

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                w_starts.into_pyarray(py).to_owned().into(),
                w_ends.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_window_numpy!(window_numpy_u32_i32, u32, i32);
define_window_numpy!(window_numpy_u32_i64, u32, i64);
