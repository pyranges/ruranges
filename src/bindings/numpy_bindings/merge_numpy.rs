use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::merge::sweep_line_merge;

macro_rules! define_merge_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, slack = 0))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<u32>>,
        )> {
            let chrs = chrs.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let (idx, m_starts, m_ends, counts) =
                py.allow_threads(|| sweep_line_merge(chrs, starts, ends, slack));
            Ok((
                idx.into_pyarray(py).to_owned().into(),
                m_starts.into_pyarray(py).to_owned().into(),
                m_ends.into_pyarray(py).to_owned().into(),
                counts.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_merge_numpy!(merge_numpy_u32_i32, u32, i32);
define_merge_numpy!(merge_numpy_u32_i64, u32, i64);
