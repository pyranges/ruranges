use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::overlaps::count_overlaps;

macro_rules! define_count_overlaps_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python<'_>,
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            chrs2: PyReadonlyArray1<$chr_ty>,
            starts2: PyReadonlyArray1<$pos_ty>,
            ends2: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
        ) -> PyResult<Py<PyArray1<u32>>> {
            let chrs = chrs.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let chrs2 = chrs2.as_slice()?;
            let starts2 = starts2.as_slice()?;
            let ends2 = ends2.as_slice()?;
            let counts = py.allow_threads(|| {
                count_overlaps(chrs, starts, ends, chrs2, starts2, ends2, slack)
            });
            Ok(counts.into_pyarray(py).to_owned().into())
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i32, u32, i32);
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i64, u32, i64);
