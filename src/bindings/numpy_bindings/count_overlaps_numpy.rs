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
            // Extract slices while GIL is held.
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            let chrs2_s = chrs2.as_slice()?;
            let starts2_s = starts2.as_slice()?;
            let ends2_s = ends2.as_slice()?;

            // Release GIL for the pure-Rust sweep.
            let counts = py.allow_threads(|| {
                count_overlaps(chrs_s, starts_s, ends_s, chrs2_s, starts2_s, ends2_s, slack)
            });

            Ok(counts.into_pyarray(py).to_owned().into())
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i32, u32, i32);
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i64, u32, i64);
