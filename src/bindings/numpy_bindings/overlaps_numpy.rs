use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::overlaps::overlaps;

macro_rules! define_chromsweep_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python,
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            chrs2: PyReadonlyArray1<$chr_ty>,
            starts2: PyReadonlyArray1<$pos_ty>,
            ends2: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            overlap_type: &str,
            sort_output: bool,
            contained: bool,
        ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
            // Extract slices while GIL is held.
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            let chrs2_s = chrs2.as_slice()?;
            let starts2_s = starts2.as_slice()?;
            let ends2_s = ends2.as_slice()?;

            // Release GIL for the pure-Rust sweep.
            let (idx1, idx2) = py.allow_threads(|| {
                overlaps(
                    chrs_s, starts_s, ends_s, chrs2_s, starts2_s, ends2_s,
                    slack, overlap_type, sort_output, contained,
                )
            });

            Ok((
                idx1.into_pyarray(py).to_owned().into(),
                idx2.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_chromsweep_numpy!(chromsweep_numpy_u32_i32, u32, i32);
define_chromsweep_numpy!(chromsweep_numpy_u32_i64, u32, i64);
