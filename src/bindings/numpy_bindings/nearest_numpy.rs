use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::nearest::nearest;

macro_rules! define_nearest_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, chrs2, starts2, ends2, slack = 0, k = 1, include_overlaps = true, direction = "any", sort_output = true))]
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
            k: usize,
            include_overlaps: bool,
            direction: &str,
            sort_output: bool,
        ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<$pos_ty>>)> {
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            let chrs2_s = chrs2.as_slice()?;
            let starts2_s = starts2.as_slice()?;
            let ends2_s = ends2.as_slice()?;

            let (idx1, idx2, dist) = py.allow_threads(|| {
                nearest(chrs_s, starts_s, ends_s, chrs2_s, starts2_s, ends2_s, slack, k, include_overlaps, direction, sort_output)
            });

            Ok((
                idx1.into_pyarray(py).to_owned().into(),
                idx2.into_pyarray(py).to_owned().into(),
                dist.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_nearest_numpy!(nearest_numpy_u32_i32, u32, i32);
define_nearest_numpy!(nearest_numpy_u32_i64, u32, i64);
