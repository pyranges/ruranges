use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::sorts;

macro_rules! define_sort_intervals_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, sort_reverse_direction = None))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            sort_reverse_direction: Option<PyReadonlyArray1<bool>>,
            py: Python<'_>,
        ) -> PyResult<Py<PyArray1<u32>>> {
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            // Extract optional slice while GIL is held.
            let rev_vec: Option<Vec<bool>> = match &sort_reverse_direction {
                Some(arr) => Some(arr.as_slice()?.to_vec()),
                None => None,
            };

            let idx = py.allow_threads(|| {
                sorts::sort_order_idx(chrs_s, starts_s, ends_s, rev_vec.as_deref())
            });

            Ok(idx.into_pyarray(py).to_owned().into())
        }
    };
}

macro_rules! define_sort_groups_numpy {
    ($fname:ident, $chr_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            py: Python<'_>,
        ) -> PyResult<Py<PyArray1<u32>>> {
            let chrs_s = chrs.as_slice()?;
            let idx = py.allow_threads(|| sorts::build_sorted_groups(chrs_s));
            Ok(idx.into_pyarray(py).to_owned().into())
        }
    };
}

define_sort_intervals_numpy!(sort_intervals_numpy_u32_i32, u32, i32);
define_sort_intervals_numpy!(sort_intervals_numpy_u32_i64, u32, i64);

define_sort_groups_numpy!(sort_groups_numpy_u32, u32);
