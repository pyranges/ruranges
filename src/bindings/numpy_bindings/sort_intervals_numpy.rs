use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::sorts;

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
            let idx = sorts::sort_order_idx(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                match &sort_reverse_direction {
                    Some(arr) => Some(arr.as_slice()?),
                    None => None,
                },
            );
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
            let idx = sorts::build_sorted_groups(
                chrs.as_slice()?,
            );
            Ok(idx.into_pyarray(py).to_owned().into())
        }
    };
}

define_sort_intervals_numpy!(sort_intervals_numpy_u64_i64, u64, i64);
define_sort_intervals_numpy!(sort_intervals_numpy_u32_i64, u32, i64);
define_sort_intervals_numpy!(sort_intervals_numpy_u32_i32, u32, i32);
define_sort_intervals_numpy!(sort_intervals_numpy_u32_i16, u32, i16);
define_sort_intervals_numpy!(sort_intervals_numpy_u16_i64, u16, i64);
define_sort_intervals_numpy!(sort_intervals_numpy_u16_i32, u16, i32);
define_sort_intervals_numpy!(sort_intervals_numpy_u16_i16, u16, i16);
define_sort_intervals_numpy!(sort_intervals_numpy_u8_i64,  u8,  i64);
define_sort_intervals_numpy!(sort_intervals_numpy_u8_i32,  u8,  i32);
define_sort_intervals_numpy!(sort_intervals_numpy_u8_i16,  u8,  i16);

define_sort_groups_numpy!(sort_groups_numpy_u64, u64);
define_sort_groups_numpy!(sort_groups_numpy_u32, u32);
define_sort_groups_numpy!(sort_groups_numpy_u16, u16);
define_sort_groups_numpy!(sort_groups_numpy_u8,  u8);