use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::group_cumsum::sweep_line_cumsum;

macro_rules! define_cumsum_numpy {
    ($fname:ident, $grp_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            groups,
            starts,
            ends,
            negative_strand = None,
            sort = true,
        ))]
        pub fn $fname(
            groups:          PyReadonlyArray1<$grp_ty>,
            starts:          PyReadonlyArray1<$pos_ty>,
            ends:            PyReadonlyArray1<$pos_ty>,
            negative_strand: Option<PyReadonlyArray1<bool>>,
            sort: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
        )>
        {
            use pyo3::exceptions::PyValueError;

            let neg = negative_strand
                .ok_or_else(|| PyValueError::new_err("negative_strand is required"))?;

            let (idxs, cumsum_starts, cumsum_ends) = sweep_line_cumsum(
                groups.as_slice()?, starts.as_slice()?, ends.as_slice()?,
                neg.as_slice()?, sort,
            );

            Ok((
                idxs          .into_pyarray(py).to_owned().into(),
                cumsum_starts .into_pyarray(py).to_owned().into(),
                cumsum_ends   .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_cumsum_numpy!(group_cumsum_numpy_u64_i64, u64, i64);
define_cumsum_numpy!(group_cumsum_numpy_u32_i64, u32, i64);
define_cumsum_numpy!(group_cumsum_numpy_u32_i32, u32, i32);
define_cumsum_numpy!(group_cumsum_numpy_u32_i16, u32, i16);
define_cumsum_numpy!(group_cumsum_numpy_u16_i64, u16, i64);
define_cumsum_numpy!(group_cumsum_numpy_u16_i32, u16, i32);
define_cumsum_numpy!(group_cumsum_numpy_u16_i16, u16, i16);
define_cumsum_numpy!(group_cumsum_numpy_u8_i64,  u8,  i64);
define_cumsum_numpy!(group_cumsum_numpy_u8_i32,  u8,  i32);
define_cumsum_numpy!(group_cumsum_numpy_u8_i16,  u8,  i16);