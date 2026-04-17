use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ruranges_core::group_cumsum::sweep_line_cumsum;

macro_rules! define_cumsum_numpy {
    ($fname:ident, $grp_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (groups, starts, ends, negative_strand = None, sort = true))]
        pub fn $fname(
            groups: PyReadonlyArray1<$grp_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            negative_strand: Option<PyReadonlyArray1<bool>>,
            sort: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
        )> {
            use pyo3::exceptions::PyValueError;

            let neg = negative_strand
                .ok_or_else(|| PyValueError::new_err("negative_strand is required"))?;

            let groups_s = groups.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            let neg_s = neg.as_slice()?;

            let (idxs, cumsum_starts, cumsum_ends) =
                py.allow_threads(|| sweep_line_cumsum(groups_s, starts_s, ends_s, neg_s, sort));

            Ok((
                idxs.into_pyarray(py).to_owned().into(),
                cumsum_starts.into_pyarray(py).to_owned().into(),
                cumsum_ends.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_cumsum_numpy!(group_cumsum_numpy_u32_i32, u32, i32);
define_cumsum_numpy!(group_cumsum_numpy_u32_i64, u32, i64);
