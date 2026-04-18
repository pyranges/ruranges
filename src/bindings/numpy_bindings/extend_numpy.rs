use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ruranges_core::extend;

macro_rules! define_extend_numpy {
    ($fname:ident, $grp_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
                                            groups,
                                            starts,
                                            ends,
                                            negative_strand,      // optional (Python requires a default)
                                            ext_3,
                                            ext_5
                                        ))]
        pub fn $fname(
            groups: PyReadonlyArray1<$grp_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            negative_strand: PyReadonlyArray1<bool>,
            ext_3: $pos_ty,
            ext_5: $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(Py<PyArray1<$pos_ty>>, Py<PyArray1<$pos_ty>>)> {
            let groups = groups.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let negative_strand = negative_strand.as_slice()?;
            let (new_starts, new_ends) =
                py.allow_threads(|| extend::extend_grp(groups, starts, ends, negative_strand, ext_3, ext_5));

            Ok((
                new_starts.into_pyarray(py).to_owned().into(),
                new_ends.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_extend_numpy!(extend_numpy_u32_i32, u32, i32);
define_extend_numpy!(extend_numpy_u32_i64, u32, i64);
