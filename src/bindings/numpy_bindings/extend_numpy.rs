use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::extend;

/// Generate a `extend_numpy_<int>` wrapper for any signed integer type.
macro_rules! define_extend_numpy {
    ($fname:ident, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            groups,
            starts,
            ends,
            negative_strand,
            ext      = None,
            ext_3    = None,
            ext_5    = None
        ))]
        pub fn $fname(
            groups: Option<PyReadonlyArray1<u32>>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends:   PyReadonlyArray1<$pos_ty>,
            negative_strand: PyReadonlyArray1<bool>,
            ext:   Option<$pos_ty>,
            ext_3: Option<$pos_ty>,
            ext_5: Option<$pos_ty>,
            py:    Python<'_>,
        ) -> PyResult<(Py<PyArray1<$pos_ty>>, Py<PyArray1<$pos_ty>>)> {
            // choose grouped / un-grouped implementation
            let (starts, ends) = match groups {
                Some(groups) => extend::extend_grp(
                    groups.as_slice()?,
                    starts.as_slice()?,
                    ends.as_slice()?,
                    negative_strand.as_slice()?,
                    ext, ext_3, ext_5,
                ),
                None => extend::extend(
                    starts.as_slice()?,
                    ends.as_slice()?,
                    negative_strand.as_slice()?,
                    ext, ext_3, ext_5,
                ),
            };

            Ok((
                starts.into_pyarray(py).to_owned().into(),
                ends  .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ───────────────────────────
define_extend_numpy!(extend_numpy_i64, i64);
define_extend_numpy!(extend_numpy_i32, i32);
define_extend_numpy!(extend_numpy_i16, i16);