use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::extend;

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
            groups:           PyReadonlyArray1<$grp_ty>,
            starts:           PyReadonlyArray1<$pos_ty>,
            ends:             PyReadonlyArray1<$pos_ty>,
            negative_strand:  PyReadonlyArray1<bool>,
            ext_3: $pos_ty,
            ext_5: $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(Py<PyArray1<$pos_ty>>, Py<PyArray1<$pos_ty>>)> {

            let (new_starts, new_ends) = extend::extend_grp(
                    groups.as_slice()?, starts.as_slice()?, ends.as_slice()?,
                    negative_strand.as_slice()?, ext_3, ext_5,
                );

            Ok((
                new_starts.into_pyarray(py).to_owned().into(),
                new_ends  .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_extend_numpy!(extend_numpy_u64_i64, u64, i64);
define_extend_numpy!(extend_numpy_u32_i64, u32, i64);
define_extend_numpy!(extend_numpy_u32_i32, u32, i32);
define_extend_numpy!(extend_numpy_u32_i16, u32, i16);
define_extend_numpy!(extend_numpy_u16_i64, u16, i64);
define_extend_numpy!(extend_numpy_u16_i32, u16, i32);
define_extend_numpy!(extend_numpy_u16_i16, u16, i16);
define_extend_numpy!(extend_numpy_u8_i64,  u8,  i64);
define_extend_numpy!(extend_numpy_u8_i32,  u8,  i32);
define_extend_numpy!(extend_numpy_u8_i16,  u8,  i16);