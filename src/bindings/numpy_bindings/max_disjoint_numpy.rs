use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::max_disjoint::max_disjoint;

macro_rules! define_max_disjoint_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, slack = 0))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs:   PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends:   PyReadonlyArray1<$pos_ty>,
            slack:  $pos_ty,
            py: Python<'_>,
        ) -> PyResult<Py<PyArray1<u32>>> {
            let idx = max_disjoint(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                slack,
            );
            Ok(idx.into_pyarray(py).to_owned().into())
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_max_disjoint_numpy!(max_disjoint_numpy_u64_i64, u64, i64);
define_max_disjoint_numpy!(max_disjoint_numpy_u32_i64, u32, i64);
define_max_disjoint_numpy!(max_disjoint_numpy_u32_i32, u32, i32);
define_max_disjoint_numpy!(max_disjoint_numpy_u32_i16, u32, i16);
define_max_disjoint_numpy!(max_disjoint_numpy_u16_i64, u16, i64);
define_max_disjoint_numpy!(max_disjoint_numpy_u16_i32, u16, i32);
define_max_disjoint_numpy!(max_disjoint_numpy_u16_i16, u16, i16);
define_max_disjoint_numpy!(max_disjoint_numpy_u8_i64,  u8,  i64);
define_max_disjoint_numpy!(max_disjoint_numpy_u8_i32,  u8,  i32);
define_max_disjoint_numpy!(max_disjoint_numpy_u8_i16,  u8,  i16);