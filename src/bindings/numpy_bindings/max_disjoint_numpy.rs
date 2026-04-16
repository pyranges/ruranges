use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::max_disjoint::max_disjoint;

macro_rules! define_max_disjoint_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, slack = 0, sort_output = true))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            sort_output: bool,
            py: Python<'_>,
        ) -> PyResult<Py<PyArray1<u32>>> {
            let idx = max_disjoint(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                slack,
                sort_output,
            );
            Ok(idx.into_pyarray(py).to_owned().into())
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_max_disjoint_numpy!(max_disjoint_numpy_u32_i32, u32, i32);
define_max_disjoint_numpy!(max_disjoint_numpy_u32_i64, u32, i64);
