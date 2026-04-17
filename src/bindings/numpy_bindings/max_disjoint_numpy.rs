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
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;

            let idx = py.allow_threads(|| max_disjoint(chrs_s, starts_s, ends_s, slack, sort_output));

            Ok(idx.into_pyarray(py).to_owned().into())
        }
    };
}

define_max_disjoint_numpy!(max_disjoint_numpy_u32_i32, u32, i32);
define_max_disjoint_numpy!(max_disjoint_numpy_u32_i64, u32, i64);
