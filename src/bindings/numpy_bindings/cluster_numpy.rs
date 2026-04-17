use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::cluster::sweep_line_cluster;

macro_rules! define_cluster_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, slack = 0))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
            let chrs_s = chrs.as_slice()?;
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;

            let (cluster_ids, idx) =
                py.allow_threads(|| sweep_line_cluster(chrs_s, starts_s, ends_s, slack));

            Ok((
                cluster_ids.into_pyarray(py).to_owned().into(),
                idx.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_cluster_numpy!(cluster_numpy_u32_i32, u32, i32);
define_cluster_numpy!(cluster_numpy_u32_i64, u32, i64);
