use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ruranges_core::map_to_global::map_to_global;

macro_rules! define_map_to_global_numpy {
    ($fname:ident, $code_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            ex_tx, ex_local_start, ex_local_end,
            q_tx, q_start, q_end,
            ex_chr_code, ex_genome_start, ex_genome_end, ex_fwd, q_fwd,
            sort_output = true,
        ))]
        #[allow(non_snake_case)]
        pub fn $fname<'py>(
            py: Python<'py>,
            ex_tx: PyReadonlyArray1<$code_ty>,
            ex_local_start: PyReadonlyArray1<$pos_ty>,
            ex_local_end: PyReadonlyArray1<$pos_ty>,
            q_tx: PyReadonlyArray1<$code_ty>,
            q_start: PyReadonlyArray1<$pos_ty>,
            q_end: PyReadonlyArray1<$pos_ty>,
            ex_chr_code: PyReadonlyArray1<$code_ty>,
            ex_genome_start: PyReadonlyArray1<$pos_ty>,
            ex_genome_end: PyReadonlyArray1<$pos_ty>,
            ex_fwd: PyReadonlyArray1<bool>,
            q_fwd: PyReadonlyArray1<bool>,
            sort_output: bool,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<bool>>,
        )> {
            let ex_tx_s = ex_tx.as_slice()?;
            let ex_local_start_s = ex_local_start.as_slice()?;
            let ex_local_end_s = ex_local_end.as_slice()?;
            let q_tx_s = q_tx.as_slice()?;
            let q_start_s = q_start.as_slice()?;
            let q_end_s = q_end.as_slice()?;
            let ex_chr_code_s = ex_chr_code.as_slice()?;
            let ex_genome_start_s = ex_genome_start.as_slice()?;
            let ex_genome_end_s = ex_genome_end.as_slice()?;
            let ex_fwd_s = ex_fwd.as_slice()?;
            let q_fwd_s = q_fwd.as_slice()?;

            let (idx, g_start, g_end, strand) = py.allow_threads(|| {
                map_to_global(
                    ex_tx_s, ex_local_start_s, ex_local_end_s,
                    q_tx_s, q_start_s, q_end_s,
                    ex_chr_code_s, ex_genome_start_s, ex_genome_end_s,
                    ex_fwd_s, q_fwd_s,
                    sort_output,
                )
            });

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                g_start.into_pyarray(py).to_owned().into(),
                g_end.into_pyarray(py).to_owned().into(),
                strand.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_map_to_global_numpy!(map_to_global_numpy_u32_i32, u32, i32);
define_map_to_global_numpy!(map_to_global_numpy_u32_i64, u32, i64);
