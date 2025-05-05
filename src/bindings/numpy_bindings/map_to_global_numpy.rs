use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::map_to_global::map_to_global;   // core algorithm

/* =======================================================================
   Macro:  expose map_to_global_<suffix>() functions to Python/NumPy
   =======================================================================

   `_dispatch_binary("map_to_global_numpy", …)` sends the arguments in
   this order:

   (groups  starts  ends)   (groups2  starts2  ends2)
     └ left table = exons ┘   └ right table = queries ┘
   extra:  ex_chr_code  ex_genome_start  ex_genome_end  ex_fwd  q_fwd
------------------------------------------------------------------------ */
macro_rules! define_map_to_global_numpy {
    ($fname:ident, $code_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname<'py>(
            py: Python<'py>,
            /* ---------- exon (annotation) table — left side ---------- */
            ex_tx:            PyReadonlyArray1<$code_ty>,
            ex_local_start:   PyReadonlyArray1<$pos_ty>,
            ex_local_end:     PyReadonlyArray1<$pos_ty>,
            /* ---------- query (local) table — right side ------------ */
            q_tx:             PyReadonlyArray1<$code_ty>,
            q_start:          PyReadonlyArray1<$pos_ty>,
            q_end:            PyReadonlyArray1<$pos_ty>,
            /* ---------- extra parameters in Rust order -------------- */
            ex_chr_code:      PyReadonlyArray1<$code_ty>,
            ex_genome_start:  PyReadonlyArray1<$pos_ty>,
            ex_genome_end:    PyReadonlyArray1<$pos_ty>,
            ex_fwd:           PyReadonlyArray1<bool>,
            q_fwd:            PyReadonlyArray1<bool>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,      // indices back into query table
            Py<PyArray1<$pos_ty>>,  // genomic start
            Py<PyArray1<$pos_ty>>,  // genomic end
            Py<PyArray1<bool>>,     // strand (+ = True)
        )> {
            let (idx, g_start, g_end, strand) = map_to_global(
                /*  exons first (left triple)  */
                ex_tx.as_slice()?,
                ex_local_start.as_slice()?,
                ex_local_end.as_slice()?,
                /*  queries second (right triple)  */
                q_tx.as_slice()?,
                q_start.as_slice()?,
                q_end.as_slice()?,
                /*  extras in declared order  */
                ex_chr_code.as_slice()?,
                ex_genome_start.as_slice()?,
                ex_genome_end.as_slice()?,
                ex_fwd.as_slice()?,
                q_fwd.as_slice()?,
            );

            Ok((
                idx     .into_pyarray(py).to_owned().into(),
                g_start .into_pyarray(py).to_owned().into(),
                g_end   .into_pyarray(py).to_owned().into(),
                strand  .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

/* ---------------------------------------------------------------------
   Concrete instantiations – extend as required
   ------------------------------------------------------------------- */
define_map_to_global_numpy!(map_to_global_numpy_u64_i64, u64, i64);
define_map_to_global_numpy!(map_to_global_numpy_u32_i64, u32, i64);
define_map_to_global_numpy!(map_to_global_numpy_u32_i32, u32, i32);
define_map_to_global_numpy!(map_to_global_numpy_u32_i16, u32, i16);
define_map_to_global_numpy!(map_to_global_numpy_u16_i64, u16, i64);
define_map_to_global_numpy!(map_to_global_numpy_u16_i32, u16, i32);
define_map_to_global_numpy!(map_to_global_numpy_u16_i16, u16, i16);
define_map_to_global_numpy!(map_to_global_numpy_u8_i64,  u8,  i64);
define_map_to_global_numpy!(map_to_global_numpy_u8_i32,  u8,  i32);
define_map_to_global_numpy!(map_to_global_numpy_u8_i16,  u8,  i16);
