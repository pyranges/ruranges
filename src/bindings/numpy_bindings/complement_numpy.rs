use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};
use rustc_hash::FxHashMap;

use crate::complement_single::sweep_line_complement;

macro_rules! define_complement_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            groups,
            starts,
            ends,
            chrom_len_ids,
            chrom_lens,
            slack     = 0,
            include_first_interval = false
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python<'_>,
            groups: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            chrom_len_ids: PyReadonlyArray1<$chr_ty>,
            chrom_lens: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            include_first_interval: bool,
        ) -> PyResult<(
            Py<PyArray1<$chr_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<u32>>,
        )> {
            let keys = chrom_len_ids.as_slice()?;
            let vals = chrom_lens.as_slice()?;
            if keys.len() != vals.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "chrom_len_ids and chrom_lens must have identical length",
                ));
            }

            let mut lens_map: FxHashMap<$chr_ty, $pos_ty> =
                FxHashMap::with_capacity_and_hasher(keys.len(), Default::default());
            for (&k, &v) in keys.iter().zip(vals.iter()) {
                lens_map.insert(k, v);
            }

            let (out_chrs, out_starts, out_ends, out_idx) = sweep_line_complement(
                groups.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                slack,
                &lens_map,
                include_first_interval,
            );

            Ok((
                out_chrs  .into_pyarray(py).to_owned().into(),
                out_starts.into_pyarray(py).to_owned().into(),
                out_ends  .into_pyarray(py).to_owned().into(),
                out_idx   .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ───────────────────────────────────────────
define_complement_numpy!(complement_numpy_u64_i64, u64, i64);
define_complement_numpy!(complement_numpy_u32_i64, u32, i64);
define_complement_numpy!(complement_numpy_u32_i32, u32, i32);
define_complement_numpy!(complement_numpy_u32_i16, u32, i16);
define_complement_numpy!(complement_numpy_u16_i64, u16, i64);
define_complement_numpy!(complement_numpy_u16_i32, u16, i32);
define_complement_numpy!(complement_numpy_u16_i16, u16, i16);
define_complement_numpy!(complement_numpy_u8_i64,  u8,  i64);
define_complement_numpy!(complement_numpy_u8_i32,  u8,  i32);
define_complement_numpy!(complement_numpy_u8_i16,  u8,  i16);