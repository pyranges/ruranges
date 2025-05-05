use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::overlaps::count_overlaps;

macro_rules! define_count_overlaps_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python<'_>,
            chrs:    PyReadonlyArray1<$chr_ty>,
            starts:  PyReadonlyArray1<$pos_ty>,
            ends:    PyReadonlyArray1<$pos_ty>,
            chrs2:   PyReadonlyArray1<$chr_ty>,
            starts2: PyReadonlyArray1<$pos_ty>,
            ends2:   PyReadonlyArray1<$pos_ty>,
            slack:   $pos_ty,
        ) -> PyResult<Py<PyArray1<u32>>> {
            let counts = count_overlaps(
                chrs.as_slice()?,   starts.as_slice()?,   ends.as_slice()?,
                chrs2.as_slice()?,  starts2.as_slice()?,  ends2.as_slice()?,
                slack,
            );
            Ok(counts.into_pyarray(py).to_owned().into())
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_count_overlaps_numpy!(count_overlaps_numpy_u64_i64, u64, i64);
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i64, u32, i64);
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i32, u32, i32);
define_count_overlaps_numpy!(count_overlaps_numpy_u32_i16, u32, i16);
define_count_overlaps_numpy!(count_overlaps_numpy_u16_i64, u16, i64);
define_count_overlaps_numpy!(count_overlaps_numpy_u16_i32, u16, i32);
define_count_overlaps_numpy!(count_overlaps_numpy_u16_i16, u16, i16);
define_count_overlaps_numpy!(count_overlaps_numpy_u8_i64,  u8,  i64);
define_count_overlaps_numpy!(count_overlaps_numpy_u8_i32,  u8,  i32);
define_count_overlaps_numpy!(count_overlaps_numpy_u8_i16,  u8,  i16);