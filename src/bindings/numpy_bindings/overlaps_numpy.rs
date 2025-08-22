use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::overlaps::overlaps;


macro_rules! define_chromsweep_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python,
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            chrs2: PyReadonlyArray1<$chr_ty>,
            starts2: PyReadonlyArray1<$pos_ty>,
            ends2: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            overlap_type: &str,
            sort_output: bool,
            contained: bool,
        ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
            let chrs_slice = chrs.as_slice()?;
            let starts_slice = starts.as_slice()?;
            let ends_slice = ends.as_slice()?;
            let chrs_slice2 = chrs2.as_slice()?;
            let starts_slice2 = starts2.as_slice()?;
            let ends_slice2 = ends2.as_slice()?;

            let (idx1, idx2) = overlaps(
                chrs_slice,
                starts_slice,
                ends_slice,
                chrs_slice2,
                starts_slice2,
                ends_slice2,
                slack,
                overlap_type,
                sort_output,
                contained,
            );
            Ok((
                idx1.into_pyarray(py).to_owned().into(),
                idx2.into_pyarray(py).to_owned().into(),
            ))
        }
    }
}

define_chromsweep_numpy!(chromsweep_numpy_u64_i64, u64, i64);
define_chromsweep_numpy!(chromsweep_numpy_u32_i64, u32, i64);
define_chromsweep_numpy!(chromsweep_numpy_u32_i32, u32, i32);
define_chromsweep_numpy!(chromsweep_numpy_u32_i16, u32, i16);
define_chromsweep_numpy!(chromsweep_numpy_u16_i64, u16, i64);
define_chromsweep_numpy!(chromsweep_numpy_u16_i32, u16, i32);
define_chromsweep_numpy!(chromsweep_numpy_u16_i16, u16, i16);
define_chromsweep_numpy!(chromsweep_numpy_u8_i64,  u8,  i64);
define_chromsweep_numpy!(chromsweep_numpy_u8_i32,  u8,  i32);
define_chromsweep_numpy!(chromsweep_numpy_u8_i16,  u8,  i16);