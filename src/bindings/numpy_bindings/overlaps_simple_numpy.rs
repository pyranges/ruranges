
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};
use crate::ruranges_structs::OverlapType;

use crate::overlaps_simple::sweep_line_overlaps; // adjust module path if needed

macro_rules! define_sweepline_numpy {
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
            contained: bool,
            no_checks: bool,
        ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>)> {
            let chrs_slice = chrs.as_slice()?;
            let starts_slice = starts.as_slice()?;
            let ends_slice = ends.as_slice()?;

            let chrs_slice2 = chrs2.as_slice()?;
            let starts_slice2 = starts2.as_slice()?;
            let ends_slice2 = ends2.as_slice()?;

            // Optional sanity checks (cheap, helps catch Python-side misuse)
            debug_assert!(chrs_slice.len() == starts_slice.len() && starts_slice.len() == ends_slice.len());
            debug_assert!(chrs_slice2.len() == starts_slice2.len() && starts_slice2.len() == ends_slice2.len());

            let (idx1_usize, idx2_usize) = sweep_line_overlaps(
                chrs_slice,
                starts_slice,
                ends_slice,
                chrs_slice2,
                starts_slice2,
                ends_slice2,
                slack,
                overlap_type,
                contained,
                no_checks,
            );

            // Convert to u32 for numpy (matching your existing API)
            debug_assert!(idx1_usize.iter().all(|&x| x <= u32::MAX as usize));
            debug_assert!(idx2_usize.iter().all(|&x| x <= u32::MAX as usize));

            let idx1: Vec<u32> = idx1_usize.into_iter().map(|x| x as u32).collect();
            let idx2: Vec<u32> = idx2_usize.into_iter().map(|x| x as u32).collect();

            Ok((
                idx1.into_pyarray(py).to_owned().into(),
                idx2.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// Same set of instantiations as your chromsweep wrapper
define_sweepline_numpy!(sweepline_numpy_u64_i64, u64, i64);
define_sweepline_numpy!(sweepline_numpy_u32_i64, u32, i64);
define_sweepline_numpy!(sweepline_numpy_u32_i32, u32, i32);
define_sweepline_numpy!(sweepline_numpy_u32_i16, u32, i16);
define_sweepline_numpy!(sweepline_numpy_u16_i64, u16, i64);
define_sweepline_numpy!(sweepline_numpy_u16_i32, u16, i32);
define_sweepline_numpy!(sweepline_numpy_u16_i16, u16, i16);
define_sweepline_numpy!(sweepline_numpy_u8_i64,  u8,  i64);
define_sweepline_numpy!(sweepline_numpy_u8_i32,  u8,  i32);
define_sweepline_numpy!(sweepline_numpy_u8_i16,  u8,  i16);

