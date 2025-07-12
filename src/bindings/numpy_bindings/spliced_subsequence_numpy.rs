use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::spliced_subsequence::{spliced_subseq, spliced_subseq_multi};

/// -------------------------------------------------------------------------
/// single-slice wrappers
/// -------------------------------------------------------------------------
macro_rules! define_spliced_subsequence_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            chrs,
            starts,
            ends,
            strand_flags,
            start,
            end     = None,
            force_plus_strand = false
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs:               PyReadonlyArray1<$chr_ty>,
            starts:             PyReadonlyArray1<$pos_ty>,
            ends:               PyReadonlyArray1<$pos_ty>,
            strand_flags:       PyReadonlyArray1<bool>,
            start:              $pos_ty,
            end:                Option<$pos_ty>,
            force_plus_strand:  bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,      // indices
            Py<PyArray1<$pos_ty>>,  // new starts
            Py<PyArray1<$pos_ty>>,  // new ends
            Py<PyArray1<bool>>,     // strand  True='+', False='-'
        )> {
            let (idx, new_starts, new_ends, strands) = spliced_subseq(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                strand_flags.as_slice()?,
                start,
                end,
                force_plus_strand,
            );

            Ok((
                idx        .into_pyarray(py).to_owned().into(),
                new_starts .into_pyarray(py).to_owned().into(),
                new_ends   .into_pyarray(py).to_owned().into(),
                strands    .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// concrete instantiations
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u64_i64, u64, i64);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u32_i64, u32, i64);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u32_i32, u32, i32);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u32_i16, u32, i16);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u16_i64, u16, i64);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u16_i32, u16, i32);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u16_i16, u16, i16);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u8_i64,  u8,  i64);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u8_i32,  u8,  i32);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u8_i16,  u8,  i16);

macro_rules! define_spliced_subsequence_multi_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            chrs,
            starts,
            ends,
            strand_flags,
            slice_starts,
            slice_ends,
            force_plus_strand = false
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs:            PyReadonlyArray1<$chr_ty>,
            starts:          PyReadonlyArray1<$pos_ty>,
            ends:            PyReadonlyArray1<$pos_ty>,
            strand_flags:    PyReadonlyArray1<bool>,
            slice_starts:    PyReadonlyArray1<$pos_ty>,
            slice_ends:      PyReadonlyArray1<$pos_ty>,
            force_plus_strand: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<bool>>,
        )> {
            let ends_opt: Vec<Option<$pos_ty>> = slice_ends
                .as_slice()?
                .iter()
                .map(|&v| Some(v))
                .collect();

            let (idx, new_starts, new_ends, strands) = spliced_subseq_multi(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                strand_flags.as_slice()?,
                slice_starts.as_slice()?,
                ends_opt.as_slice(),
                force_plus_strand,
            );

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                new_starts.into_pyarray(py).to_owned().into(),
                new_ends.into_pyarray(py).to_owned().into(),
                strands.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}


// concrete instantiations
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u64_i64, u64, i64);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u32_i64, u32, i64);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u32_i32, u32, i32);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u32_i16, u32, i16);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u16_i64, u16, i64);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u16_i32, u16, i32);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u16_i16, u16, i16);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u8_i64,  u8,  i64);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u8_i32,  u8,  i32);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u8_i16,  u8,  i16);
