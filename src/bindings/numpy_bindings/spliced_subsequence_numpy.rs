use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use crate::spliced_subsequence::{spliced_subseq, spliced_subseq_per_row};

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


/// -------------------------------------------------------------------------
/// multi-row wrappers
/// -------------------------------------------------------------------------
macro_rules! define_spliced_subsequence_per_row_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            chrs,
            starts,
            ends,
            strand_flags,
            starts_subseq,
            ends_subseq,
            strands_subseq,
            force_plus_strand = false
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs:               PyReadonlyArray1<$chr_ty>,
            starts:             PyReadonlyArray1<$pos_ty>,
            ends:               PyReadonlyArray1<$pos_ty>,
            strand_flags:       PyReadonlyArray1<bool>,
            starts_subseq:      PyReadonlyArray1<$pos_ty>,
            ends_subseq:        PyReadonlyArray1<$pos_ty>,
            strands_subseq:     PyReadonlyArray1<bool>,
            force_plus_strand:  bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,      // indices
            Py<PyArray1<$pos_ty>>,  // new starts
            Py<PyArray1<$pos_ty>>,  // new ends
            Py<PyArray1<bool>>,     // strand
        )> {
            let (idx, new_starts, new_ends, strands) = spliced_subseq_per_row(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                strand_flags.as_slice()?,
                starts_subseq.as_slice()?,
                ends_subseq.as_slice()?,
                strands_subseq.as_slice()?,
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
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u64_i64, u64, i64);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u32_i64, u32, i64);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u32_i32, u32, i32);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u32_i16, u32, i16);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u16_i64, u16, i64);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u16_i32, u16, i32);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u16_i16, u16, i16);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u8_i64,  u8,  i64);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u8_i32,  u8,  i32);
define_spliced_subsequence_per_row_numpy!(spliced_subsequence_per_row_numpy_u8_i16,  u8,  i16);
