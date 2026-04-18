use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ruranges_core::spliced_subsequence::{spliced_subseq, spliced_subseq_multi};

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
            end = None,
            force_plus_strand = false,
            sort_output = true,
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            strand_flags: PyReadonlyArray1<bool>,
            start: $pos_ty,
            end: Option<$pos_ty>,
            force_plus_strand: bool,
            sort_output: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,     // indices
            Py<PyArray1<$pos_ty>>, // new starts
            Py<PyArray1<$pos_ty>>, // new ends
            Py<PyArray1<bool>>,    // strand  True='+', False='-'
        )> {
            let chrs = chrs.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let strand_flags = strand_flags.as_slice()?;
            let (idx, new_starts, new_ends, strands) = py.allow_threads(|| {
                spliced_subseq(chrs, starts, ends, strand_flags, start, end, force_plus_strand, sort_output)
            });

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
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u32_i32, u32, i32);
define_spliced_subsequence_numpy!(spliced_subsequence_numpy_u32_i64, u32, i64);

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
            force_plus_strand = false,
            sort_output = true,
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            strand_flags: PyReadonlyArray1<bool>,
            slice_starts: PyReadonlyArray1<$pos_ty>,
            slice_ends: PyReadonlyArray1<$pos_ty>,
            force_plus_strand: bool,
            sort_output: bool,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<u32>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<bool>>,
        )> {
            let chrs = chrs.as_slice()?;
            let starts = starts.as_slice()?;
            let ends = ends.as_slice()?;
            let strand_flags = strand_flags.as_slice()?;
            let slice_starts = slice_starts.as_slice()?;
            let ends_opt: Vec<Option<$pos_ty>> =
                slice_ends.as_slice()?.iter().map(|&v| Some(v)).collect();
            let (idx, new_starts, new_ends, strands) = py.allow_threads(|| {
                spliced_subseq_multi(chrs, starts, ends, strand_flags, slice_starts, ends_opt.as_slice(), force_plus_strand, sort_output)
            });

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
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u32_i32, u32, i32);
define_spliced_subsequence_multi_numpy!(spliced_subsequence_multi_numpy_u32_i64, u32, i64);
