use std::str::FromStr;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::nearest::{nearest_with_ties, Ties};

/// Parse `ties` here rather than letting the kernel's `FromStr` panic: a
/// `PanicException` is not an `Exception`, so `except Exception` cannot catch
/// it and the interpreter is aborted instead of the caller seeing an error.
fn parse_ties(value: &str) -> PyResult<Ties> {
    Ties::from_str(value)
        .map_err(|_| PyValueError::new_err(format!("ties must be 'all' or 'first'; got {value:?}")))
}

macro_rules! define_nearest_numpy {
    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            chrs,
            starts,
            ends,
            chrs2,
            starts2,
            ends2,
            slack = 0, // <$pos_ty>::from(0) at call-site
            k = 1,
            include_overlaps = true,
            direction = "any",
            sort_output = true,
            ties = "all",
        ))]
        #[allow(non_snake_case)]
        pub fn $fname(
            py: Python<'_>,
            chrs: PyReadonlyArray1<$chr_ty>,
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            chrs2: PyReadonlyArray1<$chr_ty>,
            starts2: PyReadonlyArray1<$pos_ty>,
            ends2: PyReadonlyArray1<$pos_ty>,
            slack: $pos_ty,
            k: usize,
            include_overlaps: bool,
            direction: &str,
            sort_output: bool,
            ties: &str,
        ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<$pos_ty>>)> {
            let (idx1, idx2, dist) = nearest_with_ties(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                chrs2.as_slice()?,
                starts2.as_slice()?,
                ends2.as_slice()?,
                slack,
                k,
                include_overlaps,
                direction,
                sort_output,
                parse_ties(ties)?,
            );

            Ok((
                idx1.into_pyarray(py).to_owned().into(),
                idx2.into_pyarray(py).to_owned().into(),
                dist.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_nearest_numpy!(nearest_numpy_u32_i32, u32, i32);
define_nearest_numpy!(nearest_numpy_u32_i64, u32, i64);
