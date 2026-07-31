use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::pybacked::PyBackedStr;
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::ranks;

/// Borrow the UTF-8 buffer of every element without copying it.
///
/// `PyBackedStr` keeps the owning Python object alive and hands back a `&str`
/// pointing into its buffer, so a column's distinct values cross the boundary
/// once, as pointers, rather than being re-encoded into Rust `String`s.
fn as_str_slice(values: &[PyBackedStr]) -> Vec<&str> {
    values.iter().map(|value| &**value).collect()
}

macro_rules! define_rank_numpy {
    ($fname:ident, $core:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (values))]
        pub fn $fname(values: Vec<PyBackedStr>, py: Python<'_>) -> PyResult<Py<PyArray1<u32>>> {
            let borrowed = as_str_slice(&values);
            let ranks = py.detach(|| $core(&borrowed));
            Ok(ranks.into_pyarray(py).to_owned().into())
        }
    };
}

define_rank_numpy!(
    natural_rank_numpy,
    ranks::natural_rank,
    "Positions the values would occupy in natural order (`t2` before `t10`)."
);
define_rank_numpy!(
    lexical_rank_numpy,
    ranks::lexical_rank,
    "Positions the values would occupy in byte-lexical order (`t10` before `t9`)."
);

/// Fold one key column's codes into a running group id.
///
/// Returns the folded ids and their cardinality, so the caller can fold again.
#[pyfunction]
#[pyo3(signature = (group, group_cardinality, codes, codes_cardinality))]
pub fn fold_ranks_numpy(
    group: PyReadonlyArray1<u32>,
    group_cardinality: u32,
    codes: PyReadonlyArray1<u32>,
    codes_cardinality: u32,
    py: Python<'_>,
) -> PyResult<(Py<PyArray1<u32>>, u32)> {
    let mut folded = group.as_slice()?.to_vec();
    let codes = codes.as_slice()?;
    let cardinality = py.detach(|| {
        ranks::fold_ranks(&mut folded, group_cardinality, codes, codes_cardinality)
    });
    Ok((folded.into_pyarray(py).to_owned().into(), cardinality))
}
