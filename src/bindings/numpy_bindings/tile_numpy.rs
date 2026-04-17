use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use ruranges_core::tile::tile;

macro_rules! define_tile_numpy {
    ($fname:ident, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (starts, ends, negative_strand, tile_size))]
        pub fn $fname(
            starts: PyReadonlyArray1<$pos_ty>,
            ends: PyReadonlyArray1<$pos_ty>,
            negative_strand: PyReadonlyArray1<bool>,
            tile_size: $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<usize>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<$pos_ty>>,
            Py<PyArray1<f64>>,
        )> {
            let starts_s = starts.as_slice()?;
            let ends_s = ends.as_slice()?;
            let neg_s = negative_strand.as_slice()?;

            let (t_starts, t_ends, idx, frac) =
                py.allow_threads(|| tile(starts_s, ends_s, neg_s, tile_size));

            Ok((
                idx.into_pyarray(py).to_owned().into(),
                t_starts.into_pyarray(py).to_owned().into(),
                t_ends.into_pyarray(py).to_owned().into(),
                frac.into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

define_tile_numpy!(tile_numpy_i32, i32);
define_tile_numpy!(tile_numpy_i64, i64);
