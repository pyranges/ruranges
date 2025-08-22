use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pyfunction, Py, PyResult, Python};

use crate::tile::tile_grouped;


macro_rules! define_tile_numpy {

    ($fname:ident, $chr_ty:ty, $pos_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (chrs, starts, ends, negative_strand, tile_size))]
        pub fn $fname(
            chrs: PyReadonlyArray1<$chr_ty>,
            starts:           PyReadonlyArray1<$pos_ty>,
            ends:             PyReadonlyArray1<$pos_ty>,
            negative_strand:  PyReadonlyArray1<bool>,
            tile_size:        $pos_ty,
            py: Python<'_>,
        ) -> PyResult<(
            Py<PyArray1<usize>>,   // indices
            Py<PyArray1<$pos_ty>>, // tile starts
            Py<PyArray1<$pos_ty>>, // tile ends
            Py<PyArray1<f64>>,     // overlap fraction
        )> {
            let (t_starts, t_ends, idx, frac) = tile_grouped(
                chrs.as_slice()?,
                starts.as_slice()?,
                ends.as_slice()?,
                negative_strand.as_slice()?,
                tile_size,
            );
            Ok((
                idx     .into_pyarray(py).to_owned().into(),
                t_starts.into_pyarray(py).to_owned().into(),
                t_ends  .into_pyarray(py).to_owned().into(),
                frac    .into_pyarray(py).to_owned().into(),
            ))
        }
    };
}

// ── concrete instantiations ────────────────────────────────────────────
define_tile_numpy!(tile_numpy_u64_i64, u64, i64);
define_tile_numpy!(tile_numpy_u32_i64, u32, i64);
define_tile_numpy!(tile_numpy_u32_i32, u32, i32);
define_tile_numpy!(tile_numpy_u32_i16, u32, i16);
define_tile_numpy!(tile_numpy_u16_i64, u16, i64);
define_tile_numpy!(tile_numpy_u16_i32, u16, i32);
define_tile_numpy!(tile_numpy_u16_i16, u16, i16);
define_tile_numpy!(tile_numpy_u8_i64,  u8,  i64);
define_tile_numpy!(tile_numpy_u8_i32,  u8,  i32);
define_tile_numpy!(tile_numpy_u8_i16,  u8,  i16);