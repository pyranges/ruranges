// use std::str::FromStr;
// 
// use polars::prelude::*;
// use pyo3::exceptions::PyException;
// use pyo3::prelude::*;
// use pyo3_polars::PySeries;
// 
// use crate::cluster::sweep_line_cluster;
// use crate::merge::{self, sweep_line_merge};
// use crate::numpy_bindings::{keep_first_by_idx, OverlapType};
// use crate::overlaps::{self, sweep_line_overlaps_set1};
// use crate::ruranges_structs::OverlapPair;
// 
// /// Helper function to convert a PySeries into a contiguous slice of u32.
// fn pyseries_to_u32_slice(pyseries: PySeries) -> PyResult<Vec<u32>> {
//     // Access the inner Series from the PySeries tuple-struct.
//     let series = (pyseries.0).rechunk();
//     // Get the UInt32Chunked, mapping any Polars error into a PyException.
//     let ca = series
//         .u32()
//         .map_err(|e| PyException::new_err(e.to_string()))?;
//     // cont_slice() returns an Option<&[u32]>; convert it to a PyResult.
//     let slice = ca
//         .cont_slice()
//         .map_err(|e| PyException::new_err(e.to_string()))?;
//     Ok(slice.to_vec())
// }
// 
// fn pyseries_to_i32_slice(pyseries: PySeries) -> PyResult<Vec<i32>> {
//     let series = (pyseries.0).rechunk();
//     let ca = series
//         .i32()
//         .map_err(|e| PyException::new_err(e.to_string()))?;
//     let slice = ca
//         .cont_slice()
//         .map_err(|e| PyException::new_err(e.to_string()))?;
//     Ok(slice.to_vec())
// }
// 
// /// PyO3 wrapper function that accepts PySeries objects,
// /// converts them to slices using the helper functions,
// /// calls your native sweep_line_overlaps_set1, and returns the result.
// #[pyfunction]
// pub fn sweep_line_overlaps_set1_polars(
//     chrs: PySeries,
//     starts: PySeries,
//     ends: PySeries,
//     chrs2: PySeries,
//     starts2: PySeries,
//     ends2: PySeries,
//     slack: i32,
// ) -> PyResult<PySeries> {
//     let chrs_slice = pyseries_to_u32_slice(chrs)?;
//     let starts_slice = pyseries_to_i32_slice(starts)?;
//     let ends_slice = pyseries_to_i32_slice(ends)?;
//     let chrs2_slice = pyseries_to_u32_slice(chrs2)?;
//     let starts2_slice = pyseries_to_i32_slice(starts2)?;
//     let ends2_slice = pyseries_to_i32_slice(ends2)?;
// 
//     let overlaps = sweep_line_overlaps_set1(
//         &chrs_slice,
//         &starts_slice,
//         &ends_slice,
//         &chrs2_slice,
//         &starts2_slice,
//         &ends2_slice,
//         slack,
//     );
//     let out_series = Series::new("overlaps".into(), overlaps);
//     Ok(PySeries(out_series))
// }
// 
// #[pyfunction]
// pub fn cluster_polars(
//     chrs: PySeries,
//     starts: PySeries,
//     ends: PySeries,
//     slack: i32,
// ) -> PyResult<(PySeries, PySeries)> {
//     let chrs_slice = pyseries_to_u32_slice(chrs)?;
//     let starts_slice = pyseries_to_i32_slice(starts)?;
//     let ends_slice = pyseries_to_i32_slice(ends)?;
// 
//     let (cluster_ids, row_nmb) = sweep_line_cluster(&chrs_slice, &starts_slice, &ends_slice, slack);
//     let idx_series = Series::new("row_nmb".into(), row_nmb);
//     let cluster_series = Series::new("cluster_id".into(), cluster_ids);
//     Ok((PySeries(cluster_series), PySeries(idx_series)))
// }
// 
// #[pyfunction]
// pub fn chromsweep_polars(
//     _py: Python,
//     chrs: PySeries,
//     starts: PySeries,
//     ends: PySeries,
//     chrs2: PySeries,
//     starts2: PySeries,
//     ends2: PySeries,
//     slack: i32,
//     overlap_type: &str,
//     contained: bool,
// ) -> PyResult<(PySeries, PySeries)> {
//     let chrs_slice = &pyseries_to_u32_slice(chrs)?;
//     let starts_slice = &pyseries_to_i32_slice(starts)?;
//     let ends_slice = &pyseries_to_i32_slice(ends)?;
//     let chrs2_slice = &pyseries_to_u32_slice(chrs2)?;
//     let starts2_slice = &pyseries_to_i32_slice(starts2)?;
//     let ends2_slice = &pyseries_to_i32_slice(ends2)?;
// 
//     let overlap_type = OverlapType::from_str(overlap_type).unwrap();
//     let result = overlaps::sweep_line_overlaps(
//         chrs_slice,
//         starts_slice,
//         ends_slice,
//         chrs2_slice,
//         starts2_slice,
//         ends2_slice,
//         slack,
//     );
// 
//     // let result: (Vec<u32>, Vec<u32>) = if !contained {
//     //         let (sorted_starts, sorted_ends) = overlaps::compute_sorted_events(
//     //             chrs_slice,
//     //             starts_slice,
//     //             ends_slice,
//     //             slack,
//     //             invert,
//     //         );
//     //         let (sorted_starts2, sorted_ends2) =
//     //             overlaps::compute_sorted_events(chrs2_slice, starts2_slice, ends2_slice, 0, invert);
// 
//     //         let mut pairs = overlaps::sweep_line_overlaps_overlap_pair(
//     //             &sorted_starts,
//     //             &sorted_ends,
//     //             &sorted_starts2,
//     //             &sorted_ends2,
//     //         );
//     //         eprintln!("indices found: {:?}", pairs.len());
//     //         if overlap_type != OverlapType::All {
//     //             keep_first_by_idx(&mut pairs);
//     //         }
//     //         radsort::sort_by_key(&mut pairs, |p| (p.idx, p.idx2));
//     //         pairs.into_iter().map(|pair| (pair.idx, pair.idx2)).unzip()
//     //     } else {
//     //         let maxevents = overlaps::compute_sorted_maxevents(
//     //             chrs_slice,
//     //             starts_slice,
//     //             ends_slice,
//     //             chrs2_slice,
//     //             starts2_slice,
//     //             ends2_slice,
//     //             slack,
//     //             invert,
//     //         );
//     //         let mut pairs = overlaps::sweep_line_overlaps_containment(maxevents);
//     //         if overlap_type != OverlapType::All {
//     //             keep_first_by_idx(&mut pairs);
//     //         }
//     //         radsort::sort_by_key(&mut pairs, |p| (p.idx, p.idx2));
//     //         pairs.into_iter().map(|pair| (pair.idx, pair.idx2)).unzip()
//     // };
// 
//     let idx_series = Series::new("idx".into(), result.0);
//     let idx2_series = Series::new("idx2".into(), result.1);
//     Ok((PySeries(idx_series), PySeries(idx2_series)))
// }
// // #[pymodule]
// // fn ruranges(m: &Bound<'_, PyModule>) -> PyResult<()> {
// //     // Use add_wrapped in this version of pyo3
// //     Ok(())
// // }
// 