use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;


#[allow(clippy::too_many_arguments)]
pub fn map_to_genome(
    // -------- exon (annotation) table -------------------------------------
    ex_tx: &[u32],
    ex_local_start: &[i64],
    ex_local_end: &[i64],
    ex_chr_code: &[u32],
    ex_genome_start: &[i64],
    ex_genome_end: &[i64],
    ex_fwd: &[bool],
    // -------- query (local) table -----------------------------------------
    q_tx: &[u32],
    q_start: &[i64],
    q_end: &[i64],
    q_fwd: &[bool],
) -> (
    Vec<u32>,  // out_idx
    Vec<u32>,  // out_chr_code
    Vec<i64>,  // out_start
    Vec<i64>,  // out_end
    Vec<bool>, // out_fwd
) {
    // ------------------- sanity checks (debug-only) ------------------------
    debug_assert_eq!(ex_tx.len(), ex_local_start.len());
    debug_assert_eq!(ex_tx.len(), ex_local_end.len());
    debug_assert_eq!(ex_tx.len(), ex_chr_code.len());
    debug_assert_eq!(ex_tx.len(), ex_genome_start.len());
    debug_assert_eq!(ex_tx.len(), ex_genome_end.len());
    debug_assert_eq!(ex_tx.len(), ex_fwd.len());

    debug_assert_eq!(q_tx.len(), q_start.len());
    debug_assert_eq!(q_tx.len(), q_end.len());
    debug_assert_eq!(q_tx.len(), q_fwd.len());

    // ------------------- output buffers -----------------------------------
    let mut out_idx:   Vec<u32> = Vec::new();
    let mut out_chr:   Vec<u32> = Vec::new();
    let mut out_start: Vec<i64> = Vec::new();
    let mut out_end:   Vec<i64> = Vec::new();
    let mut out_fwd:   Vec<bool> = Vec::new();

    // ------------------- two-pointer sweep ---------------------------------
    let mut ei = 0usize;                      // exon pointer
    let mut qi = 0usize;                      // query pointer
    let ex_n = ex_tx.len();
    let q_n  = q_tx.len();

    while qi < q_n {
        let tx_code = q_tx[qi];

        // move exon pointer to this transcript (or beyond)
        while ei < ex_n && ex_tx[ei] < tx_code {
            ei += 1;
        }

        // if no exons for this transcript, skip its queries
        if ei >= ex_n || ex_tx[ei] != tx_code {
            while qi < q_n && q_tx[qi] == tx_code {
                qi += 1;
            }
            continue;
        }

        // ------------------------------------------------------------
        // process all queries with transcript == tx_code
        // ------------------------------------------------------------
        let mut ej = ei;                      // exon cursor inside tx

        while qi < q_n && q_tx[qi] == tx_code {
            let mut l     = q_start[qi];
            let   lend    = q_end[qi];
            let   idx     = qi as u32;        // row number into query table
            let   local_f = q_fwd[qi];

            // advance exon cursor until its end is after l
            while ej < ex_n && ex_tx[ej] == tx_code && ex_local_end[ej] <= l {
                ej += 1;
            }

            let mut ek = ej;
            while l < lend && ek < ex_n && ex_tx[ek] == tx_code {
                let el_start = ex_local_start[ek];
                let el_end   = ex_local_end[ek];

                if l >= el_end {
                    ek += 1;
                    continue;
                }

                // clip to current exon
                let seg_end_local = if lend < el_end { lend } else { el_end };

                // translate to genome
                let offset1 = l - el_start;
                let offset2 = seg_end_local - el_start;

                let (g_start, g_end) = if ex_fwd[ek] {
                    (
                        ex_genome_start[ek] + offset1,
                        ex_genome_start[ek] + offset2,
                    )
                } else {
                    (
                        ex_genome_end[ek] - offset2,
                        ex_genome_end[ek] - offset1,
                    )
                };

                // push result
                out_idx  .push(idx);
                out_chr  .push(ex_chr_code[ek]);
                out_start.push(g_start);
                out_end  .push(g_end);
                out_fwd  .push(local_f == ex_fwd[ek]);

                // advance inside query
                l = seg_end_local;
                if l >= lend {
                    break;
                }
                ek += 1;
            }

            qi += 1;                          // next query row
        }

        // skip remaining exons of this transcript
        while ei < ex_n && ex_tx[ei] == tx_code {
            ei += 1;
        }
    }

    (out_idx, out_chr, out_start, out_end, out_fwd)
}

#[pyfunction]
#[allow(non_snake_case)]
pub fn map_to_global_numpy<'py>(
    py: Python<'py>,
    // -------- exon table --------
    ex_tx:           PyReadonlyArray1<u32>,
    ex_local_start:  PyReadonlyArray1<i64>,
    ex_local_end:    PyReadonlyArray1<i64>,
    ex_chr_code:     PyReadonlyArray1<u32>,
    ex_genome_start: PyReadonlyArray1<i64>,
    ex_genome_end:   PyReadonlyArray1<i64>,
    ex_fwd:          PyReadonlyArray1<bool>,
    // -------- query table -------
    q_tx:    PyReadonlyArray1<u32>,
    q_start: PyReadonlyArray1<i64>,
    q_end:   PyReadonlyArray1<i64>,
    q_fwd:   PyReadonlyArray1<bool>,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<bool>>,
)> {
    let (idx, chr_code, g_start, g_end, strand) = map_to_genome(
        ex_tx.as_slice()?,
        ex_local_start.as_slice()?,
        ex_local_end.as_slice()?,
        ex_chr_code.as_slice()?,
        ex_genome_start.as_slice()?,
        ex_genome_end.as_slice()?,
        ex_fwd.as_slice()?,
        q_tx.as_slice()?,
        q_start.as_slice()?,
        q_end.as_slice()?,
        q_fwd.as_slice()?,
    );

    Ok((
        idx.into_pyarray(py).to_owned().into(),
        chr_code.into_pyarray(py).to_owned().into(),
        g_start.into_pyarray(py).to_owned().into(),
        g_end.into_pyarray(py).to_owned().into(),
        strand.into_pyarray(py).to_owned().into(),
    ))
}