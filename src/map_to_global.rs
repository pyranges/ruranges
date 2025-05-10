
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use radsort::sort_by_key;

use crate::ruranges_structs::{GroupType, PositionType, StrandInterval};


#[allow(clippy::too_many_arguments)]
pub fn map_to_global<G: GroupType, T: PositionType>(
    /* left  table (exons / annotation) */
    ex_tx:            &[G],
    ex_local_start:   &[T],
    ex_local_end:     &[T],

    /* right table (queries / local intervals) */
    q_tx:             &[G],
    q_start:          &[T],
    q_end:            &[T],

    /* extra, still in strict order */
    ex_chr_code:      &[G],
    ex_genome_start:  &[T],
    ex_genome_end:    &[T],
    ex_fwd:           &[bool],
    q_fwd:            &[bool],
) -> (Vec<u32>, Vec<T>, Vec<T>, Vec<bool>) {
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
    let mut results = Vec::new();

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
                results.push(StrandInterval {start: g_start, end: g_end, idx: idx, fwd: local_f == ex_fwd[ek]});

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

    sort_by_key(&mut results, |i| i.idx);

    let mut out_idxs    = Vec::with_capacity(results.len());
    let mut out_starts  = Vec::with_capacity(results.len());
    let mut out_ends = Vec::with_capacity(results.len());
    let mut out_strands = Vec::with_capacity(results.len());

    for rec in results {
        out_idxs.push(rec.idx);
        out_starts.push(rec.start);
        out_ends.push(rec.end);
        out_strands.push(rec.fwd);
    }

    (out_idxs, out_starts, out_ends, out_strands)
}