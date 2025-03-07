use std::error::Error;
use std::path::Path;

use rust_htslib::bam::{self, record::Cigar, Read};

use crate::ruranges_structs::GenomicData;

// Define a struct to hold the extracted BAM data.
pub type BamData = GenomicData<u32, i32>;

// This function extracts data from a BAM file and returns a BamData struct.
pub fn extract_bam_data<P: AsRef<Path>>(bam_path: P) -> Result<BamData, Box<dyn Error>> {
    // Open the BAM file for reading.
    let mut bam_reader = bam::Reader::from_path(bam_path)?;
    bam_reader.set_threads(4);

    // Initialize vectors to store the extracted data.
    let mut chroms = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut strands = Vec::new();

    // Iterate over each BAM record.
    for result in bam_reader.records() {
        let record = result?;
        // Push target id, start, end, and strand information to the vectors.
        let start = record.pos() as i32;
        chroms.push(record.tid() as u32);
        starts.push(start);
        ends.push(alignment_end(&record) as i32);
        strands.push(record.is_reverse());
    }

    // Return the data wrapped in the BamData struct.
    Ok(BamData {
        chroms: chroms,
        starts: starts,
        ends: ends,
        strands: Some(strands),
    })
}

fn alignment_end(record: &bam::Record) -> i64 {
    let pos = record.pos(); // 0-based start position
                            // Sum lengths for CIGAR operations that consume the reference.
    let ref_consuming: i64 = record
        .cigar()
        .iter()
        .map(|c| match c {
            Cigar::Match(len)
            | Cigar::Equal(len)
            | Cigar::Diff(len)
            | Cigar::Del(len)
            | Cigar::RefSkip(len) => *len as i64,
            _ => 0,
        })
        .sum();
    pos + ref_consuming
}
