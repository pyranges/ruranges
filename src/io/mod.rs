use std::{error::Error, path::PathBuf};

use bed::read_bed_file;
use csv::write_in_chunks;
use gtf::read_gtf_file;
use polars::error::PolarsResult;

use crate::ruranges_structs::{GenomicData, GroupType, PositionType};

pub mod bam;
pub mod bed;
pub mod csv;
pub mod gtf;

pub fn read_genomics_file(f: &PathBuf) -> Result<GenomicData<u32, i32>, Box<dyn Error>> {
    match f.extension().and_then(|s| s.to_str()) {
        Some("bed") => read_bed_file(f.to_path_buf()),
        Some("gtf") => read_gtf_file(f.to_path_buf()),
        _ => panic!("No valid extension found!"),
    }
}

pub fn write_genomics_file(f: PathBuf, indices: &[u32]) -> Result<(), Box<dyn Error>> {
    match f.extension().and_then(|s| s.to_str()) {
        Some("bed") | Some("gtf") => write_in_chunks(f, indices),
        _ => panic!("No writer forund for files of type {:?}!", f),
    }
}
