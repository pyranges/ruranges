use std::{error::Error, io::BufReader, path::PathBuf};

use crate::ruranges_structs::{GenomicData, GroupType, PositionType};
use polars::prelude::*;

use super::csv::{get_bool, get_cat, get_i32, read_first_row};

pub type BedData = GenomicData<u32, i32>;

pub fn read_bed_file(f: PathBuf) -> Result<BedData, Box<dyn Error>> {
    // Build a schema from the fields.
    let mut columns = vec![0, 1, 2];
    let parse_options = CsvParseOptions::default().with_separator(b'\t');

    // Open the first CSV file and create a CsvReader
    let mut fields = vec![
        Field::new(
            "Chromosome".into(),
            DataType::Categorical(None, CategoricalOrdering::Physical),
        ),
        Field::new("Start".into(), DataType::Int32),
        Field::new("End".into(), DataType::Int32),
    ];
    let include_strand = includes_strand(&f)?;
    if include_strand {
        columns.push(5);
        fields.push(Field::new("Strand".into(), DataType::Int32));
    }

    let schema = Schema::from_iter(fields);
    let arc_columns = Arc::new(columns);
    let csv = super::csv::read_csv_file(f, &schema, &arc_columns, parse_options)?;
    Ok(df_to_bed_data(&csv, include_strand)?)
}

fn includes_strand(f: &PathBuf) -> Result<bool, Box<dyn Error>> {
    let df = read_first_row(&f)?;
    Ok(df.width() >= 6)
}

fn df_to_bed_data(csv: &DataFrame, include_strand: bool) -> Result<BedData, Box<dyn Error>> {
    Ok(BedData {
        chroms: get_cat(&csv, "Chromosome")?.to_owned(),
        starts: get_i32(&csv, "Start")?.to_owned(),
        ends: get_i32(&csv, "End")?.to_owned(),
        strands: if include_strand {
            Some(get_bool(&csv, "Strand")?.to_owned())
        } else {
            None
        },
    })
}
