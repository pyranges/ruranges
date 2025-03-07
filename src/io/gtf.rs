use std::{error::Error, path::PathBuf, sync::Arc};

use polars::{
    frame::DataFrame,
    prelude::{CategoricalOrdering, CsvParseOptions, DataType, Field, Schema},
};

use crate::ruranges_structs::GenomicData;

use super::csv::{get_bool, get_cat, get_i32, get_strand};

pub type GtfData = GenomicData<u32, i32>;

pub fn read_gtf_file(f: PathBuf) -> Result<GtfData, Box<dyn Error>> {
    // Build a schema from the fields.
    let columns = vec![0, 3, 4, 6];
    let parse_options = CsvParseOptions::default().with_separator(b'\t');

    // Open the first CSV file and create a CsvReader
    let cat = DataType::Categorical(None, CategoricalOrdering::Physical);
    let fields = vec![
        Field::new("Chromosome".into(), cat.clone()), // Column 0
        Field::new("Source".into(), cat.clone()),     // Column 1
        Field::new("Feature".into(), cat.clone()),    // Column 2
        Field::new("Start".into(), DataType::Int32),  // Column 3
        Field::new("End".into(), DataType::Int32),    // Column 4
        Field::new("Score".into(), DataType::Float64), // Column 5
        Field::new("Strand".into(), cat.clone()),     // Column 6
        Field::new("Frame".into(), cat.clone()),      // Column 7
        Field::new("Attribute".into(), cat),          // Column 8
    ];

    let schema = Schema::from_iter(fields);
    let arc_columns = Arc::new(columns);
    let csv = super::csv::read_csv_file(f, &schema, &arc_columns, parse_options)?;
    Ok(df_to_gtf_data(&csv, true)?)
}

fn df_to_gtf_data(csv: &DataFrame, include_strand: bool) -> Result<GtfData, Box<dyn Error>> {
    Ok(GtfData {
        chroms: get_cat(&csv, "Chromosome")?.to_owned(),
        starts: get_i32(&csv, "Start")?.to_owned(),
        ends: get_i32(&csv, "End")?.to_owned(),
        strands: if include_strand {
            Some(get_strand(&csv, "Strand")?.to_owned())
        } else {
            None
        },
    })
}
