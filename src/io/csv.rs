use polars::{enable_string_cache, prelude::*};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

pub fn read_csv_file(
    file_path: PathBuf,
    schema: &Schema,
    columns: &Vec<usize>,
    parse_options: CsvParseOptions,
) -> Result<DataFrame, Box<dyn Error>> {
    enable_string_cache();
    Ok(CsvReadOptions::default()
        .with_has_header(false)
        .with_schema_overwrite(Some(Arc::new(schema.clone())))
        .with_projection(Some(Arc::new(columns.clone())))
        .with_rechunk(true)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?)
}

pub fn read_first_row(f: &PathBuf) -> Result<DataFrame, Box<dyn Error>> {
    let parse_options = CsvParseOptions::default().with_separator(b'\t');
    Ok(CsvReadOptions::default()
        .with_n_rows(Some(1)) // read up to `chunk_size`
        .with_has_header(false)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(f.into()))?
        .finish()?)
}

pub fn write_in_chunks(f: PathBuf, idx: &[u32]) -> Result<(), Box<dyn Error>> {
    // Sort the indices to ensure they are in increasing (global) order.
    let mut sorted_idx = idx.to_owned();
    sorted_idx.sort_unstable();
    // eprintln!("Starting to sort {:?} values", sorted_idx.len());
    // radsort::sort(sorted_idx.as_mut_slice());
    eprintln!("Done sorting");

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    // This is your chunk size for reading the CSV.
    // You may tune this to a larger value for efficiency.
    let chunk_size = 100_000;

    // We'll track the "global starting row" for each chunk.
    let mut current_start = 0_usize;

    let parse_options = CsvParseOptions::default().with_separator(b'\t');

    eprintln!("Starting to read");
    let mut reader = CsvReadOptions::default()
        .with_chunk_size(chunk_size)
        .with_has_header(false)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(f.into()))?;
    eprintln!("Done reading 1");

    let mut chunked_df = reader.batched_borrowed()?;
    eprintln!("Done reading 2");

    // This pointer (index) will walk through sorted_idx exactly once.
    let mut idx_pos = 0;
    let mut writer = CsvWriter::new(&mut handle)
        .include_header(false)
        .with_separator(b'\t');
    let mut chunk_number = 0;
    // Read the CSV file in chunks:
    while let Some(batch) = chunked_df.next_batches(1)? {
        eprintln!("Chunk number {:?}", chunk_number);
        chunk_number += 1;
        let df = &batch[0];
        let num_rows = df.height();
        if num_rows == 0 {
            // No rows read, probably end of file:
            break;
        }

        let current_end = current_start + num_rows; // global end row for this chunk

        // We gather all indices in sorted_idx that lie within [current_start, current_end).
        let mut adjusted_indices = Vec::new();
        while idx_pos < sorted_idx.len() {
            let global_index = sorted_idx[idx_pos] as usize;

            if global_index >= current_end {
                // Since sorted_idx is sorted, once we hit an index >= current_end,
                // all remaining indices are beyond this chunk.
                break;
            }
            if global_index >= current_start {
                // This means sorted_idx[idx_pos] belongs to this chunk.
                adjusted_indices.push((global_index - current_start) as u32);
            }

            // Move pointer forward.
            idx_pos += 1;
        }

        // Now adjusted_indices has all row offsets within this chunk.
        if !adjusted_indices.is_empty() {
            let chunk_ca = UInt32Chunked::from_slice("idx_chunk".into(), &adjusted_indices);
            let mut selected_df = df.take(&chunk_ca)?;

            writer.finish(&mut selected_df)?;
        }

        // Move on to the next chunk.
        current_start = current_end;

        // OPTIONAL OPTIMIZATION: If we've exhausted all indices, we can stop early.
        if idx_pos >= sorted_idx.len() {
            break; // All desired indices are processed; no need to read more chunks.
        }
    }

    Ok(())
}

// pub fn write_in_chunks(f: PathBuf, sorted_idx: &[u32]) -> Result<(), Box<dyn Error>> {
//     let stdout = std::io::stdout();
//     let mut handle = stdout.lock();
//     let chunk_size = 100_000;
//     let mut start_row = 0_usize;
//     loop {
//         let end_row = start_row + chunk_size;
//         let chunk = read_csv_file_chunk(&f, start_row, chunk_size)?;
//
//         let chunk_ca = UInt32Chunked::from_slice("idx_chunk".into(), sorted_idx);
//         let mut partial_df = chunk.take(&chunk_ca)?;
//         start_row += chunk_size;
//         if chunk.height() == 0 {
//             break;
//         }
//         CsvWriter::new(&mut handle)
//             .include_header(false)
//             .with_separator(b'\t')
//             .finish(&mut partial_df)?;
//
//     }
//     Ok(())
// }

// Create a LazyFrame from the CSV file.
// let lf = LazyCsvReader::new(f)
//     .with_has_header(true)
//     .finish()?;

// // Choose a reasonable chunk size.
// let chunk_size = 10_000;
// // Define the output file path.
// let out_path = PathBuf::from("output.csv");

// // Remove an existing output file, if any.
// let _ = std::fs::remove_file(&out_path);

// // Process the indices in chunks.
// for (i, idx_chunk) in idx.chunks(chunk_size).enumerate() {
//     // Convert the current chunk into a Series.
//     let idx_series = Series::new("idx".into(), idx_chunk);

//     // Use the lazy take operation to select the corresponding rows.
//     // If an index is repeated in idx_chunk, that row will be duplicated.
//     let df_chunk = lf.clone().take(idx_series)?.collect()?;

//     // Open the output file. Append if not the first chunk.
//     let file = OpenOptions::new()
//         .create(true)
//         .append(i > 0)
//         .open(&out_path)?;

//     // Write the current chunk to CSV.
//     // Write the header only on the first chunk.
//     CsvWriter::new(file)
//         .has_header(i == 0)
//         .finish(&df_chunk)?;
// }

// Ok(())

// loop {
//     // 2. Read a chunk of the CSV: skip `start_row`, then read up to `chunk_size`.
//     let df_chunk = CsvReader::from_path(&input_csv)?
//         .has_header(true)                 // or false, depending on your file
//         .with_skip_rows(start_row)        // skip everything up to `start_row`
//         .with_n_rows(Some(chunk_size))    // read up to `chunk_size`
//         .finish()?;

pub fn get_cat<'a>(df: &'a DataFrame, col_name: &str) -> PolarsResult<&'a [u32]> {
    df.column(col_name)?
        .categorical()
        .expect("Not categorical!")
        .physical()
        .cont_slice()
}

pub fn get_i32<'a>(df: &'a DataFrame, col_name: &str) -> PolarsResult<&'a [i32]> {
    df.column(col_name)?.i32()?.cont_slice()
}

pub fn get_bool(df: &DataFrame, col_name: &str) -> PolarsResult<Vec<bool>> {
    df.column(col_name)?
        .bool()?
        .into_iter()
        .map(|opt| opt.ok_or_else(|| PolarsError::ComputeError("Found null value".into())))
        .collect()
}

pub fn get_strand(df: &DataFrame, col_name: &str) -> PolarsResult<Vec<bool>> {
    let strand = df
        .column(col_name)?
        .categorical()
        .expect("Not categorical!");
    Ok(strand
        .iter_str() // using iter() here
        .map(|opt_val| match opt_val {
            Some("-") => true,
            Some("+") => false,
            _ => false,
        })
        .collect())
}
