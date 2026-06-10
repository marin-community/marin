//! Arrow IPC wire path.
//!
//! `decode_one_record_batch`: a `WriteRows` request carries exactly one
//! RecordBatch in one IPC *stream*; zero or more-than-one batches are a contract
//! violation. `encode_ipc` is the Query result encoder.

use std::io::Cursor;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;

use crate::errors::StatsError;

/// Decode an inbound WriteRows IPC stream into exactly one `RecordBatch`.
///
/// A stream carrying zero or more than one batch is a `SchemaValidation` error
/// (mapped to `invalid_argument`).
pub fn decode_one_record_batch(buf: &[u8]) -> Result<RecordBatch, StatsError> {
    let reader = StreamReader::try_new(Cursor::new(buf), None).map_err(|e| {
        StatsError::SchemaValidation(format!("WriteRows: failed to open IPC stream: {e}"))
    })?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    for b in reader {
        let batch = b.map_err(|e| {
            StatsError::SchemaValidation(format!("WriteRows: failed to read IPC batch: {e}"))
        })?;
        batches.push(batch);
        if batches.len() > 1 {
            return Err(StatsError::SchemaValidation(
                "WriteRows: expected exactly one RecordBatch in IPC stream, got >1".to_string(),
            ));
        }
    }
    match batches.into_iter().next() {
        Some(batch) => Ok(batch),
        None => Err(StatsError::SchemaValidation(
            "WriteRows: expected exactly one RecordBatch in IPC stream, got 0".to_string(),
        )),
    }
}

/// Encode `batches` to an Arrow IPC stream (schema + batches + EOS).
///
/// `finish()` (writing the EOS marker) is mandatory or readers see a truncated
/// stream; an empty result still emits a valid schema + EOS.
pub fn encode_ipc(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>, ArrowError> {
    let mut out: Vec<u8> = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut out, schema)?;
        for b in batches {
            w.write(b)?;
        }
        w.finish()?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

    fn worker_batch() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("mem_bytes", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Int64Array::from(vec![1_i64, 2])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn decode_one_round_trips_a_single_batch() {
        let batch = worker_batch();
        let schema = batch.schema();
        let buf = encode_ipc(&schema, std::slice::from_ref(&batch)).unwrap();
        let decoded = decode_one_record_batch(&buf).unwrap();
        assert_eq!(decoded.num_rows(), 2);
        assert_eq!(decoded.num_columns(), 2);
        assert_eq!(decoded.schema(), schema);
    }

    #[test]
    fn decode_zero_batches_errs() {
        let batch = worker_batch();
        let schema = batch.schema();
        let buf = encode_ipc(&schema, &[]).unwrap(); // schema + EOS, no batch
        assert!(matches!(
            decode_one_record_batch(&buf),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn decode_more_than_one_batch_errs() {
        let batch = worker_batch();
        let schema = batch.schema();
        let buf = encode_ipc(&schema, &[batch.clone(), batch]).unwrap();
        assert!(matches!(
            decode_one_record_batch(&buf),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn encode_empty_emits_schema_and_eos() {
        let schema: SchemaRef = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            DataType::Int64,
            true,
        )]));
        let buf = encode_ipc(&schema, &[]).unwrap();
        assert!(!buf.is_empty());
        // Decoding it back yields a reader with the schema but no batches.
        let reader = StreamReader::try_new(Cursor::new(&buf), None).unwrap();
        assert_eq!(reader.schema(), schema);
    }
}
