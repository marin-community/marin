// Code for benchmarking purposes (measure the overhead of Py <-> Rust FFI marshaling)
// The functions here are mostly passthrough (they return their inputs, only truncating the text to 100 chars)
use arrow::array::{Array, RecordBatch, StringArray, StringBuilder};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

fn truncate_text(text: &str) -> String {
    text.chars().take(100).collect()
}

fn transform_arrow_batch_impl(
    batch: &RecordBatch,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let id_col = batch.column_by_name("id").ok_or_else(|| {
        arrow::error::ArrowError::InvalidArgumentError("Column 'id' missing".to_string())
    })?;
    let text_col = batch.column_by_name("text").ok_or_else(|| {
        arrow::error::ArrowError::InvalidArgumentError("Column 'text' missing".to_string())
    })?;

    let id_array = id_col
        .as_any()
        // Convert Arrow generics into a StringArray
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            arrow::error::ArrowError::InvalidArgumentError("id is not a string array".to_string())
        })?;
    let text_array = text_col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            arrow::error::ArrowError::InvalidArgumentError("text is not a string array".to_string())
        })?;

    let rows = batch.num_rows();
    let mut id_builder = StringBuilder::with_capacity(rows, rows * 16);
    let mut head_builder = StringBuilder::with_capacity(rows, rows * 100);

    for i in 0..rows {
        if id_array.is_null(i) {
            id_builder.append_null();
        } else {
            id_builder.append_value(id_array.value(i));
        }

        if text_array.is_null(i) {
            head_builder.append_null();
        } else {
            head_builder.append_value(truncate_text(text_array.value(i)));
        }
    }

    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("head", DataType::Utf8, true),
    ]);

    RecordBatch::try_new(
        //  Atomic Reference Counted to share immutable arrays without mem copy
        Arc::new(schema),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(head_builder.finish()),
        ],
    )
}

// Rust Native (Reads Parquet -> Transforms -> Returns Arrow Batch)
#[pyfunction]
pub fn process_native(path: String) -> PyResult<PyArrowType<RecordBatch>> {
    let path = Path::new(&path);
    let file = File::open(path)?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let file_metadata = builder.metadata().file_metadata();
    let schema_descr = file_metadata.schema_descr();

    // We need 'id' and 'text' columns
    let id_idx = schema_descr
        .columns()
        .iter()
        .position(|c| c.name() == "id")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Column 'id' not found"))?;
    let text_idx = schema_descr
        .columns()
        .iter()
        .position(|c| c.name() == "text")
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Column 'text' not found")
        })?;

    // Select columns to project
    // Deets in https://arrow.apache.org/rust/parquet/arrow/struct.ProjectionMask.html
    let mask = ProjectionMask::leaves(schema_descr, [id_idx, text_idx]);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut processed_batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let processed = transform_arrow_batch_impl(&batch)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        processed_batches.push(processed);
    }

    if processed_batches.is_empty() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("head", DataType::Utf8, true),
        ]);
        let empty = RecordBatch::new_empty(Arc::new(schema));
        return Ok(PyArrowType(empty));
    }

    let schema = processed_batches[0].schema();
    let combined = concat_batches(&schema, &processed_batches)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyArrowType(combined))
}

// Arrow Batch (Single)
#[pyfunction]
pub fn process_arrow_batch(batch: PyArrowType<RecordBatch>) -> PyResult<PyArrowType<RecordBatch>> {
    let out = transform_arrow_batch_impl(&batch.0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyArrowType(out))
}

// Rust Structs
#[pyclass]
#[derive(Clone)]
pub struct Document {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub text: String,
}

#[pymethods]
impl Document {
    #[new]
    fn new(id: String, text: String) -> Self {
        Document { id, text }
    }
}

#[pyfunction]
pub fn process_rust_structs(docs: Vec<PyRef<Document>>) -> Vec<Document> {
    docs.iter()
        .map(|d| Document {
            id: d.id.clone(),
            text: truncate_text(&d.text),
        })
        .collect()
}

// Dicts Batch
#[pyfunction]
pub fn process_dicts_batch(py: Python, docs: Vec<Bound<'_, PyDict>>) -> PyResult<Vec<Py<PyAny>>> {
    let mut results = Vec::with_capacity(docs.len());
    for doc in docs {
        let id_item = doc
            .get_item("id")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Key 'id' missing"))?;
        let text_item = doc
            .get_item("text")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Key 'text' missing"))?;

        let id: String = id_item.extract()?;
        let text: String = text_item.extract()?;

        let new_dict = PyDict::new(py);
        new_dict.set_item("id", id)?;
        new_dict.set_item("head", truncate_text(&text))?;
        results.push(new_dict.unbind().into());
    }
    Ok(results)
}

// Dicts Loop
#[pyfunction]
pub fn process_dicts_loop(py: Python, doc: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
    let id_item = doc
        .get_item("id")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Key 'id' missing"))?;
    let text_item = doc
        .get_item("text")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Key 'text' missing"))?;

    let id: String = id_item.extract()?;
    let text: String = text_item.extract()?;

    let new_dict = PyDict::new(py);
    new_dict.set_item("id", id)?;
    new_dict.set_item("head", truncate_text(&text))?;

    Ok(new_dict.unbind().into())
}
