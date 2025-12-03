use arrow::array::{
    Array, AsArray, BooleanBuilder, Int64Builder, ListBuilder, RecordBatch, StringArray,
    StringBuilder, StructArray,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

use crate::hashing::HashAlgorithm;

///
// Helpers

struct SpanBuilder {
    builder: ListBuilder<ListBuilder<Int64Builder>>,
}

impl SpanBuilder {
    fn new() -> Self {
        Self {
            builder: ListBuilder::new(ListBuilder::new(Int64Builder::new())),
        }
    }

    fn append_span(&mut self, start: i64, end: i64) {
        let inner_list = self.builder.values();
        inner_list.values().append_slice(&[start, end, 1]);
        inner_list.append(true);
    }

    fn finish_row(&mut self) {
        self.builder.append(true);
    }

    fn append_null(&mut self) {
        self.builder.append_null();
    }

    fn finish(mut self) -> Arc<dyn Array> {
        Arc::new(self.builder.finish())
    }
}

fn read_column(batch: &RecordBatch, col_name: &str) -> PyResult<StringArray> {
    let col = batch
        .column_by_name(col_name)
        .ok_or_else(|| PyValueError::new_err(format!("Column '{}' missing", col_name)))?;

    match col.data_type() {
        DataType::Utf8 => Ok(col.as_string::<i32>().clone()),
        _ => {
            let casted =
                cast(col, &DataType::Utf8).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(casted.as_string::<i32>().clone())
        }
    }
}

fn maybe_read_column(batch: &RecordBatch, col_name: &str) -> PyResult<Option<StringArray>> {
    match batch.column_by_name(col_name) {
        Some(_) => Ok(Some(read_column(batch, col_name)?)),
        None => Ok(None),
    }
}

fn create_batch(
    fields: Vec<Field>,
    columns: Vec<Arc<dyn Array>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(Arc::new(schema), columns)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrowType(batch))
}

fn resolve_id(
    idx: usize,
    text: &str,
    id_array: Option<&StringArray>,
    algo: HashAlgorithm,
) -> String {
    if let Some(arr) = id_array {
        if !arr.is_null(idx) {
            return arr.value(idx).to_string();
        }
    }
    // If id column is missing, use text hash as id
    algo.hash_to_hex(text.as_bytes())
}

fn build_duplication_map(dup_map: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::with_capacity(dup_map.len());
    for (key, value) in dup_map {
        let hash_key: String = key.extract()?;
        // Check the underlying Python type for "Runtime Type Narrowing"
        let val_dict = value.downcast::<PyDict>()?;
        let canonical_id: String = val_dict
            .get_item("canonical")?
            .ok_or_else(|| PyValueError::new_err("Missing 'canonical' key"))?
            .extract()?;
        map.insert(hash_key, canonical_id);
    }
    Ok(map)
}

///
// Logic

#[pyfunction]
#[pyo3(signature = (batch, text_col, id_col, algorithm=None))]
pub fn process_batch_paragraphs(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    text_col: &str,
    id_col: &str,
    algorithm: Option<HashAlgorithm>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let batch = batch.0;
    let text_arr = read_column(&batch, text_col)?;
    let id_arr = maybe_read_column(&batch, id_col)?;
    let algo = algorithm.unwrap_or(HashAlgorithm::Xxh3_128);

    let (hash_col, id_col_out) = py.allow_threads(move || {
        let cap = text_arr.len() * 2;
        let mut hash_builder = StringBuilder::with_capacity(cap, cap * 64);
        let mut id_builder = StringBuilder::with_capacity(cap, cap * 16);
        let id_arr_ref = id_arr.as_ref();

        for i in 0..text_arr.len() {
            if text_arr.is_null(i) {
                continue;
            }
            let text = text_arr.value(i);
            let row_id = resolve_id(i, text, id_arr_ref, algo);

            for para in text.split('\n') {
                if para.is_empty() {
                    continue;
                }
                hash_builder.append_value(algo.hash_to_hex(para.as_bytes()));
                id_builder.append_value(&row_id);
            }
        }
        (hash_builder.finish(), id_builder.finish())
    });

    create_batch(
        vec![
            Field::new("hash", DataType::Utf8, false),
            Field::new("id", DataType::Utf8, true),
        ],
        vec![Arc::new(hash_col), Arc::new(id_col_out)],
    )
}

#[pyfunction]
#[pyo3(signature = (batch, text_col, id_col, algorithm=None))]
pub fn process_batch_documents(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    text_col: &str,
    id_col: &str,
    algorithm: Option<HashAlgorithm>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let batch = batch.0;
    let text_arr = read_column(&batch, text_col)?;
    let id_arr = maybe_read_column(&batch, id_col)?;
    let algo = algorithm.unwrap_or(HashAlgorithm::Xxh3_128);

    let (hash_col, id_col_out) = py.allow_threads(move || {
        let cap = text_arr.len();
        let mut hash_builder = StringBuilder::with_capacity(cap, cap * 64);
        let mut id_builder = StringBuilder::with_capacity(cap, cap * 16);
        let id_arr_ref = id_arr.as_ref();

        for i in 0..text_arr.len() {
            if text_arr.is_null(i) {
                continue;
            }
            let text = text_arr.value(i);
            let row_id = resolve_id(i, text, id_arr_ref, algo);

            hash_builder.append_value(algo.hash_to_hex(text.as_bytes()));
            id_builder.append_value(&row_id);
        }
        (hash_builder.finish(), id_builder.finish())
    });

    create_batch(
        vec![
            Field::new("hash", DataType::Utf8, false),
            Field::new("id", DataType::Utf8, true),
        ],
        vec![Arc::new(hash_col), Arc::new(id_col_out)],
    )
}

#[pyfunction]
#[pyo3(signature = (batch, text_col, id_col, dup_map, attribute_name, algorithm=None))]
pub fn mark_exact_dups_paragraphs(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    text_col: &str,
    id_col: &str,
    dup_map: &Bound<'_, PyDict>,
    attribute_name: &str,
    algorithm: Option<HashAlgorithm>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let batch = batch.0;
    let text_arr = read_column(&batch, text_col)?;
    let id_arr = maybe_read_column(&batch, id_col)?;
    let algo = algorithm.unwrap_or(HashAlgorithm::Xxh3_128);

    let dup_lookup = build_duplication_map(dup_map)?;

    let (out_ids, struct_array) = py.allow_threads(move || {
        let mut out_id_builder = StringBuilder::new();
        let mut spans_builder = SpanBuilder::new();
        let id_arr_ref = id_arr.as_ref();

        for i in 0..text_arr.len() {
            if text_arr.is_null(i) {
                out_id_builder.append_null();
                spans_builder.append_null();
                continue;
            }

            let text = text_arr.value(i);
            let row_id = resolve_id(i, text, id_arr_ref, algo);
            out_id_builder.append_value(&row_id);

            let mut offset = 0;
            for para in text.split('\n') {
                let len = para.len();
                if !para.is_empty() {
                    let hash = algo.hash_to_hex(para.as_bytes());
                    let is_dup = if let Some(canon_id) = dup_lookup.get(&hash) {
                        canon_id != &row_id
                    } else {
                        false
                    };

                    if is_dup {
                        spans_builder.append_span(offset as i64, (offset + len) as i64);
                    }
                }
                offset += len + 1;
            }
            spans_builder.finish_row();
        }

        let spans_array = spans_builder.finish();
        let attr_field = Field::new(attribute_name, spans_array.data_type().clone(), false);
        let struct_array = StructArray::from(vec![(Arc::new(attr_field), spans_array)]);

        (out_id_builder.finish(), struct_array)
    });

    create_batch(
        vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("attributes", struct_array.data_type().clone(), false),
        ],
        vec![Arc::new(out_ids), Arc::new(struct_array)],
    )
}

#[pyfunction]
#[pyo3(signature = (batch, text_col, id_col, dup_map, attribute_name, algorithm=None))]
pub fn mark_exact_dups_documents(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    text_col: &str,
    id_col: &str,
    dup_map: &Bound<'_, PyDict>,
    attribute_name: &str,
    algorithm: Option<HashAlgorithm>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let batch = batch.0;
    let text_arr = read_column(&batch, text_col)?;
    let id_arr = maybe_read_column(&batch, id_col)?;
    let algo = algorithm.unwrap_or(HashAlgorithm::Xxh3_128);

    let dup_lookup = build_duplication_map(dup_map)?;

    let (out_ids, struct_array) = py.allow_threads(move || {
        let mut out_id_builder = StringBuilder::new();
        let mut is_dup_builder = BooleanBuilder::new();
        let id_arr_ref = id_arr.as_ref();

        for i in 0..text_arr.len() {
            if text_arr.is_null(i) {
                out_id_builder.append_null();
                is_dup_builder.append_null();
                continue;
            }

            let text = text_arr.value(i);
            let row_id = resolve_id(i, text, id_arr_ref, algo);
            out_id_builder.append_value(&row_id);

            let hash = algo.hash_to_hex(text.as_bytes());

            let is_dup = if let Some(canon_id) = dup_lookup.get(&hash) {
                canon_id != &row_id
            } else {
                false
            };

            is_dup_builder.append_value(is_dup);
        }

        let is_dup_array = is_dup_builder.finish();
        let attr_field = Field::new(attribute_name, DataType::Boolean, false);
        let struct_array = StructArray::from(vec![(
            Arc::new(attr_field),
            Arc::new(is_dup_array) as Arc<dyn Array>,
        )]);

        (out_id_builder.finish(), struct_array)
    });

    create_batch(
        vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("attributes", struct_array.data_type().clone(), false),
        ],
        vec![Arc::new(out_ids), Arc::new(struct_array)],
    )
}
