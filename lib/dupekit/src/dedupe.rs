use crate::ops;
use arrow::array::{
    Array, BooleanBuilder, Int64Builder, ListBuilder, RecordBatch, StringBuilder, StructArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

use crate::hashing::{HashAlgorithm, DEFAULT_HASH_ALGO};

fn build_lookup(dup_map: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::with_capacity(dup_map.len());
    for (k, v) in dup_map {
        let val = v
            .downcast::<PyDict>()?
            .get_item("canonical")?
            .unwrap()
            .extract::<String>()?;
        map.insert(k.extract()?, val);
    }
    Ok(map)
}

fn record_batch_from_columns(
    fields: Vec<Field>,
    columns: Vec<Arc<dyn Array>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let schema = Schema::new(fields);
    Ok(PyArrowType(
        RecordBatch::try_new(Arc::new(schema), columns)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    ))
}

#[pyfunction]
#[pyo3(signature = (batch, dup_map, attribute_name, algorithm=None))]
pub fn mark_paragraph_duplicates(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    dup_map: &Bound<'_, PyDict>,
    attribute_name: &str,
    algorithm: Option<HashAlgorithm>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let input = batch.0;
    let dup_lookup = build_lookup(dup_map)?;

    // This function expects an input batch of DOCUMENTS, not paragraphs.
    // It will internally split, hash, and aggregate.
    let text_arr = ops::get_string_array(&input, "text")?;
    let id_arr = ops::get_string_array(&input, "id")?;
    let algo = algorithm.unwrap_or(DEFAULT_HASH_ALGO);

    let (id_out, spans) = py.allow_threads(move || {
        let mut id_builder = StringBuilder::with_capacity(input.num_rows(), input.num_rows() * 16);
        let mut span_list_builder = ListBuilder::new(ListBuilder::new(Int64Builder::new()));

        for i in 0..input.num_rows() {
            let doc_id = id_arr.value(i);
            id_builder.append_value(doc_id);

            if text_arr.is_valid(i) {
                let text = text_arr.value(i);
                for span in ops::split_paragraphs_str(text) {
                    if !span.text.is_empty() {
                        let h = algo.hash_to_hex(span.text.as_bytes());
                        if let Some(canon) = dup_lookup.get(&h) {
                            if canon != doc_id {
                                span_list_builder
                                    .values()
                                    .values()
                                    .append_slice(&[span.start, span.end, 1]);
                                // Close inner list
                                span_list_builder.values().append(true);
                            }
                        }
                    }
                }
            }
            // Close span list
            span_list_builder.append(true);
        }
        (id_builder.finish(), span_list_builder.finish())
    });

    let spans_arr = Arc::new(spans) as Arc<dyn Array>;
    let struct_arr = StructArray::from(vec![(
        Arc::new(Field::new(
            attribute_name,
            spans_arr.data_type().clone(),
            false,
        )),
        spans_arr,
    )]);

    record_batch_from_columns(
        vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("attributes", struct_arr.data_type().clone(), false),
        ],
        vec![Arc::new(id_out), Arc::new(struct_arr)],
    )
}

#[pyfunction]
#[pyo3(signature = (batch, dup_map, attribute_name, hash_col=None, algorithm=None))]
pub fn mark_document_duplicates(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    dup_map: &Bound<'_, PyDict>,
    attribute_name: &str,
    hash_col: Option<String>,
    algorithm: Option<HashAlgorithm>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let input = batch.0;
    let dup_lookup = build_lookup(dup_map)?;
    let id_arr = ops::get_string_array(&input, "id")?;
    let algo = algorithm.unwrap_or(DEFAULT_HASH_ALGO);

    let hash_arr = if let Some(col_name) = hash_col {
        ops::get_string_array(&input, &col_name)?
    } else {
        let text_arr = ops::get_string_array(&input, "text")?;
        ops::hash_array(&text_arr, algo)?
    };

    let (id_out, bools) = py.allow_threads(move || {
        let mut id_builder = StringBuilder::with_capacity(input.num_rows(), input.num_rows() * 16);
        let mut bool_builder = BooleanBuilder::with_capacity(input.num_rows());

        for i in 0..input.num_rows() {
            let doc_id = id_arr.value(i);
            id_builder.append_value(doc_id);

            if hash_arr.is_valid(i) {
                let h = hash_arr.value(i);
                // Check map: if it exists and canonical id != current id, it's a dupe
                let is_dup = dup_lookup.get(h).map(|c| c != doc_id).unwrap_or(false);
                bool_builder.append_value(is_dup);
            } else {
                bool_builder.append_null();
            }
        }
        (id_builder.finish(), bool_builder.finish())
    });

    let bools_arr = Arc::new(bools) as Arc<dyn Array>;
    let struct_arr = StructArray::from(vec![(
        Arc::new(Field::new(attribute_name, DataType::Boolean, false)),
        bools_arr,
    )]);

    record_batch_from_columns(
        vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("attributes", struct_arr.data_type().clone(), false),
        ],
        vec![Arc::new(id_out), Arc::new(struct_arr)],
    )
}
