use crate::hashing::HashAlgorithm;
use arrow::array::{
    Array, GenericStringArray, Int64Builder, ListBuilder, RecordBatch, StringArray, StringBuilder,
    StructArray,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields, Schema};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

pub struct SubSpan<'a> {
    pub text: &'a str,
    pub start: i64,
    pub end: i64,
}

pub fn split_paragraphs_str(text: &str) -> impl Iterator<Item = SubSpan<'_>> {
    let mut offset = 0;
    text.split('\n').map(move |para| {
        let len = para.len();
        let start = offset;
        offset += len + 1; // +1 for the newline
        SubSpan {
            text: para,
            start: start as i64,
            end: (start + len) as i64,
        }
    })
}

pub fn get_string_array(batch: &RecordBatch, name: &str) -> PyResult<Arc<StringArray>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("Column '{}' missing", name)))?;

    if *col.data_type() == DataType::Utf8 {
        return Ok(col
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .clone()
            .into());
    }

    let casted = cast(col, &DataType::Utf8)
        .map_err(|e| PyValueError::new_err(format!("Failed to cast column '{}': {}", name, e)))?;
    Ok(casted
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .clone()
        .into())
}

pub fn hash_array(
    arr: &GenericStringArray<i32>,
    algo: HashAlgorithm,
) -> PyResult<Arc<StringArray>> {
    let mut builder = StringBuilder::with_capacity(arr.len(), arr.len() * 32);
    for i in 0..arr.len() {
        if arr.is_valid(i) {
            builder.append_value(algo.hash_to_hex(arr.value(i).as_bytes()));
        } else {
            builder.append_null();
        }
    }
    Ok(Arc::new(builder.finish()))
}

pub fn add_column(batch: &RecordBatch, name: &str, col: Arc<dyn Array>) -> PyResult<RecordBatch> {
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().map(|f| f.clone()).collect();
    fields.push(Arc::new(Field::new(name, col.data_type().clone(), true)));

    let mut columns = batch.columns().to_vec();
    columns.push(col);

    let schema = Arc::new(Schema::new(Fields::from(fields)));
    RecordBatch::try_new(schema, columns).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub fn select_columns(batch: &RecordBatch, columns: &[String]) -> PyResult<RecordBatch> {
    let indices: Vec<usize> = columns
        .iter()
        .map(|name| batch.schema().index_of(name))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    batch
        .project(&indices)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub fn split_paragraphs(
    text_arr: &StringArray,
    id_arr: &StringArray,
) -> PyResult<(Arc<StringArray>, Arc<StructArray>, Arc<StringArray>)> {
    let mut para_text_builder = StringBuilder::new();
    let mut span_builder = ListBuilder::new(Int64Builder::new());
    let mut doc_id_builder = StringBuilder::new();

    for i in 0..text_arr.len() {
        if text_arr.is_null(i) {
            continue;
        }
        let text = text_arr.value(i);
        let doc_id = id_arr.value(i);

        for span in split_paragraphs_str(text) {
            if !span.text.is_empty() {
                para_text_builder.append_value(span.text);
                span_builder.values().append_slice(&[span.start, span.end]);
                span_builder.append(true);
                doc_id_builder.append_value(doc_id);
            }
        }
    }

    let span_array = Arc::new(span_builder.finish()) as Arc<dyn Array>;
    let span_struct = StructArray::from(vec![(
        Arc::new(Field::new("span", span_array.data_type().clone(), false)),
        span_array,
    )]);

    Ok((
        Arc::new(para_text_builder.finish()),
        Arc::new(span_struct),
        Arc::new(doc_id_builder.finish()),
    ))
}
