use crate::hashing::{HashAlgorithm, DEFAULT_HASH_ALGO};
use crate::minhash_ops;
use crate::ops;
use arrow::array::{Array, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::borrow::Borrow;
use std::sync::Arc;

#[derive(Clone)]
#[pyclass(module = "dupekit")]
pub enum Transformation {
    ResolveIds {
        text_col: String,
        id_col: String,
        output_col: String,
    },
    SplitParagraphs {
        text_col: String,
        id_col: String,
    },
    Hash {
        input_col: String,
        output_col: String,
        algo: HashAlgorithm,
    },
    SelectColumns {
        columns: Vec<String>,
    },
    // MinHash Pipeline Ops
    CleanText {
        input_col: String,
        output_col: String,
    },
    MinHash {
        input_col: String,
        output_col: String,
        num_perms: usize,
        ngram_size: usize,
        seed: u64,
    },
    MinHashLSH {
        input_col: String,
        output_col: String,
        num_bands: usize,
    },
}

#[pymethods]
impl Transformation {
    #[staticmethod]
    #[pyo3(name = "ResolveIds")]
    fn resolve_ids(text_col: String, id_col: String, output_col: String) -> Transformation {
        Self::ResolveIds {
            text_col,
            id_col,
            output_col,
        }
    }

    #[staticmethod]
    #[pyo3(name = "SplitParagraphs")]
    fn split_paragraphs(text_col: String, id_col: String) -> Transformation {
        Self::SplitParagraphs { text_col, id_col }
    }

    #[staticmethod]
    #[pyo3(name = "Hash")]
    fn hash(input_col: String, output_col: String, algo: HashAlgorithm) -> Transformation {
        Self::Hash {
            input_col,
            output_col,
            algo,
        }
    }

    #[staticmethod]
    #[pyo3(name = "SelectColumns")]
    fn select_columns(columns: Vec<String>) -> Transformation {
        Self::SelectColumns { columns }
    }

    #[staticmethod]
    #[pyo3(name = "CleanText")]
    fn clean_text(input_col: String, output_col: String) -> Transformation {
        Self::CleanText {
            input_col,
            output_col,
        }
    }

    #[staticmethod]
    #[pyo3(name = "MinHash")]
    fn min_hash(
        input_col: String,
        output_col: String,
        num_perms: usize,
        ngram_size: usize,
        seed: u64,
    ) -> Transformation {
        Self::MinHash {
            input_col,
            output_col,
            num_perms,
            ngram_size,
            seed,
        }
    }

    #[staticmethod]
    #[pyo3(name = "MinHashLSH")]
    fn min_hash_lsh(
        input_col: String,
        output_col: String,
        num_bands: usize,
    ) -> Transformation {
        Self::MinHashLSH {
            input_col,
            output_col,
            num_bands,
        }
    }
}

fn apply_transformation(batch: RecordBatch, step: &Transformation) -> PyResult<RecordBatch> {
    match step {
        Transformation::ResolveIds {
            text_col,
            id_col,
            output_col,
        } => {
            let text_arr = ops::get_string_array(&batch, text_col)?;
            let maybe_id_arr = if batch.column_by_name(id_col).is_some() {
                Some(ops::get_string_array(&batch, id_col)?)
            } else {
                None
            };
            let algo = DEFAULT_HASH_ALGO; // Default for ID imputation

            let mut builder = StringBuilder::with_capacity(batch.num_rows(), batch.num_rows() * 16);
            for i in 0..batch.num_rows() {
                if let Some(id_arr) = &maybe_id_arr {
                    if id_arr.is_valid(i) {
                        builder.append_value(id_arr.value(i));
                        continue;
                    }
                }
                if text_arr.is_valid(i) {
                    builder.append_value(algo.hash_to_hex(text_arr.value(i).as_bytes()));
                } else {
                    builder.append_null();
                }
            }
            ops::add_column(&batch, output_col, Arc::new(builder.finish()))
        }

        Transformation::SplitParagraphs { text_col, id_col } => {
            let text_arr = ops::get_string_array(&batch, text_col)?;
            let id_arr = ops::get_string_array(&batch, id_col)?;
            let (para_text, para_span, doc_id) = ops::split_paragraphs(&text_arr, &id_arr)?;
            let schema = Arc::new(Schema::new(vec![
                Field::new("doc_id", doc_id.data_type().clone(), true),
                Field::new("paragraph_text", para_text.data_type().clone(), true),
                Field::new("paragraph_span", para_span.data_type().clone(), false),
            ]));
            RecordBatch::try_new(schema, vec![doc_id, para_text, para_span])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        Transformation::Hash {
            input_col,
            output_col,
            algo,
        } => {
            let input_arr = ops::get_string_array(&batch, input_col)?;
            let hashed_arr = ops::hash_array(&input_arr, *algo)?;
            ops::add_column(&batch, output_col, hashed_arr)
        }

        Transformation::SelectColumns { columns } => ops::select_columns(&batch, columns),

        Transformation::CleanText {
            input_col,
            output_col,
        } => {
            let input_arr = ops::get_string_array(&batch, input_col)?;
            let clean_arr = minhash_ops::clean_text(&input_arr)?;
            ops::add_column(&batch, output_col, clean_arr)
        }

        Transformation::MinHash {
            input_col,
            output_col,
            num_perms,
            ngram_size,
            seed,
        } => {
            let input_arr = ops::get_string_array(&batch, input_col)?;
            let signature_arr =
                minhash_ops::compute_minhash(&input_arr, *num_perms, *ngram_size, *seed)?;
            ops::add_column(&batch, output_col, signature_arr.into())
        }

        Transformation::MinHashLSH {
            input_col,
            output_col,
            num_bands,
        } => {
            let input_arr = batch.column_by_name(input_col).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Column '{}' missing", input_col))
            })?;
            let buckets_arr = minhash_ops::compute_lsh(input_arr.as_ref(), *num_bands)?;
            ops::add_column(&batch, output_col, buckets_arr.into())
        }
    }
}

#[pyfunction]
pub fn transform(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    steps: Vec<PyRef<Transformation>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let rust_steps: Vec<Transformation> = steps.iter().map(|step| (**step).clone()).collect();

    py.allow_threads(move || {
        let mut current_batch = batch.0;
        for step in &rust_steps {
            current_batch = apply_transformation(current_batch, step)?;
        }
        Ok(PyArrowType(current_batch))
    })
}
