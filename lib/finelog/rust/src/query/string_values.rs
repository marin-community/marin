//! Reading a UTF-8 column without caring which of arrow's two string layouts it
//! arrived in.
//!
//! The scan produces `Utf8View` (see
//! [`crate::query::provider::view_typed_schema`]) while batches built in memory —
//! test fixtures, the typed-empty table, a SQL expression result — are still
//! `Utf8`. Code that reads values by row index should accept either, because the
//! alternative is casting a whole column to one layout before touching it, which
//! on the scan path copies every value including the ones a predicate is about to
//! reject.

use arrow::array::{Array, ArrayRef, AsArray, StringArray, StringViewArray};
use arrow::datatypes::DataType;

/// A borrowed UTF-8 column, read by row index.
pub enum StringValues<'a> {
    Utf8(&'a StringArray),
    View(&'a StringViewArray),
}

impl<'a> StringValues<'a> {
    /// Borrow `arr` if it holds UTF-8 in either layout, else `None`.
    pub fn new(arr: &'a ArrayRef) -> Option<Self> {
        match arr.data_type() {
            DataType::Utf8 => Some(StringValues::Utf8(arr.as_string::<i32>())),
            DataType::Utf8View => Some(StringValues::View(arr.as_string_view())),
            _ => None,
        }
    }

    pub fn is_null(&self, i: usize) -> bool {
        match self {
            StringValues::Utf8(a) => a.is_null(i),
            StringValues::View(a) => a.is_null(i),
        }
    }

    pub fn value(&self, i: usize) -> &str {
        match self {
            StringValues::Utf8(a) => a.value(i),
            StringValues::View(a) => a.value(i),
        }
    }
}
