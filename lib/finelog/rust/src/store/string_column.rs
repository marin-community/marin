//! Borrowed access to Arrow's UTF-8 column layouts.

use arrow::array::{Array, LargeStringArray, StringArray, StringViewArray};

pub(crate) enum StringColumn<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
    View(&'a StringViewArray),
}

impl<'a> StringColumn<'a> {
    pub(crate) fn new(array: &'a dyn Array) -> Option<Self> {
        if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
            return Some(Self::Utf8(values));
        }
        if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
            return Some(Self::Large(values));
        }
        array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .map(Self::View)
    }

    pub(crate) fn value(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(values) => (!values.is_null(row)).then(|| values.value(row)),
            Self::Large(values) => (!values.is_null(row)).then(|| values.value(row)),
            Self::View(values) => (!values.is_null(row)).then(|| values.value(row)),
        }
    }
}
