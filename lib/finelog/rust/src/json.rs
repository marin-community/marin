//! Shared scalar JSON semantics for query evaluation and derived summaries.

use serde_json::Value as JsonValue;

/// Return the raw value at a top-level object key.
///
/// Invalid JSON, non-object documents, and absent keys return `None`. An
/// explicit JSON null remains `Some(JsonValue::Null)` so callers can preserve
/// the distinction until they apply SQL null semantics.
pub(crate) fn object_value(text: &str, key: &str) -> Option<JsonValue> {
    match serde_json::from_str::<JsonValue>(text).ok()? {
        JsonValue::Object(mut map) => map.remove(key),
        _ => None,
    }
}

/// Render one JSON value using Finelog's `json_get` text semantics.
pub(crate) fn value_as_text(value: JsonValue) -> Option<String> {
    match value {
        JsonValue::Null => None,
        JsonValue::String(value) => Some(value),
        value => Some(value.to_string()),
    }
}

/// Evaluate Finelog's `json_get` semantics for one string document.
pub(crate) fn get_text(document: &str, key: &str) -> Option<String> {
    value_as_text(object_value(document, key)?)
}
