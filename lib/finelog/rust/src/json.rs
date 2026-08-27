//! Shared scalar JSON semantics for query evaluation and derived summaries.

use std::borrow::Cow;
use std::fmt;

use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::Deserializer;
use serde_json::Value as JsonValue;

struct ObjectValueVisitor<'a> {
    key: &'a str,
}

impl<'de> Visitor<'de> for ObjectValueVisitor<'_> {
    type Value = Option<JsonValue>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut found = None;
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            if key == self.key {
                found = Some(map.next_value()?);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(found)
    }
}

/// Return the raw value at a top-level object key.
///
/// Invalid JSON, non-object documents, and absent keys return `None`. An
/// explicit JSON null remains `Some(JsonValue::Null)` so callers can preserve
/// the distinction until they apply SQL null semantics.
pub(crate) fn object_value(text: &str, key: &str) -> Option<JsonValue> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let value = deserializer
        .deserialize_any(ObjectValueVisitor { key })
        .ok()?;
    deserializer.end().ok()?;
    value
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
