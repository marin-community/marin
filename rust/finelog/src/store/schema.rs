//! Schema dataclasses, Arrow bridge, and validation helpers.
//!
//! `Column` / `Schema` are the in-process representation of a registered
//! table's column layout. They convert to/from the wire proto `Schema`, an
//! `arrow::datatypes::Schema`, and a JSON sidecar form persisted in the
//! catalog.

use std::sync::Arc;

use arrow::array::{new_null_array, ArrayRef, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef, TimeUnit};
use serde::{Deserialize, Serialize};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    Column as ProtoColumn, ColumnType, Schema as ProtoSchema, SchemaView,
};

/// Default implicit ordering-key column name when `Schema.key_column` is empty.
pub const IMPLICIT_KEY_COLUMN: &str = "timestamp_ms";

/// Per-row monotonic counter assigned server-side at write time. Stored on
/// every namespace's parquet segments and visible to SQL queries; never
/// transmitted on the wire and never declared by callers.
pub const IMPLICIT_SEQ_COLUMN: &str = "seq";

/// Max bytes per WriteRows request body.
pub const MAX_WRITE_ROWS_BYTES: usize = 16 * 1024 * 1024;

/// Max rows per RecordBatch. Exactly `1_000_000` (NOT `1 << 20`).
pub const MAX_WRITE_ROWS_ROWS: usize = 1_000_000;

/// One column in a registered schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub r#type: ColumnType,
    pub nullable: bool,
}

impl Column {
    pub fn new(name: impl Into<String>, r#type: ColumnType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            r#type,
            nullable,
        }
    }
}

/// Registered column layout for a namespace.
///
/// `columns` are in registered order (preserved on disk so projections produce
/// stable ordering across additive evolutions). `key_column` empty means the
/// server falls back to `timestamp_ms`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub columns: Vec<Column>,
    pub key_column: String,
}

impl Schema {
    pub fn new(columns: Vec<Column>, key_column: impl Into<String>) -> Self {
        Self {
            columns,
            key_column: key_column.into(),
        }
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// ColumnType <-> Arrow DataType.
// ---------------------------------------------------------------------------

/// Map a `ColumnType` to its Arrow `DataType`.
///
/// `COLUMN_TYPE_UNKNOWN` has no Arrow analogue and returns `None`.
pub fn arrow_type_for(t: ColumnType) -> Option<DataType> {
    match t {
        ColumnType::COLUMN_TYPE_STRING => Some(DataType::Utf8),
        ColumnType::COLUMN_TYPE_INT64 => Some(DataType::Int64),
        ColumnType::COLUMN_TYPE_INT32 => Some(DataType::Int32),
        ColumnType::COLUMN_TYPE_FLOAT64 => Some(DataType::Float64),
        ColumnType::COLUMN_TYPE_BOOL => Some(DataType::Boolean),
        ColumnType::COLUMN_TYPE_TIMESTAMP_MS => {
            Some(DataType::Timestamp(TimeUnit::Millisecond, None))
        }
        ColumnType::COLUMN_TYPE_BYTES => Some(DataType::Binary),
        ColumnType::COLUMN_TYPE_UNKNOWN => None,
    }
}

/// Convert a `Schema` to an `arrow::datatypes::Schema`, preserving nullability.
pub fn schema_to_arrow(schema: &Schema) -> SchemaRef {
    let fields: Vec<Field> = schema
        .columns
        .iter()
        .map(|c| {
            let dt = arrow_type_for(c.r#type).expect("registered column has a known Arrow type");
            Field::new(&c.name, dt, c.nullable)
        })
        .collect();
    Arc::new(ArrowSchema::new(fields))
}

// ---------------------------------------------------------------------------
// Proto conversions.
// ---------------------------------------------------------------------------

/// Decode a wire schema from its view.
///
/// Wire schemas never carry implicit columns (`seq`); a client that includes
/// one is rejected. Rejects `COLUMN_TYPE_UNKNOWN` and unknown wire ints.
pub fn schema_from_proto_view(view: &SchemaView) -> Result<Schema, StatsError> {
    let mut cols = Vec::new();
    for c in view.columns.iter() {
        let name = c.name.unwrap_or("");
        let ctype = c
            .r#type
            .and_then(|ev| ev.as_known())
            .unwrap_or(ColumnType::COLUMN_TYPE_UNKNOWN);
        if ctype == ColumnType::COLUMN_TYPE_UNKNOWN {
            return Err(StatsError::SchemaValidation(format!(
                "column {name:?}: unknown column type"
            )));
        }
        if name == IMPLICIT_SEQ_COLUMN {
            return Err(StatsError::SchemaValidation(format!(
                "column {IMPLICIT_SEQ_COLUMN:?} is reserved (server-assigned implicit column)"
            )));
        }
        cols.push(Column::new(name, ctype, c.nullable.unwrap_or(false)));
    }
    Ok(Schema::new(cols, view.key_column.unwrap_or("")))
}

/// Encode a schema for the wire, stripping the implicit `seq` column.
pub fn schema_to_proto_owned(schema: &Schema) -> ProtoSchema {
    let columns: Vec<ProtoColumn> = schema
        .columns
        .iter()
        .filter(|c| c.name != IMPLICIT_SEQ_COLUMN)
        .map(|c| {
            ProtoColumn::default()
                .with_name(&c.name)
                .with_type(c.r#type)
                .with_nullable(c.nullable)
        })
        .collect();
    ProtoSchema {
        columns,
        ..Default::default()
    }
    .with_key_column(&schema.key_column)
}

// ---------------------------------------------------------------------------
// JSON conversions (catalog sidecar form).
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct JsonColumn {
    name: String,
    /// Proto enum *name* (e.g. "COLUMN_TYPE_STRING"); stable across edits.
    r#type: String,
    nullable: bool,
}

#[derive(Serialize, Deserialize)]
struct JsonSchema {
    key_column: String,
    columns: Vec<JsonColumn>,
}

fn column_type_name(t: ColumnType) -> &'static str {
    match t {
        ColumnType::COLUMN_TYPE_UNKNOWN => "COLUMN_TYPE_UNKNOWN",
        ColumnType::COLUMN_TYPE_STRING => "COLUMN_TYPE_STRING",
        ColumnType::COLUMN_TYPE_INT64 => "COLUMN_TYPE_INT64",
        ColumnType::COLUMN_TYPE_FLOAT64 => "COLUMN_TYPE_FLOAT64",
        ColumnType::COLUMN_TYPE_BOOL => "COLUMN_TYPE_BOOL",
        ColumnType::COLUMN_TYPE_TIMESTAMP_MS => "COLUMN_TYPE_TIMESTAMP_MS",
        ColumnType::COLUMN_TYPE_BYTES => "COLUMN_TYPE_BYTES",
        ColumnType::COLUMN_TYPE_INT32 => "COLUMN_TYPE_INT32",
    }
}

/// Decode a column-type name: proto enum NAME, with a legacy lowercase fallback
/// (`string`, `int64`, ...) for registry DBs predating the proto-enum form.
fn column_type_from_json(name: &str) -> Result<ColumnType, StatsError> {
    let resolved = match name {
        "string" => Some(ColumnType::COLUMN_TYPE_STRING),
        "int64" => Some(ColumnType::COLUMN_TYPE_INT64),
        "int32" => Some(ColumnType::COLUMN_TYPE_INT32),
        "float64" => Some(ColumnType::COLUMN_TYPE_FLOAT64),
        "bool" => Some(ColumnType::COLUMN_TYPE_BOOL),
        "timestamp_ms" => Some(ColumnType::COLUMN_TYPE_TIMESTAMP_MS),
        "bytes" => Some(ColumnType::COLUMN_TYPE_BYTES),
        "COLUMN_TYPE_UNKNOWN" => Some(ColumnType::COLUMN_TYPE_UNKNOWN),
        "COLUMN_TYPE_STRING" => Some(ColumnType::COLUMN_TYPE_STRING),
        "COLUMN_TYPE_INT64" => Some(ColumnType::COLUMN_TYPE_INT64),
        "COLUMN_TYPE_FLOAT64" => Some(ColumnType::COLUMN_TYPE_FLOAT64),
        "COLUMN_TYPE_BOOL" => Some(ColumnType::COLUMN_TYPE_BOOL),
        "COLUMN_TYPE_TIMESTAMP_MS" => Some(ColumnType::COLUMN_TYPE_TIMESTAMP_MS),
        "COLUMN_TYPE_BYTES" => Some(ColumnType::COLUMN_TYPE_BYTES),
        "COLUMN_TYPE_INT32" => Some(ColumnType::COLUMN_TYPE_INT32),
        _ => None,
    };
    resolved.ok_or_else(|| {
        StatsError::Internal(format!("unknown column type name {name:?} in catalog JSON"))
    })
}

/// Serialize a schema to the catalog JSON sidecar form (proto enum NAMES).
pub fn schema_to_json(schema: &Schema) -> String {
    let payload = JsonSchema {
        key_column: schema.key_column.clone(),
        columns: schema
            .columns
            .iter()
            .map(|c| JsonColumn {
                name: c.name.clone(),
                r#type: column_type_name(c.r#type).to_string(),
                nullable: c.nullable,
            })
            .collect(),
    };
    serde_json::to_string(&payload).expect("schema JSON serialization never fails")
}

/// Deserialize a schema from the catalog JSON sidecar form.
pub fn schema_from_json(text: &str) -> Result<Schema, StatsError> {
    let payload: JsonSchema = serde_json::from_str(text)
        .map_err(|e| StatsError::Internal(format!("catalog schema JSON parse: {e}")))?;
    let mut cols = Vec::with_capacity(payload.columns.len());
    for c in payload.columns {
        cols.push(Column::new(
            c.name,
            column_type_from_json(&c.r#type)?,
            c.nullable,
        ));
    }
    Ok(Schema::new(cols, payload.key_column))
}

// ---------------------------------------------------------------------------
// Implicit seq + key resolution.
// ---------------------------------------------------------------------------

/// Return `schema` with the implicit non-nullable INT64 `seq` column prepended.
/// No-op if `seq` is already declared.
pub fn with_implicit_seq(schema: Schema) -> Schema {
    if schema.columns.iter().any(|c| c.name == IMPLICIT_SEQ_COLUMN) {
        return schema;
    }
    let mut columns = Vec::with_capacity(schema.columns.len() + 1);
    columns.push(Column::new(
        IMPLICIT_SEQ_COLUMN,
        ColumnType::COLUMN_TYPE_INT64,
        false,
    ));
    columns.extend(schema.columns);
    Schema::new(columns, schema.key_column)
}

/// Resolve the ordering key column name (presence-only), raising if invalid.
///
/// If `key_column` is set it must name an existing column; otherwise the schema
/// must contain a `timestamp_ms` column. Deliberately does NOT enforce the
/// proto comment's INT64/TIMESTAMP_MS type rule — presence is the only check.
pub fn resolve_key_column(schema: &Schema) -> Result<String, StatsError> {
    if !schema.key_column.is_empty() {
        if schema.column(&schema.key_column).is_none() {
            return Err(StatsError::SchemaValidation(format!(
                "key_column={:?} is not present in the schema columns",
                schema.key_column
            )));
        }
        return Ok(schema.key_column.clone());
    }
    if schema.column(IMPLICIT_KEY_COLUMN).is_none() {
        return Err(StatsError::SchemaValidation(format!(
            "schema declares no key_column and has no implicit '{IMPLICIT_KEY_COLUMN}' column"
        )));
    }
    Ok(IMPLICIT_KEY_COLUMN.to_string())
}

// ---------------------------------------------------------------------------
// Schema merge (additive-only).
// ---------------------------------------------------------------------------

/// Return the effective schema for a re-register against `registered`.
///
/// - identical / requested ⊆ registered -> `registered` unchanged.
/// - requested adds nullable columns -> the union (registered then new).
/// - any conflicting column (type or nullability change) -> `SchemaConflict`.
/// - a new non-nullable column -> `SchemaConflict`.
/// - a differing `key_column` is a *hint*: warn and keep the registered value.
pub fn merge_schemas(registered: &Schema, requested: &Schema) -> Result<Schema, StatsError> {
    if registered.key_column != requested.key_column {
        tracing::warn!(
            registered = %registered.key_column,
            requested = %requested.key_column,
            "register: key_column hint mismatch — using registered",
        );
    }

    let mut extras: Vec<Column> = Vec::new();
    for rc in &requested.columns {
        match registered.column(&rc.name) {
            None => {
                if !rc.nullable {
                    return Err(StatsError::SchemaConflict(format!(
                        "non-additive change: new column {:?} must be nullable for evolve-merge",
                        rc.name
                    )));
                }
                extras.push(rc.clone());
            }
            Some(existing) => {
                if existing.r#type != rc.r#type {
                    return Err(StatsError::SchemaConflict(format!(
                        "column {:?}: type mismatch registered={} requested={}",
                        rc.name,
                        column_type_name(existing.r#type),
                        column_type_name(rc.r#type),
                    )));
                }
                if existing.nullable != rc.nullable {
                    return Err(StatsError::SchemaConflict(format!(
                        "column {:?}: nullable mismatch registered={} requested={}",
                        rc.name, existing.nullable, rc.nullable,
                    )));
                }
            }
        }
    }

    if extras.is_empty() {
        return Ok(registered.clone());
    }
    let mut merged = registered.columns.clone();
    merged.extend(extras);
    Ok(Schema::new(merged, registered.key_column.clone()))
}

// ---------------------------------------------------------------------------
// Per-batch validation: Arrow IPC schema vs registered schema.
//
// The append path consumes the `AlignedBatch` (arrays + fields in registered
// column order, `seq` skipped), stamps `seq`, and builds the final batch in
// one pass.
// ---------------------------------------------------------------------------

/// Map an Arrow `DataType` back to a `ColumnType`, decoding dictionary types to
/// their value type and rejecting nested/union/map types.
///
/// Dictionary-encoded columns are accepted transparently (the *value* type is
/// reported); list/large-list/struct/union/map and any other unsupported type
/// are rejected.
pub fn arrow_to_column_type(dt: &DataType) -> Result<ColumnType, StatsError> {
    match dt {
        DataType::Dictionary(_, value) => arrow_to_column_type(value),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Struct(_)
        | DataType::Union(_, _)
        | DataType::Map(_, _) => Err(StatsError::SchemaValidation(format!(
            "nested/union arrow type {dt:?} is not supported"
        ))),
        DataType::Utf8 => Ok(ColumnType::COLUMN_TYPE_STRING),
        DataType::Int64 => Ok(ColumnType::COLUMN_TYPE_INT64),
        DataType::Int32 => Ok(ColumnType::COLUMN_TYPE_INT32),
        DataType::Float64 => Ok(ColumnType::COLUMN_TYPE_FLOAT64),
        DataType::Boolean => Ok(ColumnType::COLUMN_TYPE_BOOL),
        // Only tz-naive ms timestamps map to TIMESTAMP_MS; a tz-aware column
        // falls through to the unsupported-type error.
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            Ok(ColumnType::COLUMN_TYPE_TIMESTAMP_MS)
        }
        DataType::Binary => Ok(ColumnType::COLUMN_TYPE_BYTES),
        other => Err(StatsError::SchemaValidation(format!(
            "unsupported arrow type {other:?}"
        ))),
    }
}

/// Replace any dictionary-encoded columns of `batch` with their decoded value
/// arrays, returning the decoded batch (unchanged if no dictionary columns).
///
/// Dictionary encoding is a wire-only optimization; the on-disk parquet schema
/// stores plain value types.
pub fn decode_dictionary_columns(batch: &RecordBatch) -> Result<RecordBatch, StatsError> {
    let schema = batch.schema();
    let mut changed = false;
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        if let DataType::Dictionary(_, value_type) = field.data_type() {
            let decoded = cast(col, value_type).map_err(|e| {
                StatsError::SchemaValidation(format!(
                    "column {:?}: failed to decode dictionary to {value_type:?}: {e}",
                    field.name()
                ))
            })?;
            columns.push(decoded);
            fields.push(Field::new(
                field.name(),
                value_type.as_ref().clone(),
                field.is_nullable(),
            ));
            changed = true;
        } else {
            columns.push(Arc::clone(col));
            fields.push(field.as_ref().clone());
        }
    }
    if !changed {
        return Ok(batch.clone());
    }
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|e| StatsError::Internal(format!("rebuilding dictionary-decoded batch: {e}")))
}

/// Validated, schema-aligned arrays for the append hot path.
///
/// `arrays`/`fields` are in registered column order with the implicit `seq`
/// column skipped — the namespace stamps `seq` under the insertion lock and
/// builds the final batch in one pass. `byte_size` sums raw buffer sizes (a
/// monotone approximation feeding the flush-trigger accounting).
#[derive(Debug, Clone)]
pub struct AlignedBatch {
    pub arrays: Vec<ArrayRef>,
    pub fields: Vec<Field>,
    pub num_rows: usize,
    pub byte_size: i64,
}

fn array_buffer_size(arr: &ArrayRef) -> i64 {
    arr.to_data().buffers().iter().map(|b| b.len() as i64).sum()
}

/// Validate an incoming `RecordBatch` against a registered schema.
///
/// Returns the aligned arrays + fields in registered column order with the
/// implicit `seq` column skipped; missing nullable columns are NULL-filled.
/// Rejects: a batch column
/// literally named `seq`, a duplicate column, an unknown column, a missing
/// non-nullable column, a type mismatch (after dictionary decode), and any
/// nested/union arrow type.
pub fn validate_and_align_batch(
    batch: &RecordBatch,
    registered: &Schema,
) -> Result<AlignedBatch, StatsError> {
    let decoded = decode_dictionary_columns(batch)?;
    let decoded_schema = decoded.schema();

    // Build name -> (field, array) map of the inbound batch, rejecting
    // duplicates and the reserved `seq` column.
    let mut by_name_batch: std::collections::HashMap<&str, (DataType, ArrayRef)> =
        std::collections::HashMap::new();
    for (i, field) in decoded_schema.fields().iter().enumerate() {
        let name = field.name().as_str();
        if name == IMPLICIT_SEQ_COLUMN {
            return Err(StatsError::SchemaValidation(format!(
                "column {IMPLICIT_SEQ_COLUMN:?} is reserved (server-assigned implicit column)"
            )));
        }
        if by_name_batch.contains_key(name) {
            return Err(StatsError::SchemaValidation(format!(
                "duplicate column {name:?} in batch"
            )));
        }
        by_name_batch.insert(
            name,
            (field.data_type().clone(), Arc::clone(decoded.column(i))),
        );
    }

    // Reject any inbound column not in the registered schema.
    let registered_names: std::collections::HashSet<&str> =
        registered.columns.iter().map(|c| c.name.as_str()).collect();
    for name in by_name_batch.keys() {
        if !registered_names.contains(name) {
            return Err(StatsError::SchemaValidation(format!(
                "unknown column {name:?} not in registered schema"
            )));
        }
    }

    let n_rows = decoded.num_rows();
    let mut aligned_arrays: Vec<ArrayRef> = Vec::new();
    let mut aligned_fields: Vec<Field> = Vec::new();
    let mut byte_size: i64 = 0;
    for col in &registered.columns {
        if col.name == IMPLICIT_SEQ_COLUMN {
            continue;
        }
        let arrow_dt =
            arrow_type_for(col.r#type).expect("registered column has a known Arrow type");
        match by_name_batch.get(col.name.as_str()) {
            Some((actual_dt, array)) => {
                let actual_type = arrow_to_column_type(actual_dt)?;
                if actual_type != col.r#type {
                    return Err(StatsError::SchemaValidation(format!(
                        "column {:?}: type mismatch registered={} batch={}",
                        col.name,
                        column_type_name(col.r#type),
                        column_type_name(actual_type),
                    )));
                }
                byte_size += array_buffer_size(array);
                aligned_arrays.push(Arc::clone(array));
            }
            None => {
                if !col.nullable {
                    return Err(StatsError::SchemaValidation(format!(
                        "column {:?}: missing required (non-nullable) column",
                        col.name
                    )));
                }
                let null_array = new_null_array(&arrow_dt, n_rows);
                byte_size += array_buffer_size(&null_array);
                aligned_arrays.push(null_array);
            }
        }
        aligned_fields.push(Field::new(&col.name, arrow_dt, col.nullable));
    }

    Ok(AlignedBatch {
        arrays: aligned_arrays,
        fields: aligned_fields,
        num_rows: n_rows,
        byte_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(name: &str, t: ColumnType, nullable: bool) -> Column {
        Column::new(name, t, nullable)
    }

    fn worker_schema() -> Schema {
        Schema::new(
            vec![
                col("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                col("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        )
    }

    #[test]
    fn arrow_type_map_covers_all_seven_types() {
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_STRING),
            Some(DataType::Utf8)
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_INT64),
            Some(DataType::Int64)
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_INT32),
            Some(DataType::Int32)
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_FLOAT64),
            Some(DataType::Float64)
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_BOOL),
            Some(DataType::Boolean)
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_TIMESTAMP_MS),
            Some(DataType::Timestamp(TimeUnit::Millisecond, None))
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_BYTES),
            Some(DataType::Binary)
        );
        assert_eq!(arrow_type_for(ColumnType::COLUMN_TYPE_UNKNOWN), None);
    }

    #[test]
    fn schema_to_proto_strips_implicit_seq() {
        let stored = with_implicit_seq(worker_schema());
        let proto = schema_to_proto_owned(&stored);
        let names: Vec<&str> = proto
            .columns
            .iter()
            .map(|c| c.name.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(names, vec!["worker_id", "mem_bytes", "timestamp_ms"]);
    }

    #[test]
    fn with_implicit_seq_prepends_int64_seq_and_is_idempotent() {
        let stored = with_implicit_seq(worker_schema());
        assert_eq!(stored.columns[0].name, "seq");
        assert_eq!(stored.columns[0].r#type, ColumnType::COLUMN_TYPE_INT64);
        assert!(!stored.columns[0].nullable);
        let again = with_implicit_seq(stored.clone());
        assert_eq!(again, stored);
    }

    #[test]
    fn resolve_key_column_presence_only() {
        // explicit present
        let s = Schema::new(
            vec![
                col("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                col("ts", ColumnType::COLUMN_TYPE_TIMESTAMP_MS, false),
            ],
            "ts",
        );
        assert_eq!(resolve_key_column(&s).unwrap(), "ts");

        // explicit absent -> err
        let s = Schema::new(
            vec![col("worker_id", ColumnType::COLUMN_TYPE_STRING, false)],
            "ts",
        );
        assert!(resolve_key_column(&s).is_err());

        // implicit timestamp_ms present -> ok
        assert_eq!(
            resolve_key_column(&worker_schema()).unwrap(),
            "timestamp_ms"
        );

        // neither -> err
        let s = Schema::new(
            vec![
                col("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                col("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        assert!(resolve_key_column(&s).is_err());
    }

    #[test]
    fn resolve_key_column_string_key_accepted_despite_proto_comment() {
        // A string explicit key is accepted (presence-only); the proto comment
        // about INT64/TIMESTAMP_MS is deliberately not enforced.
        let s = Schema::new(
            vec![
                col("k", ColumnType::COLUMN_TYPE_STRING, false),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "k",
        );
        assert_eq!(resolve_key_column(&s).unwrap(), "k");
    }

    #[test]
    fn json_round_trip_uses_proto_names_and_accepts_legacy() {
        let stored = with_implicit_seq(worker_schema());
        let json = schema_to_json(&stored);
        assert!(json.contains("COLUMN_TYPE_INT64"));
        let back = schema_from_json(&json).unwrap();
        assert_eq!(back, stored);

        // legacy lowercase names rehydrate.
        let legacy = r#"{"key_column":"","columns":[{"name":"x","type":"int64","nullable":true}]}"#;
        let s = schema_from_json(legacy).unwrap();
        assert_eq!(s.columns[0].r#type, ColumnType::COLUMN_TYPE_INT64);
    }

    #[test]
    fn merge_identical_and_subset_return_registered() {
        let reg = with_implicit_seq(worker_schema());
        assert_eq!(merge_schemas(&reg, &reg).unwrap(), reg);

        let subset = Schema::new(
            vec![
                col("seq", ColumnType::COLUMN_TYPE_INT64, false),
                col("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
            ],
            "",
        );
        assert_eq!(merge_schemas(&reg, &subset).unwrap(), reg);
    }

    #[test]
    fn merge_additive_nullable_extends_in_order() {
        let reg = with_implicit_seq(worker_schema());
        let mut req_cols = reg.columns.clone();
        req_cols.push(col("note", ColumnType::COLUMN_TYPE_STRING, true));
        let req = Schema::new(req_cols, "");
        let merged = merge_schemas(&reg, &req).unwrap();
        assert_eq!(
            merged.column_names(),
            vec!["seq", "worker_id", "mem_bytes", "timestamp_ms", "note"]
        );
    }

    #[test]
    fn merge_type_change_rejects() {
        let reg = with_implicit_seq(worker_schema());
        let req = Schema::new(
            vec![col("mem_bytes", ColumnType::COLUMN_TYPE_FLOAT64, false)],
            "",
        );
        assert!(matches!(
            merge_schemas(&reg, &req),
            Err(StatsError::SchemaConflict(_))
        ));
    }

    #[test]
    fn merge_nullable_change_rejects() {
        let reg = with_implicit_seq(worker_schema());
        let req = Schema::new(
            vec![col("mem_bytes", ColumnType::COLUMN_TYPE_INT64, true)],
            "",
        );
        assert!(matches!(
            merge_schemas(&reg, &req),
            Err(StatsError::SchemaConflict(_))
        ));
    }

    #[test]
    fn merge_new_non_nullable_rejects() {
        let reg = with_implicit_seq(worker_schema());
        let mut req_cols = reg.columns.clone();
        req_cols.push(col("cpu_pct", ColumnType::COLUMN_TYPE_FLOAT64, false));
        let req = Schema::new(req_cols, "");
        assert!(matches!(
            merge_schemas(&reg, &req),
            Err(StatsError::SchemaConflict(_))
        ));
    }

    #[test]
    fn merge_key_column_hint_coerced_to_registered() {
        let reg = Schema::new(
            vec![
                col("seq", ColumnType::COLUMN_TYPE_INT64, false),
                col("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        let req = Schema::new(reg.columns.clone(), "timestamp_ms");
        let merged = merge_schemas(&reg, &req).unwrap();
        assert_eq!(merged.key_column, reg.key_column); // kept registered (empty)
    }

    // -----------------------------------------------------------------------
    // validate_and_align_batch / arrow_to_column_type.
    // -----------------------------------------------------------------------

    use arrow::array::{
        Array, ArrayRef, BooleanArray, DictionaryArray, Float64Array, Int32Array, Int64Array,
        ListArray, StringArray,
    };
    use arrow::datatypes::Int32Type;

    /// Registered store-form worker schema (with implicit `seq`).
    fn worker_stored() -> Schema {
        with_implicit_seq(worker_schema())
    }

    fn batch(fields: Vec<Field>, arrays: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays).unwrap()
    }

    #[test]
    fn arrow_to_column_type_round_trips_all_seven() {
        for t in [
            ColumnType::COLUMN_TYPE_STRING,
            ColumnType::COLUMN_TYPE_INT64,
            ColumnType::COLUMN_TYPE_INT32,
            ColumnType::COLUMN_TYPE_FLOAT64,
            ColumnType::COLUMN_TYPE_BOOL,
            ColumnType::COLUMN_TYPE_TIMESTAMP_MS,
            ColumnType::COLUMN_TYPE_BYTES,
        ] {
            let dt = arrow_type_for(t).unwrap();
            assert_eq!(arrow_to_column_type(&dt).unwrap(), t);
        }
        // timestamp_ms has no timezone.
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_TIMESTAMP_MS).unwrap(),
            DataType::Timestamp(TimeUnit::Millisecond, None)
        );
    }

    #[test]
    fn align_full_batch_passes_through_in_registered_order() {
        let b = batch(
            vec![
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("mem_bytes", DataType::Int64, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![
                Arc::new(StringArray::from(vec!["w1", "w2"])),
                Arc::new(Int64Array::from(vec![10_i64, 20])),
                Arc::new(Int64Array::from(vec![100_i64, 200])),
            ],
        );
        let aligned = validate_and_align_batch(&b, &worker_stored()).unwrap();
        assert_eq!(aligned.num_rows, 2);
        let names: Vec<&str> = aligned.fields.iter().map(|f| f.name().as_str()).collect();
        // `seq` is skipped; registered order preserved.
        assert_eq!(names, vec!["worker_id", "mem_bytes", "timestamp_ms"]);
        assert!(aligned.byte_size > 0);
    }

    #[test]
    fn align_missing_nullable_null_fills() {
        let mut cols = worker_schema().columns;
        cols.push(Column::new("note", ColumnType::COLUMN_TYPE_STRING, true));
        let registered = with_implicit_seq(Schema::new(cols, ""));
        let b = batch(
            vec![
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("mem_bytes", DataType::Int64, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![
                Arc::new(StringArray::from(vec!["w1"])),
                Arc::new(Int64Array::from(vec![10_i64])),
                Arc::new(Int64Array::from(vec![100_i64])),
            ],
        );
        let aligned = validate_and_align_batch(&b, &registered).unwrap();
        assert_eq!(aligned.fields.len(), 4);
        let note = &aligned.arrays[3];
        assert_eq!(note.len(), 1);
        assert_eq!(note.null_count(), 1); // NULL-filled
        assert_eq!(note.data_type(), &DataType::Utf8);
    }

    #[test]
    fn align_missing_non_nullable_rejected() {
        // omit the non-nullable `mem_bytes`.
        let b = batch(
            vec![
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![
                Arc::new(StringArray::from(vec!["w1"])),
                Arc::new(Int64Array::from(vec![100_i64])),
            ],
        );
        assert!(matches!(
            validate_and_align_batch(&b, &worker_stored()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn align_unknown_column_rejected() {
        let b = batch(
            vec![
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("mem_bytes", DataType::Int64, false),
                Field::new("timestamp_ms", DataType::Int64, false),
                Field::new("bogus", DataType::Int64, true),
            ],
            vec![
                Arc::new(StringArray::from(vec!["w1"])),
                Arc::new(Int64Array::from(vec![10_i64])),
                Arc::new(Int64Array::from(vec![100_i64])),
                Arc::new(Int64Array::from(vec![1_i64])),
            ],
        );
        assert!(matches!(
            validate_and_align_batch(&b, &worker_stored()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn align_type_mismatch_rejected() {
        // mem_bytes is Int64 in the schema; send Float64.
        let b = batch(
            vec![
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("mem_bytes", DataType::Float64, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![
                Arc::new(StringArray::from(vec!["w1"])),
                Arc::new(Float64Array::from(vec![1.5_f64])),
                Arc::new(Int64Array::from(vec![100_i64])),
            ],
        );
        assert!(matches!(
            validate_and_align_batch(&b, &worker_stored()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn align_dictionary_column_decoded_to_value_type() {
        // worker_id arrives dictionary(int32 -> utf8); accepted and decoded.
        let keys = Int32Array::from(vec![0_i32, 1, 0]);
        let values = StringArray::from(vec!["a", "b"]);
        let dict: DictionaryArray<Int32Type> =
            DictionaryArray::try_new(keys, Arc::new(values)).unwrap();
        let dict_dt = dict.data_type().clone();
        let b = batch(
            vec![
                Field::new("worker_id", dict_dt, false),
                Field::new("mem_bytes", DataType::Int64, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![
                Arc::new(dict),
                Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
                Arc::new(Int64Array::from(vec![10_i64, 20, 30])),
            ],
        );
        let aligned = validate_and_align_batch(&b, &worker_stored()).unwrap();
        assert_eq!(aligned.num_rows, 3);
        // Decoded to the value type (Utf8), not left as a dictionary.
        assert_eq!(aligned.arrays[0].data_type(), &DataType::Utf8);
    }

    #[test]
    fn align_nested_type_rejected() {
        // worker_id arrives as a List, which is unsupported.
        let list =
            ListArray::from_iter_primitive::<arrow::datatypes::Int64Type, _, _>(vec![Some(vec![
                Some(1_i64),
            ])]);
        let list_dt = list.data_type().clone();
        let b = batch(
            vec![Field::new("worker_id", list_dt, false)],
            vec![Arc::new(list) as ArrayRef],
        );
        assert!(matches!(
            validate_and_align_batch(&b, &worker_stored()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn align_literal_seq_column_rejected() {
        let b = batch(
            vec![
                Field::new("seq", DataType::Int64, false),
                Field::new("worker_id", DataType::Utf8, false),
            ],
            vec![
                Arc::new(Int64Array::from(vec![1_i64])),
                Arc::new(StringArray::from(vec!["w1"])),
            ],
        );
        assert!(matches!(
            validate_and_align_batch(&b, &worker_stored()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn align_bool_column_type() {
        // exercise the boolean arrow type mapping through a tiny one-col schema.
        let registered = with_implicit_seq(Schema::new(
            vec![
                Column::new("flag", ColumnType::COLUMN_TYPE_BOOL, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ));
        let b = batch(
            vec![
                Field::new("flag", DataType::Boolean, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(Int64Array::from(vec![1_i64, 2])),
            ],
        );
        let aligned = validate_and_align_batch(&b, &registered).unwrap();
        assert_eq!(aligned.arrays[0].data_type(), &DataType::Boolean);
    }
}
