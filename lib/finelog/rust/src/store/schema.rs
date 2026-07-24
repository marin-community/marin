//! Schema dataclasses, Arrow bridge, and validation helpers.
//!
//! `Column` / `Schema` are the in-process representation of a registered
//! table's column layout. They convert to/from the wire proto `Schema`, an
//! `arrow::datatypes::Schema`, and a JSON sidecar form persisted in the
//! catalog.

use std::sync::Arc;

use arrow::array::{new_null_array, ArrayData, ArrayRef, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema, SchemaRef, TimeUnit};
use buffa::MessageField;
use serde::{Deserialize, Serialize};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    Column as ProtoColumn, ColumnIndex as ProtoColumnIndex, ColumnType, Schema as ProtoSchema,
    SchemaView,
};

/// Default implicit ordering-key column name when `Schema.key_column` is empty.
pub const IMPLICIT_KEY_COLUMN: &str = "timestamp_ms";

/// Per-row monotonic counter assigned server-side at write time. Stored on
/// every namespace's parquet segments and visible to SQL queries; never
/// transmitted on the wire and never declared by callers.
pub const IMPLICIT_SEQ_COLUMN: &str = "seq";

/// Origin-cluster column, added to every registered table that does not declare
/// one (see [`with_implicit_cluster`]). Local writers leave it empty; the
/// cross-cluster forwarder stamps it with the origin cluster on the way to a hub
/// finelog, so a hub that collects rows from many federated clusters can filter or
/// group them by the cluster that produced them.
pub const IMPLICIT_CLUSTER_COLUMN: &str = "cluster";

/// Max bytes per WriteRows request body.
pub const MAX_WRITE_ROWS_BYTES: usize = 16 * 1024 * 1024;

/// Max rows per RecordBatch. Exactly `1_000_000` (NOT `1 << 20`).
pub const MAX_WRITE_ROWS_ROWS: usize = 1_000_000;

/// Secondary indexes a column carries. Each index type is its own field so
/// adding one is additive; `ColumnIndex::default()` (all-false) is unindexed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColumnIndex {
    /// Per-row-group trigram substring index in each segment's `.tgm` sidecar.
    /// Only meaningful for STRING columns.
    pub trigram: bool,
}

/// One column in a registered schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub r#type: ColumnType,
    pub nullable: bool,
    pub index: ColumnIndex,
}

impl Column {
    pub fn new(name: impl Into<String>, r#type: ColumnType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            r#type,
            nullable,
            index: ColumnIndex::default(),
        }
    }

    /// Builder: maintain a trigram substring index for this column.
    pub fn with_trigram_index(mut self) -> Self {
        self.index.trigram = true;
        self
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

/// The Arrow `DataType` for `COLUMN_TYPE_MAP`: a `Map<Utf8,Utf8>` with a
/// non-nullable string key and a nullable string value.
///
/// The entries/key/value field names and the unsorted flag match pyarrow's
/// `pa.map_(pa.string(), pa.string())` exactly, so a batch a client sends is
/// bit-identical to this declared type and the write path accepts it cast-free.
/// arrow-rs's default parquet writer round-trips a map with these names
/// losslessly (it keeps the entries field name and rehydrates the key/value
/// names from the embedded Arrow schema), so on-disk segments read back as the
/// same type.
pub fn map_utf8_utf8_type() -> DataType {
    DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Utf8, true),
            ])),
            false,
        )),
        false,
    )
}

/// Whether a map's entries `DataType` is a `Struct` of a string key and a
/// string value (the shape `COLUMN_TYPE_MAP` accepts), regardless of the field
/// names or their nullability.
fn is_utf8_utf8_entries(entries: &DataType) -> bool {
    matches!(
        entries,
        DataType::Struct(fields)
            if fields.len() == 2
                && fields[0].data_type() == &DataType::Utf8
                && fields[1].data_type() == &DataType::Utf8
    )
}

/// Map a `ColumnType` to its **storage** Arrow `DataType`.
///
/// The storage canonical unit for `COLUMN_TYPE_TIMESTAMP_MS` is MICROSECOND,
/// not millisecond. The proto/wire logical type is named `..._MS` (clients send
/// `Timestamp(Millisecond)`), but the reference duckdb store persisted parquet
/// with duckdb's native microsecond `TIMESTAMP`, so every segment on disk is
/// `Timestamp(Microsecond, None)`. We canonicalize storage to match the data on
/// disk: reads/adoption/compaction are then cast-free and lossless, and the
/// write path casts the millisecond wire form up to microsecond at ingress
/// (see [`validate_and_align_batch`]).
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
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        }
        ColumnType::COLUMN_TYPE_BYTES => Some(DataType::Binary),
        ColumnType::COLUMN_TYPE_MAP => Some(map_utf8_utf8_type()),
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
        let mut column = Column::new(name, ctype, c.nullable.unwrap_or(false));
        column.index.trigram = c
            .index
            .as_option()
            .and_then(|ix| ix.trigram)
            .unwrap_or(false);
        cols.push(column);
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
            ProtoColumn {
                index: MessageField::some(
                    ProtoColumnIndex::default().with_trigram(c.index.trigram),
                ),
                ..Default::default()
            }
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

#[derive(Serialize, Deserialize, Default)]
struct JsonColumnIndex {
    #[serde(default)]
    trigram: bool,
}

#[derive(Serialize, Deserialize)]
struct JsonColumn {
    name: String,
    /// Proto enum *name* (e.g. "COLUMN_TYPE_STRING"); stable across edits.
    r#type: String,
    nullable: bool,
    /// Absent in catalog rows written before column indexes existed;
    /// `serde(default)` decodes those as an empty (unindexed) `ColumnIndex`.
    #[serde(default)]
    index: JsonColumnIndex,
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
        ColumnType::COLUMN_TYPE_MAP => "COLUMN_TYPE_MAP",
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
        "map" => Some(ColumnType::COLUMN_TYPE_MAP),
        "COLUMN_TYPE_UNKNOWN" => Some(ColumnType::COLUMN_TYPE_UNKNOWN),
        "COLUMN_TYPE_STRING" => Some(ColumnType::COLUMN_TYPE_STRING),
        "COLUMN_TYPE_INT64" => Some(ColumnType::COLUMN_TYPE_INT64),
        "COLUMN_TYPE_FLOAT64" => Some(ColumnType::COLUMN_TYPE_FLOAT64),
        "COLUMN_TYPE_BOOL" => Some(ColumnType::COLUMN_TYPE_BOOL),
        "COLUMN_TYPE_TIMESTAMP_MS" => Some(ColumnType::COLUMN_TYPE_TIMESTAMP_MS),
        "COLUMN_TYPE_BYTES" => Some(ColumnType::COLUMN_TYPE_BYTES),
        "COLUMN_TYPE_INT32" => Some(ColumnType::COLUMN_TYPE_INT32),
        "COLUMN_TYPE_MAP" => Some(ColumnType::COLUMN_TYPE_MAP),
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
                index: JsonColumnIndex {
                    trigram: c.index.trigram,
                },
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
        let mut column = Column::new(c.name, column_type_from_json(&c.r#type)?, c.nullable);
        column.index.trigram = c.index.trigram;
        cols.push(column);
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

/// Return `schema` with the implicit nullable STRING `cluster` column appended.
/// No-op if a `cluster` column is already declared.
pub fn with_implicit_cluster(schema: Schema) -> Schema {
    if schema
        .columns
        .iter()
        .any(|c| c.name == IMPLICIT_CLUSTER_COLUMN)
    {
        return schema;
    }
    let Schema {
        mut columns,
        key_column,
    } = schema;
    columns.push(Column::new(
        IMPLICIT_CLUSTER_COLUMN,
        ColumnType::COLUMN_TYPE_STRING,
        true,
    ));
    Schema::new(columns, key_column)
}

/// Resolve the ordering key column name, raising if invalid.
///
/// If `key_column` is set it must name an existing column; otherwise the schema
/// must contain a `timestamp_ms` column. The only type rule enforced is that the
/// key may not be a `MAP` column: compaction sorts the key through
/// `RowConverter`/`lexsort`, which cannot order a nested map, so a map key would
/// wedge every compaction. The proto comment's INT64/TIMESTAMP_MS rule is
/// otherwise deliberately not enforced (a STRING key is accepted).
pub fn resolve_key_column(schema: &Schema) -> Result<String, StatsError> {
    let resolved = if !schema.key_column.is_empty() {
        if schema.column(&schema.key_column).is_none() {
            return Err(StatsError::SchemaValidation(format!(
                "key_column={:?} is not present in the schema columns",
                schema.key_column
            )));
        }
        schema.key_column.clone()
    } else {
        if schema.column(IMPLICIT_KEY_COLUMN).is_none() {
            return Err(StatsError::SchemaValidation(format!(
                "schema declares no key_column and has no implicit '{IMPLICIT_KEY_COLUMN}' column"
            )));
        }
        IMPLICIT_KEY_COLUMN.to_string()
    };
    if schema.column(&resolved).map(|c| c.r#type) == Some(ColumnType::COLUMN_TYPE_MAP) {
        return Err(StatsError::SchemaValidation(format!(
            "key_column={resolved:?} is a MAP column, which cannot be an ordering key"
        )));
    }
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// Schema merge (additive-only).
// ---------------------------------------------------------------------------

/// Return the effective schema for a re-register against `registered`.
///
/// - identical / requested ⊆ registered -> `registered` unchanged.
/// - requested adds nullable columns -> the union (registered then new).
/// - a conflicting column *type* -> `SchemaConflict`.
/// - a new non-nullable column -> `SchemaConflict`.
/// - a nullability difference on an existing column is *not* a conflict: warn
///   and keep the registered nullability (adopt-from-disk widens compacted
///   columns to nullable, and re-registration with the original schema must be
///   accepted, not rejected).
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
                // A nullability difference on an existing column is NOT a conflict.
                // Adopt-from-disk widens every compacted column to nullable
                // (DuckDB's COPY drops Arrow non-nullability), and the adopt
                // design relies on a later RegisterTable with the original
                // non-nullable schema being accepted rather than rejected. Keep
                // the registered nullability (the per-batch align path enforces
                // column presence against it); treating this as a conflict would
                // permanently wedge every namespace with non-nullable columns
                // once a compacted segment is adopted.
                if existing.nullable != rc.nullable {
                    tracing::warn!(
                        column = %rc.name,
                        registered = existing.nullable,
                        requested = rc.nullable,
                        "register: nullability differs — keeping registered",
                    );
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
/// their value type and rejecting unsupported nested/union types.
///
/// Dictionary-encoded columns are accepted transparently (the *value* type is
/// reported). A `Map<Utf8,Utf8>` maps to `COLUMN_TYPE_MAP` regardless of its
/// entries/key/value field names or sorted flag (so a parquet-round-tripped or
/// non-pyarrow map still decodes); a map with any other key/value type is
/// rejected. list/large-list/struct/union and any other unsupported type are
/// rejected.
pub fn arrow_to_column_type(dt: &DataType) -> Result<ColumnType, StatsError> {
    match dt {
        DataType::Dictionary(_, value) => arrow_to_column_type(value),
        DataType::Map(field, _) if is_utf8_utf8_entries(field.data_type()) => {
            Ok(ColumnType::COLUMN_TYPE_MAP)
        }
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
        // Any tz-naive timestamp maps to the single logical TIMESTAMP_MS type,
        // regardless of physical unit: microsecond from disk (the duckdb-written
        // storage canonical), millisecond from the wire. The physical unit is
        // reconciled to the storage canonical at the boundaries (`arrow_type_for`
        // + `validate_and_align_batch`). A tz-aware column has no proto analogue
        // and falls through to the unsupported-type error.
        DataType::Timestamp(_, None) => Ok(ColumnType::COLUMN_TYPE_TIMESTAMP_MS),
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

/// Sum the raw buffer bytes an array occupies, recursing into child data so a
/// nested column (e.g. a `Map`'s key/value buffers) is counted in full rather
/// than only its top-level offset/validity buffers. A monotone approximation
/// feeding the flush-trigger accounting.
fn array_buffer_size(arr: &ArrayRef) -> i64 {
    fn data_buffer_size(data: &ArrayData) -> i64 {
        let own: i64 = data.buffers().iter().map(|b| b.len() as i64).sum();
        let children: i64 = data.child_data().iter().map(data_buffer_size).sum();
        own + children
    }
    data_buffer_size(&arr.to_data())
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
                // Same logical ColumnType, but the physical Arrow unit may differ
                // (e.g. a millisecond wire timestamp vs the microsecond storage
                // canonical). Cast to the storage type so the appended array
                // matches its declared field; a no-op when already equal.
                let storage_array = if *actual_dt == arrow_dt {
                    Arc::clone(array)
                } else {
                    cast(array, &arrow_dt).map_err(|e| {
                        StatsError::SchemaValidation(format!(
                            "column {:?}: cannot cast batch type {actual_dt:?} to storage type \
                             {arrow_dt:?}: {e}",
                            col.name
                        ))
                    })?
                };
                byte_size += array_buffer_size(&storage_array);
                aligned_arrays.push(storage_array);
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
    fn arrow_type_map_covers_all_column_types() {
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
        // Storage canonical for TIMESTAMP_MS is microsecond (matches the
        // duckdb-written parquet on disk), not the wire's millisecond.
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_TIMESTAMP_MS),
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_BYTES),
            Some(DataType::Binary)
        );
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_MAP),
            Some(map_utf8_utf8_type())
        );
        assert_eq!(arrow_type_for(ColumnType::COLUMN_TYPE_UNKNOWN), None);
    }

    /// The canonical `Map<Utf8,Utf8>` storage type is byte-identical to pyarrow's
    /// `pa.map_(pa.string(), pa.string())` wire form: entries/key/value field
    /// names, a non-nullable key, a nullable value, and the unsorted flag. Any
    /// drift here forces a cast (or an outright reject) at the write and
    /// compaction DataType-equality gates, so pin the exact shape.
    #[test]
    fn map_type_matches_pyarrow_wire_form() {
        let DataType::Map(entries, sorted) = map_utf8_utf8_type() else {
            panic!("COLUMN_TYPE_MAP must be a Map DataType");
        };
        assert!(!sorted, "the map is unsorted (pyarrow keysSorted=false)");
        assert_eq!(entries.name(), "entries");
        assert!(
            !entries.is_nullable(),
            "the entries struct field is non-null"
        );
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("map entries must be a Struct");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "key");
        assert_eq!(fields[0].data_type(), &DataType::Utf8);
        assert!(!fields[0].is_nullable(), "the key is non-null");
        assert_eq!(fields[1].name(), "value");
        assert_eq!(fields[1].data_type(), &DataType::Utf8);
        assert!(fields[1].is_nullable(), "the value is nullable");
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
    fn with_implicit_cluster_appends_nullable_string_cluster_and_is_idempotent() {
        let stored = with_implicit_cluster(worker_schema());
        let last = stored.columns.last().unwrap();
        assert_eq!(last.name, "cluster");
        assert_eq!(last.r#type, ColumnType::COLUMN_TYPE_STRING);
        assert!(last.nullable, "the implicit cluster column is nullable");
        // The rest of the schema is untouched, and the key column is preserved.
        assert_eq!(
            stored.column_names(),
            vec!["worker_id", "mem_bytes", "timestamp_ms", "cluster"]
        );
        let again = with_implicit_cluster(stored.clone());
        assert_eq!(again, stored, "re-applying it does not add a second column");
    }

    #[test]
    fn with_implicit_cluster_keeps_a_declared_cluster_column_as_is() {
        // A schema that already declares `cluster` (e.g. the privileged `log`
        // namespace, whose writer supplies it) is left exactly as it is — the
        // column is not moved, duplicated, or re-typed.
        let declared = Schema::new(
            vec![
                col("cluster", ColumnType::COLUMN_TYPE_STRING, false),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        assert_eq!(with_implicit_cluster(declared.clone()), declared);
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
    fn resolve_key_column_rejects_a_map_key() {
        // A MAP column cannot be the ordering key (compaction can't sort a map).
        let s = Schema::new(
            vec![
                col("labels", ColumnType::COLUMN_TYPE_MAP, false),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "labels",
        );
        assert!(matches!(
            resolve_key_column(&s),
            Err(StatsError::SchemaValidation(_))
        ));
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
    fn merge_nullability_difference_keeps_registered() {
        // A re-register that flips an existing column's nullability is accepted
        // (not a conflict) and keeps the registered nullability. This is the
        // path that un-wedges a namespace whose compacted segments were adopted
        // as all-nullable: the client re-registers with the original
        // non-nullable schema and the merge succeeds as a no-op.
        let reg = with_implicit_seq(worker_schema());
        let widened = Schema::new(
            vec![col("mem_bytes", ColumnType::COLUMN_TYPE_INT64, true)],
            "",
        );
        assert_eq!(merge_schemas(&reg, &widened).unwrap(), reg);

        let narrowed = Schema::new(
            vec![col("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false)],
            "",
        );
        let reg_nullable = Schema::new(
            vec![
                col("seq", ColumnType::COLUMN_TYPE_INT64, false),
                col("worker_id", ColumnType::COLUMN_TYPE_STRING, true),
                col("mem_bytes", ColumnType::COLUMN_TYPE_INT64, true),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, true),
            ],
            "",
        );
        assert_eq!(
            merge_schemas(&reg_nullable, &narrowed).unwrap(),
            reg_nullable
        );
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
    fn arrow_to_column_type_round_trips_all_types() {
        for t in [
            ColumnType::COLUMN_TYPE_STRING,
            ColumnType::COLUMN_TYPE_INT64,
            ColumnType::COLUMN_TYPE_INT32,
            ColumnType::COLUMN_TYPE_FLOAT64,
            ColumnType::COLUMN_TYPE_BOOL,
            ColumnType::COLUMN_TYPE_TIMESTAMP_MS,
            ColumnType::COLUMN_TYPE_BYTES,
            ColumnType::COLUMN_TYPE_MAP,
        ] {
            let dt = arrow_type_for(t).unwrap();
            assert_eq!(arrow_to_column_type(&dt).unwrap(), t);
        }
        // Storage canonical for the logical timestamp type is microsecond, with
        // no timezone.
        assert_eq!(
            arrow_type_for(ColumnType::COLUMN_TYPE_TIMESTAMP_MS).unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        // Any tz-naive timestamp unit (microsecond from disk, millisecond from
        // the wire) maps back to the single logical TIMESTAMP_MS type.
        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            assert_eq!(
                arrow_to_column_type(&DataType::Timestamp(unit, None)).unwrap(),
                ColumnType::COLUMN_TYPE_TIMESTAMP_MS
            );
        }
        // A tz-aware timestamp has no proto analogue and is rejected.
        assert!(arrow_to_column_type(&DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into())
        ))
        .is_err());
    }

    #[test]
    fn validate_casts_millisecond_wire_timestamp_to_microsecond_storage() {
        use arrow::array::{TimestampMicrosecondArray, TimestampMillisecondArray};
        let registered = with_implicit_seq(Schema::new(
            vec![col("ts", ColumnType::COLUMN_TYPE_TIMESTAMP_MS, false)],
            "",
        ));
        // The client sends the wire form: Timestamp(Millisecond).
        let b = batch(
            vec![Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            )],
            vec![Arc::new(TimestampMillisecondArray::from(vec![
                1_700_000_000_000_i64,
                1_700_000_000_001,
            ]))],
        );
        let aligned = validate_and_align_batch(&b, &registered).unwrap();
        // Stored as the microsecond canonical; values scaled up by 1000.
        assert_eq!(
            aligned.fields[0].data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let arr = aligned.arrays[0]
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(
            arr.values(),
            &[1_700_000_000_000_000_i64, 1_700_000_000_001_000]
        );
    }

    #[test]
    fn validate_accepts_microsecond_disk_timestamp_unchanged() {
        use arrow::array::TimestampMicrosecondArray;
        let registered = with_implicit_seq(Schema::new(
            vec![col("ts", ColumnType::COLUMN_TYPE_TIMESTAMP_MS, false)],
            "",
        ));
        // A microsecond batch (the storage canonical, e.g. a re-ingested legacy
        // row) passes through without a value change.
        let b = batch(
            vec![Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            )],
            vec![Arc::new(TimestampMicrosecondArray::from(vec![
                1_700_000_000_000_000_i64,
            ]))],
        );
        let aligned = validate_and_align_batch(&b, &registered).unwrap();
        let arr = aligned.arrays[0]
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(arr.values(), &[1_700_000_000_000_000_i64]);
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

    // -----------------------------------------------------------------------
    // COLUMN_TYPE_MAP (Map<Utf8,Utf8>).
    // -----------------------------------------------------------------------

    /// A `Map<Utf8,Utf8>` array in the canonical (pyarrow) shape: entries/key/
    /// value field names, a non-null key and a nullable value. Each row is a list
    /// of `(key, Option<value>)` pairs.
    fn canonical_map_array(rows: Vec<Vec<(&str, Option<&str>)>>) -> ArrayRef {
        use arrow::array::{MapBuilder, MapFieldNames, StringBuilder};
        let names = MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        };
        let mut b = MapBuilder::new(Some(names), StringBuilder::new(), StringBuilder::new());
        for row in rows {
            for (k, v) in row {
                b.keys().append_value(k);
                b.values().append_option(v);
            }
            b.append(true).unwrap();
        }
        let map = b.finish();
        // The builder must produce exactly the declared storage type.
        assert_eq!(map.data_type(), &map_utf8_utf8_type());
        Arc::new(map)
    }

    #[test]
    fn arrow_to_column_type_accepts_any_utf8_map_and_rejects_others() {
        use arrow::datatypes::Int64Type;
        // arrow-rs's default MapBuilder names ("entries"/"keys"/"values") and
        // parquet's native ("key_value"/"key"/"value") both decode to MAP —
        // arrow_to_column_type is field-name and nullability agnostic.
        for (entry, key, value) in [("entries", "keys", "values"), ("key_value", "key", "value")] {
            let dt = DataType::Map(
                Arc::new(Field::new(
                    entry,
                    DataType::Struct(Fields::from(vec![
                        Field::new(key, DataType::Utf8, false),
                        Field::new(value, DataType::Utf8, true),
                    ])),
                    false,
                )),
                false,
            );
            assert_eq!(
                arrow_to_column_type(&dt).unwrap(),
                ColumnType::COLUMN_TYPE_MAP
            );
        }
        // A map whose value is not a string is unsupported.
        let int_valued = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int64, true),
                ])),
                false,
            )),
            false,
        );
        assert!(arrow_to_column_type(&int_valued).is_err());
        // A list is still rejected (proves the Map arm didn't widen the reject).
        let _ = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1_i64)])]);
    }

    #[test]
    fn array_buffer_size_counts_map_child_buffers() {
        // The byte estimate must recurse into a map's key/value child buffers,
        // not just its top-level offset/validity buffers (otherwise the RAM
        // flush-trigger under-accounts map columns). One row with keys
        // "scope"/"region" and values "fleet"/"us-east" = 23 bytes of string
        // data alone, above the ~8-byte top-level map offset buffer.
        let map = canonical_map_array(vec![vec![
            ("scope", Some("fleet")),
            ("region", Some("us-east")),
        ]]);
        assert!(
            array_buffer_size(&map) >= 23,
            "map byte size must include child key/value bytes"
        );
    }

    #[test]
    fn map_column_survives_parquet_round_trip() {
        // Writing the canonical map with arrow-rs's parquet writer and reading it
        // back yields a byte-identical DataType (the writer embeds the Arrow
        // schema, so entries/key/value names, key/value nullability, and the
        // sorted flag round-trip). This is what keeps the compaction
        // `project_to_schema` full-DataType-equality gate from rejecting a
        // re-read segment.
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::arrow::ArrowWriter;

        let field = Field::new("labels", map_utf8_utf8_type(), true);
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        let map = canonical_map_array(vec![
            vec![("scope", Some("fleet")), ("region", Some("us-east"))],
            vec![],
        ]);
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![map]).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let mut w = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(buf))
            .unwrap()
            .build()
            .unwrap();
        let back: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        let read_type = back[0].schema().field(0).data_type().clone();
        assert_eq!(read_type, map_utf8_utf8_type());
        assert_eq!(
            arrow_to_column_type(&read_type).unwrap(),
            ColumnType::COLUMN_TYPE_MAP
        );
    }

    #[test]
    fn align_accepts_native_map_column() {
        // A registered nullable MAP column accepts a canonical MapArray batch and
        // passes it through as the canonical storage type (no cast).
        let registered = with_implicit_seq(Schema::new(
            vec![
                Column::new("labels", ColumnType::COLUMN_TYPE_MAP, true),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ));
        let map = canonical_map_array(vec![vec![("scope", Some("fleet"))], vec![("region", None)]]);
        let b = batch(
            vec![
                Field::new("labels", map_utf8_utf8_type(), true),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            vec![map, Arc::new(Int64Array::from(vec![1_i64, 2]))],
        );
        let aligned = validate_and_align_batch(&b, &registered).unwrap();
        assert_eq!(aligned.arrays[0].data_type(), &map_utf8_utf8_type());
        assert_eq!(aligned.num_rows, 2);
    }

    #[test]
    fn align_missing_nullable_map_null_fills() {
        // A nullable MAP column absent from the batch is NULL-filled with a
        // typed empty MapArray (exercises new_null_array over a Map DataType).
        let registered = with_implicit_seq(Schema::new(
            vec![
                Column::new("labels", ColumnType::COLUMN_TYPE_MAP, true),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ));
        let b = batch(
            vec![Field::new("timestamp_ms", DataType::Int64, false)],
            vec![Arc::new(Int64Array::from(vec![1_i64]))],
        );
        let aligned = validate_and_align_batch(&b, &registered).unwrap();
        let labels = &aligned.arrays[0];
        assert_eq!(labels.data_type(), &map_utf8_utf8_type());
        assert_eq!(labels.len(), 1);
        assert_eq!(labels.null_count(), 1);
    }

    #[test]
    fn map_column_json_round_trips() {
        let stored = with_implicit_seq(Schema::new(
            vec![
                col("labels", ColumnType::COLUMN_TYPE_MAP, true),
                col("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ));
        let json = schema_to_json(&stored);
        assert!(json.contains("COLUMN_TYPE_MAP"));
        assert_eq!(schema_from_json(&json).unwrap(), stored);
    }
}
