//! Validation and canonicalization for immutable table specifications.

use std::collections::BTreeSet;

use buffa::{MessageField, MessageView};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    ArtifactPolicy, L0Mode, RemoteRetentionPolicy, SourceLayout, TableSpec as ProtoTableSpec,
    TableSpecView,
};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_from_proto_view, schema_to_proto_owned, Schema};
use crate::store::segment::MAX_ROW_GROUP_ROWS;

pub const DEFAULT_TARGET_OBJECT_BYTES: u64 = 256 * 1024 * 1024;
pub const DEFAULT_MAX_QUERY_TIME_MS: u64 = 10 * 60 * 1_000;

/// How long a newly activated definition stays rollbackable, and how long the
/// definition it replaced keeps its retired objects, when a specification does
/// not state a window of its own.
pub const DEFAULT_ROLLBACK_WINDOW_MS: u64 = 60 * 60 * 1_000;

/// Buffered-byte size at which an append forces an early flush, short-circuiting
/// the flush-rate cooldown so a write burst can't buffer unboundedly (and bounds
/// a single L0's size).
pub const SEGMENT_TARGET_BYTES: i64 = 100 * 1024 * 1024;

/// Maximum idle gap before a buffer is flushed on age alone. With steady writes
/// the per-append nudge drives flushes; this is the ceiling for a quiet table.
pub const DEFAULT_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// The operating knobs a table's specification resolves to.
///
/// Registration materializes every default into the stored specification
/// ([`normalize_operating_policy`] runs before the spec is persisted), so
/// resolving a registered specification is plain field reads. The fallbacks
/// here cover the one real absence: a legacy table with no specification.
#[derive(Debug, Clone)]
pub struct TablePolicy {
    pub l0_mode: L0Mode,
    pub table_spec_version: u64,
    pub max_buffer_bytes: i64,
    pub max_flush_age: std::time::Duration,
    pub max_query_time_ms: u64,
    pub rollback_window_ms: u64,
    pub target_object_bytes: i64,
    pub source_layout: Option<SourceLayout>,
}

impl Default for TablePolicy {
    fn default() -> Self {
        Self {
            l0_mode: L0Mode::L0_MODE_LEGACY_LOCAL,
            table_spec_version: 0,
            max_buffer_bytes: SEGMENT_TARGET_BYTES,
            max_flush_age: DEFAULT_FLUSH_INTERVAL,
            max_query_time_ms: DEFAULT_MAX_QUERY_TIME_MS,
            rollback_window_ms: DEFAULT_ROLLBACK_WINDOW_MS,
            target_object_bytes: DEFAULT_TARGET_OBJECT_BYTES as i64,
            source_layout: None,
        }
    }
}

impl TablePolicy {
    /// Resolve a specification's operating policy; `None` is a legacy table
    /// running under the defaults.
    pub fn resolve(spec: Option<&ProtoTableSpec>) -> Self {
        let Some(spec) = spec else {
            return Self::default();
        };
        let Some(operating) = spec.operating_policy.as_option() else {
            return Self::default();
        };
        let l0_mode = operating
            .l0_mode
            .and_then(|mode| mode.as_known())
            .filter(|mode| *mode != L0Mode::L0_MODE_UNSPECIFIED)
            .unwrap_or(L0Mode::L0_MODE_LEGACY_LOCAL);
        Self {
            l0_mode,
            table_spec_version: spec.version.unwrap_or(0),
            max_buffer_bytes: i64::try_from(
                operating
                    .max_buffer_bytes
                    .unwrap_or(SEGMENT_TARGET_BYTES as u64),
            )
            .unwrap_or(i64::MAX),
            max_flush_age: std::time::Duration::from_millis(
                operating
                    .max_flush_age_ms
                    .unwrap_or(DEFAULT_FLUSH_INTERVAL.as_millis() as u64),
            ),
            max_query_time_ms: max_query_time_ms(spec),
            rollback_window_ms: rollback_window_ms(spec),
            target_object_bytes: spec
                .source_layout
                .as_option()
                .and_then(|layout| layout.target_object_bytes)
                .and_then(|bytes| i64::try_from(bytes).ok())
                .unwrap_or(DEFAULT_TARGET_OBJECT_BYTES as i64),
            source_layout: spec.source_layout.as_option().cloned(),
        }
    }

    /// Whether this table's L0 is written as immutable objects rather than local
    /// files.
    pub fn object_backed(&self) -> bool {
        self.l0_mode == L0Mode::L0_MODE_OBJECT_STORE
    }
}

/// Whether activating `next` requires rewriting the table's existing rows
/// into its physical layout; `false` activates it in the registration's own
/// state commit.
///
/// `active` is `None` for a table's first versioned definition; a table that
/// already holds rows under it is on version 0 and its history is imported
/// through the same rewrite as any other layout change.
///
/// A change that no online migration can express — a different ordering key, or
/// a logical schema that existing rows cannot be read under — is rejected here
/// rather than recorded as a transition. Only additive schema changes are
/// query-compatible: an object written under the old definition must still
/// answer queries planned against the new one.
pub fn definition_requires_rewrite(
    active: Option<&ProtoTableSpec>,
    next: &ProtoTableSpec,
    has_rows: bool,
) -> Result<bool, StatsError> {
    let Some(active) = active else {
        // Version 0 has no recorded definition, so there is nothing to compare
        // against. Its history still has to be rewritten into version 1's
        // layout before that version can answer queries.
        return Ok(has_rows);
    };
    check_logical_compatibility(active, next)?;
    Ok(has_rows && active.source_layout != next.source_layout)
}

/// Reject a logical schema change that existing objects cannot serve.
fn check_logical_compatibility(
    active: &ProtoTableSpec,
    next: &ProtoTableSpec,
) -> Result<(), StatsError> {
    let (Some(active_schema), Some(next_schema)) = (
        active.logical_schema.as_option(),
        next.logical_schema.as_option(),
    ) else {
        return Err(StatsError::SchemaValidation(
            "table_spec.logical_schema is required on both definition versions".to_string(),
        ));
    };
    let active_key = active_schema.key_column.as_deref().unwrap_or("");
    let next_key = next_schema.key_column.as_deref().unwrap_or("");
    if active_key != next_key {
        return Err(StatsError::SchemaConflict(format!(
            "table_spec key column {active_key:?} cannot change to {next_key:?}; \
             an incompatible logical change is not migratable online"
        )));
    }
    for active_column in &active_schema.columns {
        let name = active_column.name.as_deref().unwrap_or("");
        let Some(next_column) = next_schema
            .columns
            .iter()
            .find(|column| column.name.as_deref().unwrap_or("") == name)
        else {
            return Err(StatsError::SchemaConflict(format!(
                "table_spec drops column {name:?}; an incompatible logical change is not \
                 migratable online"
            )));
        };
        if active_column.r#type != next_column.r#type {
            return Err(StatsError::SchemaConflict(format!(
                "table_spec changes the type of column {name:?}; an incompatible logical change \
                 is not migratable online"
            )));
        }
        if active_column.nullable.unwrap_or(false) && !next_column.nullable.unwrap_or(false) {
            return Err(StatsError::SchemaConflict(format!(
                "table_spec makes nullable column {name:?} required; an incompatible logical \
                 change is not migratable online"
            )));
        }
    }
    // A column the new definition adds must be readable as null on every object
    // written before it existed.
    for next_column in &next_schema.columns {
        let name = next_column.name.as_deref().unwrap_or("");
        let is_new = !active_schema
            .columns
            .iter()
            .any(|column| column.name.as_deref().unwrap_or("") == name);
        if is_new && !next_column.nullable.unwrap_or(false) {
            return Err(StatsError::SchemaConflict(format!(
                "table_spec adds required column {name:?}; existing rows cannot supply it"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ValidatedTableSpec {
    pub spec: ProtoTableSpec,
    pub schema: Schema,
    pub cache_policy: StoragePolicy,
    pub hash: [u8; 32],
    pub l0_mode: L0Mode,
}

impl ValidatedTableSpec {
    pub fn from_view(
        view: &TableSpecView<'_>,
        request_schema: &Schema,
        request_policy: &StoragePolicy,
    ) -> Result<Self, StatsError> {
        if view.version.unwrap_or(0) == 0 {
            return Err(StatsError::SchemaValidation(
                "table_spec.version must be at least 1".to_string(),
            ));
        }
        let logical_schema = view.logical_schema.as_option().ok_or_else(|| {
            StatsError::SchemaValidation("table_spec.logical_schema is required".to_string())
        })?;
        let schema = schema_from_proto_view(logical_schema)?;
        if &schema != request_schema {
            return Err(StatsError::SchemaValidation(
                "table_spec.logical_schema must equal register_table.schema during mixed-version rollout"
                    .to_string(),
            ));
        }

        let mut spec = view.to_owned_message();
        normalize_source_layout(&mut spec, &schema)?;
        normalize_artifact_policy(&mut spec, &schema)?;
        let (cache_policy, l0_mode) = normalize_operating_policy(&mut spec, request_policy)?;
        let hash: [u8; 32] = Sha256::digest(canonical_json_bytes(&spec)?).into();
        Ok(Self {
            spec,
            schema,
            cache_policy,
            hash,
            l0_mode,
        })
    }
}

pub fn canonical_json_bytes(spec: &ProtoTableSpec) -> Result<Vec<u8>, StatsError> {
    serde_json::to_vec(spec)
        .map_err(|error| StatsError::Internal(format!("serialize table specification: {error}")))
}

pub fn table_spec_from_json(json: &str) -> Result<ProtoTableSpec, StatsError> {
    serde_json::from_str(json)
        .map_err(|error| StatsError::Internal(format!("decode table specification: {error}")))
}

fn normalize_source_layout(spec: &mut ProtoTableSpec, schema: &Schema) -> Result<(), StatsError> {
    let layout = spec.source_layout.get_or_insert_default();
    if layout.sort_columns.is_empty() {
        layout.sort_columns = schema.sort_columns.clone();
    } else if layout.sort_columns != schema.sort_columns {
        return Err(StatsError::SchemaValidation(
            "table_spec.source_layout.sort_columns must equal logical_schema.sort_columns"
                .to_string(),
        ));
    }

    let schema_row_group_rows = if schema.max_row_group_rows == 0 {
        MAX_ROW_GROUP_ROWS as u32
    } else {
        schema.max_row_group_rows
    };
    match layout.max_row_group_rows.unwrap_or(0) {
        0 => layout.max_row_group_rows = Some(schema_row_group_rows),
        value if value != schema_row_group_rows => {
            return Err(StatsError::SchemaValidation(format!(
                "table_spec.source_layout.max_row_group_rows {value} does not match logical_schema value {schema_row_group_rows}"
            )));
        }
        _ => {}
    }
    if layout.target_object_bytes.unwrap_or(0) == 0 {
        layout.target_object_bytes = Some(DEFAULT_TARGET_OBJECT_BYTES);
    }
    validate_partition(layout, schema)
}

fn validate_partition(layout: &SourceLayout, schema: &Schema) -> Result<(), StatsError> {
    let Some(partition) = layout.partition.as_option() else {
        return Ok(());
    };
    let spec_id = partition.spec_id.unwrap_or(0);
    if spec_id == 0 || u32::try_from(spec_id).is_err() {
        return Err(StatsError::SchemaValidation(format!(
            "partition spec_id {spec_id} must be between 1 and {}",
            u32::MAX
        )));
    }
    let mut names = BTreeSet::new();
    for field in &partition.fields {
        let source = field.source_column.as_deref().unwrap_or("");
        let name = field.name.as_deref().unwrap_or("");
        let Some(source_column) = schema.column(source) else {
            return Err(StatsError::SchemaValidation(format!(
                "partition source column {source:?} is not in logical_schema"
            )));
        };
        if matches!(
            source_column.r#type,
            crate::proto::finelog::stats::ColumnType::COLUMN_TYPE_BYTES
                | crate::proto::finelog::stats::ColumnType::COLUMN_TYPE_MAP
                | crate::proto::finelog::stats::ColumnType::COLUMN_TYPE_FLOAT64_LIST
                | crate::proto::finelog::stats::ColumnType::COLUMN_TYPE_INT64_LIST
                | crate::proto::finelog::stats::ColumnType::COLUMN_TYPE_UNKNOWN
        ) {
            return Err(StatsError::SchemaValidation(format!(
                "partition source column {source:?} must have a scalar string-renderable type"
            )));
        }
        if name.is_empty() || !names.insert(name.to_string()) {
            return Err(StatsError::SchemaValidation(format!(
                "partition field name {name:?} must be non-empty and unique"
            )));
        }
        match field.transform.as_ref() {
            Some(crate::proto::finelog::stats::partition_field::Transform::Identity(_)) => {}
            Some(crate::proto::finelog::stats::partition_field::Transform::Bucket(bucket))
                if bucket.buckets.unwrap_or(0) > 0 => {}
            Some(crate::proto::finelog::stats::partition_field::Transform::Bucket(_)) => {
                return Err(StatsError::SchemaValidation(format!(
                    "partition field {name:?} bucket count must be positive"
                )));
            }
            None => {
                return Err(StatsError::SchemaValidation(format!(
                    "partition field {name:?} requires a transform"
                )));
            }
        }
    }
    Ok(())
}

fn normalize_artifact_policy(spec: &mut ProtoTableSpec, schema: &Schema) -> Result<(), StatsError> {
    let expected_schema = schema_to_proto_owned(schema);
    let mut expected_indexes = expected_schema
        .columns
        .into_iter()
        .filter_map(|column| {
            let index = column.index.into_option()?;
            let has_index = index.trigram.unwrap_or(false)
                || index.value_counts.unwrap_or(false)
                || !index.exact_values.is_empty();
            has_index.then(|| crate::proto::finelog::stats::ColumnArtifactPolicy {
                column: column.name,
                index: MessageField::some(index),
                ..Default::default()
            })
        })
        .collect::<Vec<_>>();
    expected_indexes.sort_by(|left, right| left.column.cmp(&right.column));
    let mut expected_projections = expected_schema.projections;
    expected_projections.sort_by(|left, right| left.name.cmp(&right.name));
    let mut expected_extrema = expected_schema.grouped_extrema;
    expected_extrema.sort_by(|left, right| {
        (
            &left.filter_column,
            &left.group_json_column,
            &left.group_json_key,
            &left.extrema_column,
        )
            .cmp(&(
                &right.filter_column,
                &right.group_json_column,
                &right.group_json_key,
                &right.extrema_column,
            ))
    });

    if spec.artifact_policy.is_unset() {
        spec.artifact_policy = MessageField::some(ArtifactPolicy {
            revision: Some(1),
            indexes: expected_indexes,
            projections: expected_projections,
            grouped_extrema: expected_extrema,
            ..Default::default()
        });
        return Ok(());
    }

    let policy = spec.artifact_policy.get_or_insert_default();
    policy
        .indexes
        .sort_by(|left, right| left.column.cmp(&right.column));
    policy
        .projections
        .sort_by(|left, right| left.name.cmp(&right.name));
    policy.grouped_extrema.sort_by(|left, right| {
        (
            &left.filter_column,
            &left.group_json_column,
            &left.group_json_key,
            &left.extrema_column,
        )
            .cmp(&(
                &right.filter_column,
                &right.group_json_column,
                &right.group_json_key,
                &right.extrema_column,
            ))
    });
    if policy.indexes != expected_indexes
        || policy.projections != expected_projections
        || policy.grouped_extrema != expected_extrema
    {
        return Err(StatsError::SchemaValidation(
            "table_spec.artifact_policy must equal the index, projection, and grouped-extrema declarations in logical_schema"
                .to_string(),
        ));
    }
    if policy.revision.unwrap_or(0) == 0 {
        policy.revision = Some(1);
    }
    Ok(())
}

fn normalize_operating_policy(
    spec: &mut ProtoTableSpec,
    request_policy: &StoragePolicy,
) -> Result<(StoragePolicy, L0Mode), StatsError> {
    let policy = spec.operating_policy.get_or_insert_default();
    let configured_mode = policy
        .l0_mode
        .and_then(|value| value.as_known())
        .unwrap_or(L0Mode::L0_MODE_LEGACY_LOCAL);
    let l0_mode = if configured_mode == L0Mode::L0_MODE_UNSPECIFIED {
        L0Mode::L0_MODE_LEGACY_LOCAL
    } else {
        configured_mode
    };
    if l0_mode == L0Mode::L0_MODE_LOCAL_EPHEMERAL {
        return Err(StatsError::SchemaValidation(
            "table_spec local-ephemeral L0 is not supported in format version 1".to_string(),
        ));
    }
    policy.l0_mode = Some(l0_mode.into());

    if policy.local_cache.is_unset() {
        policy.local_cache = MessageField::some(request_policy.to_proto_owned());
    }
    let cache_policy = match policy.local_cache.as_option() {
        Some(proto) => StoragePolicy::from_proto_owned(proto),
        None => request_policy.clone(),
    };
    if !request_policy.is_empty() && &cache_policy != request_policy {
        return Err(StatsError::SchemaValidation(
            "table_spec.operating_policy.local_cache must equal register_table.storage_policy during mixed-version rollout"
                .to_string(),
        ));
    }

    if policy.remote_retention.is_unset() {
        policy.remote_retention = MessageField::some(RemoteRetentionPolicy {
            retain_forever: Some(true),
            ..Default::default()
        });
    }
    if !policy
        .remote_retention
        .as_option()
        .and_then(|retention| retention.retain_forever)
        .unwrap_or(false)
    {
        return Err(StatsError::SchemaValidation(
            "table_spec remote retention must be retain_forever in format version 1".to_string(),
        ));
    }
    if policy.max_buffer_bytes.unwrap_or(0) == 0 {
        policy.max_buffer_bytes = Some(SEGMENT_TARGET_BYTES as u64);
    }
    if policy.max_flush_age_ms.unwrap_or(0) == 0 {
        policy.max_flush_age_ms = Some(DEFAULT_FLUSH_INTERVAL.as_millis() as u64);
    }
    if policy.max_query_time_ms.unwrap_or(0) == 0 {
        policy.max_query_time_ms = Some(DEFAULT_MAX_QUERY_TIME_MS);
    }
    if policy.rollback_window_ms.unwrap_or(0) == 0 {
        policy.rollback_window_ms = Some(DEFAULT_ROLLBACK_WINDOW_MS);
    }
    Ok((cache_policy, l0_mode))
}

pub fn max_query_time_ms(spec: &ProtoTableSpec) -> u64 {
    spec.operating_policy
        .as_option()
        .and_then(|policy| policy.max_query_time_ms)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_QUERY_TIME_MS)
}

pub fn rollback_window_ms(spec: &ProtoTableSpec) -> u64 {
    spec.operating_policy
        .as_option()
        .and_then(|policy| policy.rollback_window_ms)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_ROLLBACK_WINDOW_MS)
}

#[cfg(test)]
mod tests {
    use buffa::{Message, MessageField, MessageView};

    use super::*;
    use crate::proto::finelog::stats::{ColumnType, TableSpecView};
    use crate::store::schema::{schema_to_proto_owned, Column};

    fn schema() -> Schema {
        Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false)
                    .with_trigram_index(),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        )
    }

    fn validate(spec: &ProtoTableSpec, policy: &StoragePolicy) -> ValidatedTableSpec {
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        ValidatedTableSpec::from_view(&view, &schema(), policy).unwrap()
    }

    #[test]
    fn validation_resolves_defaults_before_hashing() {
        let spec = ProtoTableSpec {
            version: Some(1),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema())),
            ..Default::default()
        };
        let first = validate(&spec, &StoragePolicy::default());
        let second = validate(&spec, &StoragePolicy::default());
        assert_eq!(first.hash, second.hash);
        assert_eq!(
            first
                .spec
                .source_layout
                .as_option()
                .and_then(|layout| layout.max_row_group_rows),
            Some(MAX_ROW_GROUP_ROWS as u32)
        );
        assert_eq!(
            first
                .spec
                .operating_policy
                .as_option()
                .and_then(|policy| policy.max_query_time_ms),
            Some(DEFAULT_MAX_QUERY_TIME_MS)
        );
        assert_eq!(first.l0_mode, L0Mode::L0_MODE_LEGACY_LOCAL);
        assert_eq!(
            first
                .spec
                .artifact_policy
                .as_option()
                .map(|policy| policy.indexes.len()),
            Some(1)
        );
    }

    #[test]
    fn mixed_version_registration_rejects_cache_policy_disagreement() {
        let spec = ProtoTableSpec {
            version: Some(1),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema())),
            operating_policy: MessageField::some(crate::proto::finelog::stats::OperatingPolicy {
                local_cache: MessageField::some(
                    StoragePolicy {
                        max_segments: Some(5),
                        ..Default::default()
                    }
                    .to_proto_owned(),
                ),
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        let error = ValidatedTableSpec::from_view(
            &view,
            &schema(),
            &StoragePolicy {
                max_segments: Some(6),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, StatsError::SchemaValidation(_)));
    }

    fn spec_for(version: u64, schema: &Schema, layout: SourceLayout) -> ProtoTableSpec {
        let spec = ProtoTableSpec {
            version: Some(version),
            logical_schema: MessageField::some(schema_to_proto_owned(schema)),
            source_layout: MessageField::some(layout),
            ..Default::default()
        };
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        ValidatedTableSpec::from_view(&view, schema, &StoragePolicy::default())
            .unwrap()
            .spec
    }

    #[test]
    fn an_unchanged_layout_is_metadata_only_even_over_existing_rows() {
        let active = spec_for(1, &schema(), SourceLayout::default());
        let mut next = spec_for(2, &schema(), SourceLayout::default());
        next.operating_policy
            .get_or_insert_default()
            .max_query_time_ms = Some(30_000);
        assert!(!definition_requires_rewrite(Some(&active), &next, true).unwrap());
    }

    #[test]
    fn a_layout_change_over_existing_rows_is_a_compatible_rewrite() {
        let active = spec_for(1, &schema(), SourceLayout::default());
        let next = spec_for(
            2,
            &schema(),
            SourceLayout {
                target_object_bytes: Some(8 * 1024 * 1024),
                ..Default::default()
            },
        );
        assert!(definition_requires_rewrite(Some(&active), &next, true).unwrap());
        // An empty table has nothing to rewrite.
        assert!(!definition_requires_rewrite(Some(&active), &next, false).unwrap());
    }

    #[test]
    fn version_zero_history_converts_through_the_same_rewrite() {
        let next = spec_for(1, &schema(), SourceLayout::default());
        assert!(definition_requires_rewrite(None, &next, true).unwrap());
        assert!(!definition_requires_rewrite(None, &next, false).unwrap());
    }

    #[test]
    fn incompatible_logical_changes_are_rejected() {
        let active = spec_for(1, &schema(), SourceLayout::default());

        let rekeyed = Schema::new(schema().columns, "worker_id");
        let error = definition_requires_rewrite(
            Some(&active),
            &spec_for(2, &rekeyed, SourceLayout::default()),
            true,
        )
        .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)), "{error}");

        let dropped = Schema::new(
            vec![Column::new(
                "timestamp_ms",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            )],
            "timestamp_ms",
        );
        let error = definition_requires_rewrite(
            Some(&active),
            &spec_for(2, &dropped, SourceLayout::default()),
            true,
        )
        .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)), "{error}");

        let retyped = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        );
        let error = definition_requires_rewrite(
            Some(&active),
            &spec_for(2, &retyped, SourceLayout::default()),
            true,
        )
        .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)), "{error}");

        let required_addition = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false)
                    .with_trigram_index(),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("region", ColumnType::COLUMN_TYPE_STRING, false),
            ],
            "timestamp_ms",
        );
        let error = definition_requires_rewrite(
            Some(&active),
            &spec_for(2, &required_addition, SourceLayout::default()),
            true,
        )
        .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)), "{error}");
    }

    #[test]
    fn a_nullable_column_may_be_added() {
        let active = spec_for(1, &schema(), SourceLayout::default());
        let extended = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false)
                    .with_trigram_index(),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("region", ColumnType::COLUMN_TYPE_STRING, true),
            ],
            "timestamp_ms",
        );
        assert!(!definition_requires_rewrite(
            Some(&active),
            &spec_for(2, &extended, SourceLayout::default()),
            true
        )
        .unwrap());
    }

    #[test]
    fn validation_rejects_unimplemented_ephemeral_l0() {
        let spec = ProtoTableSpec {
            version: Some(1),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema())),
            operating_policy: MessageField::some(crate::proto::finelog::stats::OperatingPolicy {
                l0_mode: Some(L0Mode::L0_MODE_LOCAL_EPHEMERAL.into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        let error =
            ValidatedTableSpec::from_view(&view, &schema(), &StoragePolicy::default()).unwrap_err();
        assert!(matches!(error, StatsError::SchemaValidation(_)));
    }
}
