//! Validation and canonicalization for immutable table specifications.

use std::collections::BTreeSet;

use buffa::{MessageField, MessageView};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    ArtifactPolicy, L0Mode, RemoteRetentionPolicy, SourceLayout, TableSpec as ProtoTableSpec,
    TableSpecView,
};
use crate::store::namespace::{DEFAULT_FLUSH_INTERVAL, SEGMENT_TARGET_BYTES};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_from_proto_view, schema_to_proto_owned, Schema};
use crate::store::segment::MAX_ROW_GROUP_ROWS;

pub const DEFAULT_TARGET_OBJECT_BYTES: u64 = 256 * 1024 * 1024;
pub const DEFAULT_MAX_QUERY_TIME_MS: u64 = 10 * 60 * 1_000;

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
    let cache_policy = StoragePolicy::from_proto_owned(
        policy
            .local_cache
            .as_option()
            .expect("local cache policy was initialized"),
    );
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
    Ok((cache_policy, l0_mode))
}

pub fn migration_phase_for_state(
    has_desired: bool,
) -> crate::proto::finelog::stats::MigrationPhase {
    if has_desired {
        crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_DUAL_WRITE
    } else {
        crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_ACTIVATED
    }
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
