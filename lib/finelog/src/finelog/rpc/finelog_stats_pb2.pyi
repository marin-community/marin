from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ColumnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COLUMN_TYPE_UNKNOWN: _ClassVar[ColumnType]
    COLUMN_TYPE_STRING: _ClassVar[ColumnType]
    COLUMN_TYPE_INT64: _ClassVar[ColumnType]
    COLUMN_TYPE_FLOAT64: _ClassVar[ColumnType]
    COLUMN_TYPE_BOOL: _ClassVar[ColumnType]
    COLUMN_TYPE_TIMESTAMP_MS: _ClassVar[ColumnType]
    COLUMN_TYPE_BYTES: _ClassVar[ColumnType]
    COLUMN_TYPE_INT32: _ClassVar[ColumnType]
    COLUMN_TYPE_MAP: _ClassVar[ColumnType]
    COLUMN_TYPE_FLOAT64_LIST: _ClassVar[ColumnType]
    COLUMN_TYPE_INT64_LIST: _ClassVar[ColumnType]

class L0Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    L0_MODE_UNSPECIFIED: _ClassVar[L0Mode]
    L0_MODE_LEGACY_LOCAL: _ClassVar[L0Mode]
    L0_MODE_OBJECT_STORE: _ClassVar[L0Mode]
    L0_MODE_LOCAL_EPHEMERAL: _ClassVar[L0Mode]

class MigrationPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MIGRATION_PHASE_UNSPECIFIED: _ClassVar[MigrationPhase]
    MIGRATION_PHASE_DUAL_WRITE: _ClassVar[MigrationPhase]
    MIGRATION_PHASE_BACKFILL: _ClassVar[MigrationPhase]
    MIGRATION_PHASE_VERIFY: _ClassVar[MigrationPhase]
    MIGRATION_PHASE_ACTIVATED: _ClassVar[MigrationPhase]
    MIGRATION_PHASE_OBSERVING: _ClassVar[MigrationPhase]
    MIGRATION_PHASE_RETIRED: _ClassVar[MigrationPhase]
COLUMN_TYPE_UNKNOWN: ColumnType
COLUMN_TYPE_STRING: ColumnType
COLUMN_TYPE_INT64: ColumnType
COLUMN_TYPE_FLOAT64: ColumnType
COLUMN_TYPE_BOOL: ColumnType
COLUMN_TYPE_TIMESTAMP_MS: ColumnType
COLUMN_TYPE_BYTES: ColumnType
COLUMN_TYPE_INT32: ColumnType
COLUMN_TYPE_MAP: ColumnType
COLUMN_TYPE_FLOAT64_LIST: ColumnType
COLUMN_TYPE_INT64_LIST: ColumnType
L0_MODE_UNSPECIFIED: L0Mode
L0_MODE_LEGACY_LOCAL: L0Mode
L0_MODE_OBJECT_STORE: L0Mode
L0_MODE_LOCAL_EPHEMERAL: L0Mode
MIGRATION_PHASE_UNSPECIFIED: MigrationPhase
MIGRATION_PHASE_DUAL_WRITE: MigrationPhase
MIGRATION_PHASE_BACKFILL: MigrationPhase
MIGRATION_PHASE_VERIFY: MigrationPhase
MIGRATION_PHASE_ACTIVATED: MigrationPhase
MIGRATION_PHASE_OBSERVING: MigrationPhase
MIGRATION_PHASE_RETIRED: MigrationPhase

class ColumnIndex(_message.Message):
    __slots__ = ("trigram", "exact_values", "value_counts")
    TRIGRAM_FIELD_NUMBER: _ClassVar[int]
    EXACT_VALUES_FIELD_NUMBER: _ClassVar[int]
    VALUE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    trigram: bool
    exact_values: _containers.RepeatedScalarFieldContainer[str]
    value_counts: bool
    def __init__(self, trigram: _Optional[bool] = ..., exact_values: _Optional[_Iterable[str]] = ..., value_counts: _Optional[bool] = ...) -> None: ...

class Column(_message.Message):
    __slots__ = ("name", "type", "nullable", "index")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NULLABLE_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: ColumnType
    nullable: bool
    index: ColumnIndex
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[ColumnType, str]] = ..., nullable: _Optional[bool] = ..., index: _Optional[_Union[ColumnIndex, _Mapping]] = ...) -> None: ...

class CoveringProjection(_message.Message):
    __slots__ = ("name", "predicate_column", "predicate_values", "columns")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_COLUMN_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_VALUES_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    name: str
    predicate_column: str
    predicate_values: _containers.RepeatedScalarFieldContainer[str]
    columns: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., predicate_column: _Optional[str] = ..., predicate_values: _Optional[_Iterable[str]] = ..., columns: _Optional[_Iterable[str]] = ...) -> None: ...

class GroupedExtrema(_message.Message):
    __slots__ = ("filter_column", "group_json_column", "group_json_key", "extrema_column")
    FILTER_COLUMN_FIELD_NUMBER: _ClassVar[int]
    GROUP_JSON_COLUMN_FIELD_NUMBER: _ClassVar[int]
    GROUP_JSON_KEY_FIELD_NUMBER: _ClassVar[int]
    EXTREMA_COLUMN_FIELD_NUMBER: _ClassVar[int]
    filter_column: str
    group_json_column: str
    group_json_key: str
    extrema_column: str
    def __init__(self, filter_column: _Optional[str] = ..., group_json_column: _Optional[str] = ..., group_json_key: _Optional[str] = ..., extrema_column: _Optional[str] = ...) -> None: ...

class Schema(_message.Message):
    __slots__ = ("columns", "key_column", "projections", "grouped_extrema", "sort_columns", "max_row_group_rows")
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    KEY_COLUMN_FIELD_NUMBER: _ClassVar[int]
    PROJECTIONS_FIELD_NUMBER: _ClassVar[int]
    GROUPED_EXTREMA_FIELD_NUMBER: _ClassVar[int]
    SORT_COLUMNS_FIELD_NUMBER: _ClassVar[int]
    MAX_ROW_GROUP_ROWS_FIELD_NUMBER: _ClassVar[int]
    columns: _containers.RepeatedCompositeFieldContainer[Column]
    key_column: str
    projections: _containers.RepeatedCompositeFieldContainer[CoveringProjection]
    grouped_extrema: _containers.RepeatedCompositeFieldContainer[GroupedExtrema]
    sort_columns: _containers.RepeatedScalarFieldContainer[str]
    max_row_group_rows: int
    def __init__(self, columns: _Optional[_Iterable[_Union[Column, _Mapping]]] = ..., key_column: _Optional[str] = ..., projections: _Optional[_Iterable[_Union[CoveringProjection, _Mapping]]] = ..., grouped_extrema: _Optional[_Iterable[_Union[GroupedExtrema, _Mapping]]] = ..., sort_columns: _Optional[_Iterable[str]] = ..., max_row_group_rows: _Optional[int] = ...) -> None: ...

class StoragePolicy(_message.Message):
    __slots__ = ("max_segments", "max_bytes", "max_age_seconds")
    MAX_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    max_segments: int
    max_bytes: int
    max_age_seconds: int
    def __init__(self, max_segments: _Optional[int] = ..., max_bytes: _Optional[int] = ..., max_age_seconds: _Optional[int] = ...) -> None: ...

class IdentityTransform(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BucketTransform(_message.Message):
    __slots__ = ("buckets",)
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    buckets: int
    def __init__(self, buckets: _Optional[int] = ...) -> None: ...

class PartitionField(_message.Message):
    __slots__ = ("source_column", "name", "identity", "bucket")
    SOURCE_COLUMN_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    source_column: str
    name: str
    identity: IdentityTransform
    bucket: BucketTransform
    def __init__(self, source_column: _Optional[str] = ..., name: _Optional[str] = ..., identity: _Optional[_Union[IdentityTransform, _Mapping]] = ..., bucket: _Optional[_Union[BucketTransform, _Mapping]] = ...) -> None: ...

class PartitionSpec(_message.Message):
    __slots__ = ("spec_id", "fields")
    SPEC_ID_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    spec_id: int
    fields: _containers.RepeatedCompositeFieldContainer[PartitionField]
    def __init__(self, spec_id: _Optional[int] = ..., fields: _Optional[_Iterable[_Union[PartitionField, _Mapping]]] = ...) -> None: ...

class SourceLayout(_message.Message):
    __slots__ = ("partition", "sort_columns", "max_row_group_rows", "target_object_bytes")
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    SORT_COLUMNS_FIELD_NUMBER: _ClassVar[int]
    MAX_ROW_GROUP_ROWS_FIELD_NUMBER: _ClassVar[int]
    TARGET_OBJECT_BYTES_FIELD_NUMBER: _ClassVar[int]
    partition: PartitionSpec
    sort_columns: _containers.RepeatedScalarFieldContainer[str]
    max_row_group_rows: int
    target_object_bytes: int
    def __init__(self, partition: _Optional[_Union[PartitionSpec, _Mapping]] = ..., sort_columns: _Optional[_Iterable[str]] = ..., max_row_group_rows: _Optional[int] = ..., target_object_bytes: _Optional[int] = ...) -> None: ...

class ColumnArtifactPolicy(_message.Message):
    __slots__ = ("column", "index")
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    column: str
    index: ColumnIndex
    def __init__(self, column: _Optional[str] = ..., index: _Optional[_Union[ColumnIndex, _Mapping]] = ...) -> None: ...

class ArtifactPolicy(_message.Message):
    __slots__ = ("revision", "indexes", "projections", "grouped_extrema")
    REVISION_FIELD_NUMBER: _ClassVar[int]
    INDEXES_FIELD_NUMBER: _ClassVar[int]
    PROJECTIONS_FIELD_NUMBER: _ClassVar[int]
    GROUPED_EXTREMA_FIELD_NUMBER: _ClassVar[int]
    revision: int
    indexes: _containers.RepeatedCompositeFieldContainer[ColumnArtifactPolicy]
    projections: _containers.RepeatedCompositeFieldContainer[CoveringProjection]
    grouped_extrema: _containers.RepeatedCompositeFieldContainer[GroupedExtrema]
    def __init__(self, revision: _Optional[int] = ..., indexes: _Optional[_Iterable[_Union[ColumnArtifactPolicy, _Mapping]]] = ..., projections: _Optional[_Iterable[_Union[CoveringProjection, _Mapping]]] = ..., grouped_extrema: _Optional[_Iterable[_Union[GroupedExtrema, _Mapping]]] = ...) -> None: ...

class RemoteRetentionPolicy(_message.Message):
    __slots__ = ("retain_forever",)
    RETAIN_FOREVER_FIELD_NUMBER: _ClassVar[int]
    retain_forever: bool
    def __init__(self, retain_forever: _Optional[bool] = ...) -> None: ...

class OperatingPolicy(_message.Message):
    __slots__ = ("l0_mode", "local_cache", "remote_retention", "max_buffer_bytes", "max_flush_age_ms", "max_query_time_ms", "rollback_window_ms")
    L0_MODE_FIELD_NUMBER: _ClassVar[int]
    LOCAL_CACHE_FIELD_NUMBER: _ClassVar[int]
    REMOTE_RETENTION_FIELD_NUMBER: _ClassVar[int]
    MAX_BUFFER_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_FLUSH_AGE_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_QUERY_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_WINDOW_MS_FIELD_NUMBER: _ClassVar[int]
    l0_mode: L0Mode
    local_cache: StoragePolicy
    remote_retention: RemoteRetentionPolicy
    max_buffer_bytes: int
    max_flush_age_ms: int
    max_query_time_ms: int
    rollback_window_ms: int
    def __init__(self, l0_mode: _Optional[_Union[L0Mode, str]] = ..., local_cache: _Optional[_Union[StoragePolicy, _Mapping]] = ..., remote_retention: _Optional[_Union[RemoteRetentionPolicy, _Mapping]] = ..., max_buffer_bytes: _Optional[int] = ..., max_flush_age_ms: _Optional[int] = ..., max_query_time_ms: _Optional[int] = ..., rollback_window_ms: _Optional[int] = ...) -> None: ...

class TableSpec(_message.Message):
    __slots__ = ("version", "logical_schema", "source_layout", "artifact_policy", "operating_policy")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LAYOUT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_POLICY_FIELD_NUMBER: _ClassVar[int]
    OPERATING_POLICY_FIELD_NUMBER: _ClassVar[int]
    version: int
    logical_schema: Schema
    source_layout: SourceLayout
    artifact_policy: ArtifactPolicy
    operating_policy: OperatingPolicy
    def __init__(self, version: _Optional[int] = ..., logical_schema: _Optional[_Union[Schema, _Mapping]] = ..., source_layout: _Optional[_Union[SourceLayout, _Mapping]] = ..., artifact_policy: _Optional[_Union[ArtifactPolicy, _Mapping]] = ..., operating_policy: _Optional[_Union[OperatingPolicy, _Mapping]] = ...) -> None: ...

class ObjectRef(_message.Message):
    __slots__ = ("object_id", "provider_version", "etag", "byte_size", "sha256")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_VERSION_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    provider_version: str
    etag: str
    byte_size: int
    sha256: bytes
    def __init__(self, object_id: _Optional[str] = ..., provider_version: _Optional[str] = ..., etag: _Optional[str] = ..., byte_size: _Optional[int] = ..., sha256: _Optional[bytes] = ...) -> None: ...

class CatalogSegment(_message.Message):
    __slots__ = ("segment_id", "source", "level", "min_seq", "max_seq", "row_count", "created_at_ms", "min_key_value", "max_key_value", "partition_json", "table_spec_version", "retired_at_ms", "delete_after_ms", "migration_source_id", "migration_source_rows", "migration_backfill", "index_bundle", "projections", "source_segment_uuid")
    SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    MIN_SEQ_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    MIN_KEY_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAX_KEY_VALUE_FIELD_NUMBER: _ClassVar[int]
    PARTITION_JSON_FIELD_NUMBER: _ClassVar[int]
    TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    RETIRED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    DELETE_AFTER_MS_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_SOURCE_ROWS_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_BACKFILL_FIELD_NUMBER: _ClassVar[int]
    INDEX_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    PROJECTIONS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SEGMENT_UUID_FIELD_NUMBER: _ClassVar[int]
    segment_id: str
    source: ObjectRef
    level: int
    min_seq: int
    max_seq: int
    row_count: int
    created_at_ms: int
    min_key_value: str
    max_key_value: str
    partition_json: str
    table_spec_version: int
    retired_at_ms: int
    delete_after_ms: int
    migration_source_id: str
    migration_source_rows: int
    migration_backfill: bool
    index_bundle: ObjectRef
    projections: _containers.RepeatedCompositeFieldContainer[ProjectionArtifact]
    source_segment_uuid: str
    def __init__(self, segment_id: _Optional[str] = ..., source: _Optional[_Union[ObjectRef, _Mapping]] = ..., level: _Optional[int] = ..., min_seq: _Optional[int] = ..., max_seq: _Optional[int] = ..., row_count: _Optional[int] = ..., created_at_ms: _Optional[int] = ..., min_key_value: _Optional[str] = ..., max_key_value: _Optional[str] = ..., partition_json: _Optional[str] = ..., table_spec_version: _Optional[int] = ..., retired_at_ms: _Optional[int] = ..., delete_after_ms: _Optional[int] = ..., migration_source_id: _Optional[str] = ..., migration_source_rows: _Optional[int] = ..., migration_backfill: _Optional[bool] = ..., index_bundle: _Optional[_Union[ObjectRef, _Mapping]] = ..., projections: _Optional[_Iterable[_Union[ProjectionArtifact, _Mapping]]] = ..., source_segment_uuid: _Optional[str] = ...) -> None: ...

class ProjectionArtifact(_message.Message):
    __slots__ = ("name", "object")
    NAME_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    name: str
    object: ObjectRef
    def __init__(self, name: _Optional[str] = ..., object: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class TableVersionSegments(_message.Message):
    __slots__ = ("table_spec_version", "live_segments", "retired_segments")
    TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    LIVE_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    RETIRED_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    table_spec_version: int
    live_segments: _containers.RepeatedCompositeFieldContainer[CatalogSegment]
    retired_segments: _containers.RepeatedCompositeFieldContainer[CatalogSegment]
    def __init__(self, table_spec_version: _Optional[int] = ..., live_segments: _Optional[_Iterable[_Union[CatalogSegment, _Mapping]]] = ..., retired_segments: _Optional[_Iterable[_Union[CatalogSegment, _Mapping]]] = ...) -> None: ...

class TableMigrationStatus(_message.Message):
    __slots__ = ("migration_id", "from_version", "to_version", "phase", "fence_seq", "source_generation", "rows_total", "rows_completed", "observation_deadline_ms")
    MIGRATION_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_VERSION_FIELD_NUMBER: _ClassVar[int]
    TO_VERSION_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    FENCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    SOURCE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ROWS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    ROWS_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    migration_id: str
    from_version: int
    to_version: int
    phase: MigrationPhase
    fence_seq: int
    source_generation: int
    rows_total: int
    rows_completed: int
    observation_deadline_ms: int
    def __init__(self, migration_id: _Optional[str] = ..., from_version: _Optional[int] = ..., to_version: _Optional[int] = ..., phase: _Optional[_Union[MigrationPhase, str]] = ..., fence_seq: _Optional[int] = ..., source_generation: _Optional[int] = ..., rows_total: _Optional[int] = ..., rows_completed: _Optional[int] = ..., observation_deadline_ms: _Optional[int] = ...) -> None: ...

class ForwardCursor(_message.Message):
    __slots__ = ("target", "cursor")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    target: str
    cursor: int
    def __init__(self, target: _Optional[str] = ..., cursor: _Optional[int] = ...) -> None: ...

class NamespaceCatalog(_message.Message):
    __slots__ = ("format_version", "namespace", "catalog_generation", "active_table_spec_version", "desired_table_spec_version", "retained_table_specs", "persisted_high_water", "version_segments", "migration", "max_query_time_ms", "forward_cursors", "direct_query_segments", "direct_query_high_water", "tombstoned", "rollback_window_ms")
    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    CATALOG_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    DESIRED_TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    RETAINED_TABLE_SPECS_FIELD_NUMBER: _ClassVar[int]
    PERSISTED_HIGH_WATER_FIELD_NUMBER: _ClassVar[int]
    VERSION_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_FIELD_NUMBER: _ClassVar[int]
    MAX_QUERY_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    FORWARD_CURSORS_FIELD_NUMBER: _ClassVar[int]
    DIRECT_QUERY_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    DIRECT_QUERY_HIGH_WATER_FIELD_NUMBER: _ClassVar[int]
    TOMBSTONED_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_WINDOW_MS_FIELD_NUMBER: _ClassVar[int]
    format_version: int
    namespace: str
    catalog_generation: int
    active_table_spec_version: int
    desired_table_spec_version: int
    retained_table_specs: _containers.RepeatedCompositeFieldContainer[TableSpec]
    persisted_high_water: int
    version_segments: _containers.RepeatedCompositeFieldContainer[TableVersionSegments]
    migration: TableMigrationStatus
    max_query_time_ms: int
    forward_cursors: _containers.RepeatedCompositeFieldContainer[ForwardCursor]
    direct_query_segments: _containers.RepeatedCompositeFieldContainer[CatalogSegment]
    direct_query_high_water: int
    tombstoned: bool
    rollback_window_ms: int
    def __init__(self, format_version: _Optional[int] = ..., namespace: _Optional[str] = ..., catalog_generation: _Optional[int] = ..., active_table_spec_version: _Optional[int] = ..., desired_table_spec_version: _Optional[int] = ..., retained_table_specs: _Optional[_Iterable[_Union[TableSpec, _Mapping]]] = ..., persisted_high_water: _Optional[int] = ..., version_segments: _Optional[_Iterable[_Union[TableVersionSegments, _Mapping]]] = ..., migration: _Optional[_Union[TableMigrationStatus, _Mapping]] = ..., max_query_time_ms: _Optional[int] = ..., forward_cursors: _Optional[_Iterable[_Union[ForwardCursor, _Mapping]]] = ..., direct_query_segments: _Optional[_Iterable[_Union[CatalogSegment, _Mapping]]] = ..., direct_query_high_water: _Optional[int] = ..., tombstoned: _Optional[bool] = ..., rollback_window_ms: _Optional[int] = ...) -> None: ...

class CatalogHead(_message.Message):
    __slots__ = ("format_version", "namespace", "writer_epoch", "catalog_generation", "active_table_spec_version", "catalog", "tombstoned")
    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CATALOG_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    CATALOG_FIELD_NUMBER: _ClassVar[int]
    TOMBSTONED_FIELD_NUMBER: _ClassVar[int]
    format_version: int
    namespace: str
    writer_epoch: int
    catalog_generation: int
    active_table_spec_version: int
    catalog: ObjectRef
    tombstoned: bool
    def __init__(self, format_version: _Optional[int] = ..., namespace: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., catalog_generation: _Optional[int] = ..., active_table_spec_version: _Optional[int] = ..., catalog: _Optional[_Union[ObjectRef, _Mapping]] = ..., tombstoned: _Optional[bool] = ...) -> None: ...

class RegisterTableRequest(_message.Message):
    __slots__ = ("namespace", "schema", "storage_policy", "table_spec")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    STORAGE_POLICY_FIELD_NUMBER: _ClassVar[int]
    TABLE_SPEC_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    schema: Schema
    storage_policy: StoragePolicy
    table_spec: TableSpec
    def __init__(self, namespace: _Optional[str] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ..., storage_policy: _Optional[_Union[StoragePolicy, _Mapping]] = ..., table_spec: _Optional[_Union[TableSpec, _Mapping]] = ...) -> None: ...

class RegisterTableResponse(_message.Message):
    __slots__ = ("effective_schema", "effective_policy", "active_table_spec_version", "desired_table_spec_version", "transition_phase")
    EFFECTIVE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_POLICY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    DESIRED_TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_PHASE_FIELD_NUMBER: _ClassVar[int]
    effective_schema: Schema
    effective_policy: StoragePolicy
    active_table_spec_version: int
    desired_table_spec_version: int
    transition_phase: MigrationPhase
    def __init__(self, effective_schema: _Optional[_Union[Schema, _Mapping]] = ..., effective_policy: _Optional[_Union[StoragePolicy, _Mapping]] = ..., active_table_spec_version: _Optional[int] = ..., desired_table_spec_version: _Optional[int] = ..., transition_phase: _Optional[_Union[MigrationPhase, str]] = ...) -> None: ...

class WriteRowsRequest(_message.Message):
    __slots__ = ("namespace", "arrow_ipc")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    ARROW_IPC_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    arrow_ipc: bytes
    def __init__(self, namespace: _Optional[str] = ..., arrow_ipc: _Optional[bytes] = ...) -> None: ...

class WriteRowsResponse(_message.Message):
    __slots__ = ("rows_written",)
    ROWS_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    rows_written: int
    def __init__(self, rows_written: _Optional[int] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("sql",)
    SQL_FIELD_NUMBER: _ClassVar[int]
    sql: str
    def __init__(self, sql: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("arrow_ipc", "row_count")
    ARROW_IPC_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    arrow_ipc: bytes
    row_count: int
    def __init__(self, arrow_ipc: _Optional[bytes] = ..., row_count: _Optional[int] = ...) -> None: ...

class DropTableRequest(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class DropTableResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class NamespaceInfo(_message.Message):
    __slots__ = ("namespace", "schema", "row_count", "byte_size", "min_seq", "max_seq", "segment_count", "storage_policy")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIN_SEQ_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_COUNT_FIELD_NUMBER: _ClassVar[int]
    STORAGE_POLICY_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    schema: Schema
    row_count: int
    byte_size: int
    min_seq: int
    max_seq: int
    segment_count: int
    storage_policy: StoragePolicy
    def __init__(self, namespace: _Optional[str] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ..., row_count: _Optional[int] = ..., byte_size: _Optional[int] = ..., min_seq: _Optional[int] = ..., max_seq: _Optional[int] = ..., segment_count: _Optional[int] = ..., storage_policy: _Optional[_Union[StoragePolicy, _Mapping]] = ...) -> None: ...

class ListNamespacesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListNamespacesResponse(_message.Message):
    __slots__ = ("namespaces",)
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    namespaces: _containers.RepeatedCompositeFieldContainer[NamespaceInfo]
    def __init__(self, namespaces: _Optional[_Iterable[_Union[NamespaceInfo, _Mapping]]] = ...) -> None: ...

class GetTableSchemaRequest(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class GetTableSchemaResponse(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: Schema
    def __init__(self, schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...

class GetTableStatusRequest(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class GetTableStatusResponse(_message.Message):
    __slots__ = ("active_table_spec", "desired_table_spec", "migration", "catalog_generation", "migration_blocked", "migration_error")
    ACTIVE_TABLE_SPEC_FIELD_NUMBER: _ClassVar[int]
    DESIRED_TABLE_SPEC_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_FIELD_NUMBER: _ClassVar[int]
    CATALOG_GENERATION_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_BLOCKED_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_ERROR_FIELD_NUMBER: _ClassVar[int]
    active_table_spec: TableSpec
    desired_table_spec: TableSpec
    migration: TableMigrationStatus
    catalog_generation: int
    migration_blocked: bool
    migration_error: str
    def __init__(self, active_table_spec: _Optional[_Union[TableSpec, _Mapping]] = ..., desired_table_spec: _Optional[_Union[TableSpec, _Mapping]] = ..., migration: _Optional[_Union[TableMigrationStatus, _Mapping]] = ..., catalog_generation: _Optional[int] = ..., migration_blocked: _Optional[bool] = ..., migration_error: _Optional[str] = ...) -> None: ...

class AbortTableMigrationRequest(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class AbortTableMigrationResponse(_message.Message):
    __slots__ = ("catalog_generation", "active_table_spec_version")
    CATALOG_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_TABLE_SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    catalog_generation: int
    active_table_spec_version: int
    def __init__(self, catalog_generation: _Optional[int] = ..., active_table_spec_version: _Optional[int] = ...) -> None: ...
