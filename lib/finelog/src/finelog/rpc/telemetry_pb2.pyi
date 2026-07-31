from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TelemetryResourceV1(_message.Message):
    __slots__ = ("service_name", "service_instance_id", "role", "root_run_uid", "service_version", "run_id_alias", "iris_job_id", "iris_task_id", "task_index", "attempt_id", "attempt_uid", "worker_id", "node_id", "pod_uid", "container_id", "rank", "process_index", "actor_id", "engine_id", "repository", "git_revision", "image_digest", "model_id", "model_revision", "policy_step", "owner", "experiment_issue", "cluster", "entity_authority", "entity_type", "entity_uid")
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVICE_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    ROOT_RUN_UID_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VERSION_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_ALIAS_FIELD_NUMBER: _ClassVar[int]
    IRIS_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    IRIS_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    POD_UID_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    PROCESS_INDEX_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    ENGINE_ID_FIELD_NUMBER: _ClassVar[int]
    REPOSITORY_FIELD_NUMBER: _ClassVar[int]
    GIT_REVISION_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_REVISION_FIELD_NUMBER: _ClassVar[int]
    POLICY_STEP_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ISSUE_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_FIELD_NUMBER: _ClassVar[int]
    ENTITY_AUTHORITY_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_UID_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    service_instance_id: str
    role: str
    root_run_uid: str
    service_version: str
    run_id_alias: str
    iris_job_id: str
    iris_task_id: str
    task_index: int
    attempt_id: int
    attempt_uid: str
    worker_id: str
    node_id: str
    pod_uid: str
    container_id: str
    rank: int
    process_index: int
    actor_id: str
    engine_id: str
    repository: str
    git_revision: str
    image_digest: str
    model_id: str
    model_revision: str
    policy_step: int
    owner: str
    experiment_issue: int
    cluster: str
    entity_authority: str
    entity_type: str
    entity_uid: str
    def __init__(self, service_name: _Optional[str] = ..., service_instance_id: _Optional[str] = ..., role: _Optional[str] = ..., root_run_uid: _Optional[str] = ..., service_version: _Optional[str] = ..., run_id_alias: _Optional[str] = ..., iris_job_id: _Optional[str] = ..., iris_task_id: _Optional[str] = ..., task_index: _Optional[int] = ..., attempt_id: _Optional[int] = ..., attempt_uid: _Optional[str] = ..., worker_id: _Optional[str] = ..., node_id: _Optional[str] = ..., pod_uid: _Optional[str] = ..., container_id: _Optional[str] = ..., rank: _Optional[int] = ..., process_index: _Optional[int] = ..., actor_id: _Optional[str] = ..., engine_id: _Optional[str] = ..., repository: _Optional[str] = ..., git_revision: _Optional[str] = ..., image_digest: _Optional[str] = ..., model_id: _Optional[str] = ..., model_revision: _Optional[str] = ..., policy_step: _Optional[int] = ..., owner: _Optional[str] = ..., experiment_issue: _Optional[int] = ..., cluster: _Optional[str] = ..., entity_authority: _Optional[str] = ..., entity_type: _Optional[str] = ..., entity_uid: _Optional[str] = ...) -> None: ...

class TelemetryMetricV1(_message.Message):
    __slots__ = ("scope", "scope_version", "name", "description", "unit", "instrument_kind", "temporality", "start_ts_unix_nano", "reset_id", "series_id", "value", "count", "sum", "explicit_bounds", "bucket_counts", "attributes", "delivery_class", "device_uid", "device_type")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_VERSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    INSTRUMENT_KIND_FIELD_NUMBER: _ClassVar[int]
    TEMPORALITY_FIELD_NUMBER: _ClassVar[int]
    START_TS_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    RESET_ID_FIELD_NUMBER: _ClassVar[int]
    SERIES_ID_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    EXPLICIT_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    BUCKET_COUNTS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_CLASS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    scope: str
    scope_version: str
    name: str
    description: str
    unit: str
    instrument_kind: str
    temporality: str
    start_ts_unix_nano: int
    reset_id: str
    series_id: str
    value: float
    count: int
    sum: float
    explicit_bounds: _containers.RepeatedScalarFieldContainer[float]
    bucket_counts: _containers.RepeatedScalarFieldContainer[int]
    attributes: _containers.ScalarMap[str, str]
    delivery_class: str
    device_uid: str
    device_type: str
    def __init__(self, scope: _Optional[str] = ..., scope_version: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., unit: _Optional[str] = ..., instrument_kind: _Optional[str] = ..., temporality: _Optional[str] = ..., start_ts_unix_nano: _Optional[int] = ..., reset_id: _Optional[str] = ..., series_id: _Optional[str] = ..., value: _Optional[float] = ..., count: _Optional[int] = ..., sum: _Optional[float] = ..., explicit_bounds: _Optional[_Iterable[float]] = ..., bucket_counts: _Optional[_Iterable[int]] = ..., attributes: _Optional[_Mapping[str, str]] = ..., delivery_class: _Optional[str] = ..., device_uid: _Optional[str] = ..., device_type: _Optional[str] = ...) -> None: ...

class TelemetryEventV1(_message.Message):
    __slots__ = ("event_name", "severity_number", "severity_text", "outcome", "phase", "error_type", "body", "attributes", "trace_id", "span_id", "evidence_uri", "result_uri", "delivery_class", "probe_status")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EVENT_NAME_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_TEXT_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_URI_FIELD_NUMBER: _ClassVar[int]
    RESULT_URI_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_CLASS_FIELD_NUMBER: _ClassVar[int]
    PROBE_STATUS_FIELD_NUMBER: _ClassVar[int]
    event_name: str
    severity_number: int
    severity_text: str
    outcome: str
    phase: str
    error_type: str
    body: str
    attributes: _containers.ScalarMap[str, str]
    trace_id: str
    span_id: str
    evidence_uri: str
    result_uri: str
    delivery_class: str
    probe_status: str
    def __init__(self, event_name: _Optional[str] = ..., severity_number: _Optional[int] = ..., severity_text: _Optional[str] = ..., outcome: _Optional[str] = ..., phase: _Optional[str] = ..., error_type: _Optional[str] = ..., body: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ..., trace_id: _Optional[str] = ..., span_id: _Optional[str] = ..., evidence_uri: _Optional[str] = ..., result_uri: _Optional[str] = ..., delivery_class: _Optional[str] = ..., probe_status: _Optional[str] = ...) -> None: ...

class TelemetryLogV1(_message.Message):
    __slots__ = ("source", "body", "severity_number", "severity_text", "event_name", "attributes", "trace_id", "span_id")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_TEXT_FIELD_NUMBER: _ClassVar[int]
    EVENT_NAME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    source: str
    body: str
    severity_number: int
    severity_text: str
    event_name: str
    attributes: _containers.ScalarMap[str, str]
    trace_id: str
    span_id: str
    def __init__(self, source: _Optional[str] = ..., body: _Optional[str] = ..., severity_number: _Optional[int] = ..., severity_text: _Optional[str] = ..., event_name: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ..., trace_id: _Optional[str] = ..., span_id: _Optional[str] = ...) -> None: ...

class TelemetryArtifactV1(_message.Message):
    __slots__ = ("capture_type", "trigger", "start_ts_unix_nano", "end_ts_unix_nano", "outcome", "size_bytes", "sha256", "uri", "summary", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CAPTURE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    START_TS_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    END_TS_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    capture_type: str
    trigger: str
    start_ts_unix_nano: int
    end_ts_unix_nano: int
    outcome: str
    size_bytes: int
    sha256: str
    uri: str
    summary: str
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, capture_type: _Optional[str] = ..., trigger: _Optional[str] = ..., start_ts_unix_nano: _Optional[int] = ..., end_ts_unix_nano: _Optional[int] = ..., outcome: _Optional[str] = ..., size_bytes: _Optional[int] = ..., sha256: _Optional[str] = ..., uri: _Optional[str] = ..., summary: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TelemetryRecordV1(_message.Message):
    __slots__ = ("record_index", "signal", "event_ts_unix_nano", "observed_ts_unix_nano", "resource", "metric", "event", "log", "artifact")
    RECORD_INDEX_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    EVENT_TS_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_TS_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    record_index: int
    signal: str
    event_ts_unix_nano: int
    observed_ts_unix_nano: int
    resource: TelemetryResourceV1
    metric: TelemetryMetricV1
    event: TelemetryEventV1
    log: TelemetryLogV1
    artifact: TelemetryArtifactV1
    def __init__(self, record_index: _Optional[int] = ..., signal: _Optional[str] = ..., event_ts_unix_nano: _Optional[int] = ..., observed_ts_unix_nano: _Optional[int] = ..., resource: _Optional[_Union[TelemetryResourceV1, _Mapping]] = ..., metric: _Optional[_Union[TelemetryMetricV1, _Mapping]] = ..., event: _Optional[_Union[TelemetryEventV1, _Mapping]] = ..., log: _Optional[_Union[TelemetryLogV1, _Mapping]] = ..., artifact: _Optional[_Union[TelemetryArtifactV1, _Mapping]] = ...) -> None: ...

class TelemetryBatchV1(_message.Message):
    __slots__ = ("schema_version", "catalog_version", "batch_id", "records")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    CATALOG_VERSION_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    catalog_version: str
    batch_id: str
    records: _containers.RepeatedCompositeFieldContainer[TelemetryRecordV1]
    def __init__(self, schema_version: _Optional[int] = ..., catalog_version: _Optional[str] = ..., batch_id: _Optional[str] = ..., records: _Optional[_Iterable[_Union[TelemetryRecordV1, _Mapping]]] = ...) -> None: ...

class TelemetryValidationErrorV1(_message.Message):
    __slots__ = ("record_index", "field", "reason")
    RECORD_INDEX_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    record_index: int
    field: str
    reason: str
    def __init__(self, record_index: _Optional[int] = ..., field: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class TelemetryCommitV1(_message.Message):
    __slots__ = ("namespace", "first_seq", "last_seq")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    FIRST_SEQ_FIELD_NUMBER: _ClassVar[int]
    LAST_SEQ_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    first_seq: int
    last_seq: int
    def __init__(self, namespace: _Optional[str] = ..., first_seq: _Optional[int] = ..., last_seq: _Optional[int] = ...) -> None: ...

class TelemetryWriteAckV1(_message.Message):
    __slots__ = ("schema_version", "batch_id", "status", "durability", "accepted_records", "rejected_records", "commits")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DURABILITY_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_RECORDS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_RECORDS_FIELD_NUMBER: _ClassVar[int]
    COMMITS_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    batch_id: str
    status: str
    durability: str
    accepted_records: int
    rejected_records: _containers.RepeatedCompositeFieldContainer[TelemetryValidationErrorV1]
    commits: _containers.RepeatedCompositeFieldContainer[TelemetryCommitV1]
    def __init__(self, schema_version: _Optional[int] = ..., batch_id: _Optional[str] = ..., status: _Optional[str] = ..., durability: _Optional[str] = ..., accepted_records: _Optional[int] = ..., rejected_records: _Optional[_Iterable[_Union[TelemetryValidationErrorV1, _Mapping]]] = ..., commits: _Optional[_Iterable[_Union[TelemetryCommitV1, _Mapping]]] = ...) -> None: ...
