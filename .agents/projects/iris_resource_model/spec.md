# Iris resource model contract

This spec pins the final surface described in [design.md](design.md). It does
not define transitional adapters: the pull request updates all first-party
callers and accepts only the exact merge-base database schema.

## Decisions

- Public nouns are Job, Task, Attempt, Node, Slice, Endpoint, and Backend.
- `Worker` is private to RPC execution and is absent from operator APIs.
- Public history is named `activity`; `events` is removed from CLI and RPC
  surfaces.
- Public mutations are Job cancel, Task retry, and Attempt terminate.
- Slice deletion remains a provider operation and is not part of this contract.
- The controller reports accepted mutations through durable action receipts.
- The resource migration accepts one exact source fingerprint and has no repair,
  dual-write, or old-image mode.

## Python resource API

All record dataclasses below are frozen and slotted. Identifiers validate their
non-empty fields in `__post_init__`. `cluster_id` names the authority controller
for Jobs, Tasks, Attempts, and Endpoints and the execution controller for Nodes
and Slices.

### `lib/iris/src/iris/cluster/resources/identity.py`

```python
class ResourceKind(StrEnum):
    JOB = "job"
    TASK = "task"
    ATTEMPT = "attempt"
    ENDPOINT = "endpoint"
    NODE = "node"
    SLICE = "slice"


@dataclass(frozen=True, slots=True)
class ResourceKey:
    cluster_id: str
    kind: ResourceKind
    resource_id: str


@dataclass(frozen=True, slots=True)
class JobIdentity:
    key: ResourceKey
    job_uid: str


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    key: ResourceKey
    task_uid: str


@dataclass(frozen=True, slots=True)
class AttemptLocator:
    task: ResourceKey
    attempt_number: int | None


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    task: ResourceKey
    attempt_number: int
    attempt_uid: str


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    key: ResourceKey
    backend_id: str
    node_uid: str


@dataclass(frozen=True, slots=True)
class NodeLocator:
    key: ResourceKey
    backend_id: str
    node_uid: str | None = None


@dataclass(frozen=True, slots=True)
class SliceIdentity:
    key: ResourceKey
    backend_id: str
    slice_uid: str


@dataclass(frozen=True, slots=True)
class SliceLocator:
    key: ResourceKey
    backend_id: str
    slice_uid: str | None = None
```

`AttemptLocator.attempt_number=None` means the Task's exact current Attempt at
resolution time. A resolver returns an `AttemptIdentity` before any live call.
All mutations require exact identities; logical keys alone are read targets.
An Attempt activity/action key uses kind `ATTEMPT` and resource ID
`<task_id>:<attempt_number>`; its UID remains the exact authority. Endpoint keys
use the random `endpoint_id`, Node keys use the logical provider/daemon Node ID,
and Slice keys use the logical provider allocation ID.
Node and Slice locators may omit the UID to select the unique current
incarnation. The service captures that UID before enrichment. A replacement
between lookup and provider response raises `ResourceReplaced`; retired
incarnations require an explicit UID.

### `lib/iris/src/iris/cluster/resources/source.py`

```python
MAX_SOURCE_ERROR_CODE: Final[int] = 64
MAX_SOURCE_ERROR_MESSAGE: Final[int] = 512
MAX_ACTIVITY_MESSAGE: Final[int] = 2_048
MAX_ACTIVITY_ATTRIBUTES: Final[int] = 32
MAX_ACTIVITY_ATTRIBUTE_KEY: Final[int] = 64
MAX_ACTIVITY_ATTRIBUTE_VALUE: Final[int] = 512
MAX_ROOT_CAUSE_HIGHLIGHTS: Final[int] = 20
MAX_ROOT_CAUSE_HIGHLIGHT: Final[int] = 512
MAX_PROVIDER_SNAPSHOT_ITEMS: Final[int] = 50_000


class SourceState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class Freshness(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ResourceSourceStatus:
    source_id: str
    backend_id: str
    state: SourceState
    freshness: Freshness
    observed_at: Timestamp | None
    error_code: str
    error_message: str


@dataclass(frozen=True, slots=True)
class Page[T]:
    items: tuple[T, ...]
    next_page_token: str | None
    source_statuses: tuple[ResourceSourceStatus, ...]
```

Source IDs use `controller:<cluster_id>`, `backend:<backend_id>`,
`finelog:<cluster_id>`, or `federation:<peer_id>`. `backend_id` is empty only
for controller, finelog, and federation sources. A backend observation is
current until `max(2 * configured_poll_interval, 30 seconds)`, stale after that,
and unknown before its first successful observation. Controller and finelog
reads are current on successful request completion. Error codes are at most 64
characters and error messages at most 512 characters.
`AVAILABLE` has an observation time and empty error fields. `UNAVAILABLE` has a
non-empty error code and retains the last observation time when one exists.
`UNSUPPORTED` has freshness unknown, no observation time, and error code
`unsupported`.

Lists use opaque keyset tokens. Tokens bind the normalized query, caller, sort,
and last key. A mismatched or malformed token raises `InvalidPageToken`. Lists
are weakly consistent across pages and never issue one provider call per row.
Jobs sort by `(submitted_at DESC, job_uid)`, Tasks by
`(submitted_at DESC, task_uid)`, Nodes by `(backend_id, node_id, node_uid)`,
Slices by `(backend_id, slice_id, slice_uid)`, Endpoints by
`(name, endpoint_id)`, and Activity by `(occurred_at DESC, source, entry_id)`.
A page token resumes after the complete tuple. The Activity tuple is a stable
display order, not a causal order across SQLite and finelog.

### Noun records

`job.py`, `task.py`, `attempt.py`, `node.py`, `slice.py`, and `endpoint.py` own
their query and result records:

```python
@dataclass(frozen=True, slots=True)
class JobSpec:
    version: int
    name: str
    entrypoint: RuntimeEntrypoint
    resources: ResourceSpec
    environment: EnvironmentConfig
    bundle_id: str
    scheduling_timeout: Duration | None
    ports: tuple[str, ...]
    max_task_failures: int
    max_retries_failure: int
    max_retries_preemption: int
    constraints: tuple[Constraint, ...]
    coscheduling: CoschedulingConfig | None
    replicas: int
    timeout: Duration | None
    fail_if_exists: bool
    preemption_policy: JobPreemptionPolicy
    existing_job_policy: ExistingJobPolicy
    priority_band: PriorityBand
    task_image: str
    submit_argv: tuple[str, ...]
    client_revision_date: str
    container_profile: ContainerProfile


@dataclass(frozen=True, slots=True)
class JobQuery:
    owner_id: str | None = None
    parent: ResourceKey | None = None
    job_id_prefix: str | None = None
    states: frozenset[JobState] = frozenset()
    backend_id: str | None = None
    execution_cluster_id: str | None = None
    page_size: int = 50
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class JobSummary:
    identity: JobIdentity
    owner_id: str
    parent: JobIdentity | None
    state: JobState
    execution_cluster_id: str
    backend_id: str
    num_tasks: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    error_message: str


@dataclass(frozen=True, slots=True)
class JobDetail:
    summary: JobSummary
    spec: JobSpec


@dataclass(frozen=True, slots=True)
class TaskQuery:
    job: ResourceKey | None = None
    job_id_prefix: str | None = None
    states: frozenset[TaskState] = frozenset()
    backend_id: str | None = None
    authority_cluster_id: str | None = None
    execution_cluster_id: str | None = None
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class TaskSummary:
    identity: TaskIdentity
    job: JobIdentity
    task_index: int
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    current_attempt: AttemptIdentity | None
    current_node: NodeIdentity | None
    failure_count: int
    preemption_count: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    status_message: str
    error_message: str


@dataclass(frozen=True, slots=True)
class TaskDetail:
    summary: TaskSummary
    attempts: tuple[AttemptSummary, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]
    root_cause_highlights: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    identity: AttemptIdentity
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    node: NodeIdentity | None
    created_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    exit_code: int | None
    error_message: str
    terminal_reason: str


@dataclass(frozen=True, slots=True)
class AttemptRuntimeObject:
    provider_kind: str
    namespace: str
    name: str
    provider_uid: str
    provider_node_id: str
    provider_node_uid: str
    container_id: str
    observed_at: Timestamp


@dataclass(frozen=True, slots=True)
class AttemptDetail:
    summary: AttemptSummary
    runtime: AttemptRuntimeObject | None
    source_statuses: tuple[ResourceSourceStatus, ...]


class NodeHealth(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class NodeQuery:
    backend_id: str | None = None
    contains: str | None = None
    health: frozenset[NodeHealth] = frozenset()
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class NodeCapacity:
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    accelerator_kind: str
    accelerator_variant: str
    accelerator_count: int


class NodeAttributeKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"


@dataclass(frozen=True, slots=True)
class NodeAttribute:
    key: str
    kind: NodeAttributeKind
    string_value: str | None = None
    integer_value: int | None = None
    float_value: float | None = None


@dataclass(frozen=True, slots=True)
class NodeSummary:
    identity: NodeIdentity
    health: NodeHealth
    schedulable: bool
    capacity: NodeCapacity
    scaling_group_id: str | None
    slice: SliceIdentity | None
    running_task_count: int
    observed_at: Timestamp


@dataclass(frozen=True, slots=True)
class NodeDetail:
    summary: NodeSummary
    address: str | None
    attributes: tuple[NodeAttribute, ...]
    recent_attempts: tuple[AttemptSummary, ...]
    bootstrap_log_key: str | None
    source_statuses: tuple[ResourceSourceStatus, ...]


class SliceLifecycle(StrEnum):
    CREATING = "creating"
    READY = "ready"
    DELETING = "deleting"
    FAILED = "failed"


class MembershipState(StrEnum):
    UNKNOWN = "unknown"
    OBSERVED = "observed"


@dataclass(frozen=True, slots=True)
class SliceQuery:
    backend_id: str | None = None
    scaling_group_id: str | None = None
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class SliceSummary:
    identity: SliceIdentity
    scaling_group_id: str
    lifecycle: SliceLifecycle
    membership_state: MembershipState
    observed_member_count: int
    observed_at: Timestamp | None
    error_message: str


@dataclass(frozen=True, slots=True)
class SliceMember:
    provider_node_id: str
    node: NodeIdentity | None
    observed_at: Timestamp


@dataclass(frozen=True, slots=True)
class SliceDetail:
    summary: SliceSummary
    members: tuple[SliceMember, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]


class EndpointAccess(StrEnum):
    PRIVATE = "private"
    LINK = "link"


@dataclass(frozen=True, slots=True)
class EndpointQuery:
    name_prefix: str | None = None
    task: ResourceKey | None = None
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class EndpointSummary:
    key: ResourceKey
    endpoint_id: str
    name: str
    task: ResourceKey | None
    execution_cluster_id: str
    access: EndpointAccess
    lease_deadline: Timestamp | None


@dataclass(frozen=True, slots=True)
class EndpointDetail:
    summary: EndpointSummary
    address: str
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EndpointToken:
    token: str
    expires_at: Timestamp
    capability_url: str


@dataclass(frozen=True, slots=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ProfileResult:
    profile_data: bytes
    error_message: str
```

`NodeAttribute.__post_init__` requires exactly the value selected by `kind` and
rejects the other two, matching the protobuf `oneof` and SQL CHECK.

### Actions and activity

`lib/iris/src/iris/cluster/resources/action.py` defines:

```python
class ActionKind(StrEnum):
    CANCEL_JOB = "cancel_job"
    RETRY_TASK = "retry_task"
    TERMINATE_ATTEMPT = "terminate_attempt"


class ActionState(StrEnum):
    ACCEPTED = "accepted"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ActionResult(StrEnum):
    NONE = "none"
    SATISFIED = "satisfied"
    TARGET_ABSENT = "target_absent"
    PROVIDER_REJECTED = "provider_rejected"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    action_id: str
    kind: ActionKind
    target: ResourceKey
    expected_target_uid: str
    expected_attempt_uid: str | None
    state: ActionState
    result_code: ActionResult
    result_message: str
    created_at: Timestamp
    updated_at: Timestamp
    completed_at: Timestamp | None
```

`lib/iris/src/iris/cluster/resources/activity.py` defines:

```python
@dataclass(frozen=True, slots=True)
class ActivityQuery:
    target: ResourceKey
    attempt_uid: str | None = None
    after: Timestamp | None = None
    page_size: int = 200
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class ActivityEntry:
    entry_id: str
    occurred_at: Timestamp
    source: str
    severity: str
    kind: str
    message: str
    target: ResourceKey
    attempt_uid: str | None
    correlation_id: str | None
    attributes: Mapping[str, str]
```

Finelog entry IDs are `finelog:<namespace>:<sequence>` and receipt entry IDs are
`action:<action_id>`. Activity messages are at most 2,048 characters. Each entry
has at most 32 attributes; keys are at most 64 characters and values at most 512.
Task root-cause highlights are limited to 20 entries of 512 characters each.
Backend Node/Slice snapshots contain at most 50,000 items; a larger observation
is unavailable with `snapshot_too_large` rather than silently truncated.

Activity merges `iris.task_event` rows and current action receipt outcomes. It
does not claim a total order across finelog and SQLite. Missing finelog returns
receipt activity plus an unavailable source status; SQLite failure fails the
authorized request.

### Logs

`lib/iris/src/iris/cluster/resources/log.py` defines:

```python
@dataclass(frozen=True, slots=True)
class LogQuery:
    after: Timestamp | None = None
    cursor: int = 0
    max_lines: int = 1_000
    substring: str = ""
    minimum_level: LogLevel = LogLevel.UNKNOWN
    tail: bool = False


@dataclass(frozen=True, slots=True)
class LogPage:
    entries: tuple[LogEntry, ...]
    next_cursor: int
    source_statuses: tuple[ResourceSourceStatus, ...]
```

Job logs aggregate all Attempts of Tasks directly owned by the exact Job UID;
descendant Jobs have separate logs. Task logs aggregate all historical and
current Attempts of the exact Task UID. Attempt logs include only the exact
Attempt UID. Entries are globally ordered by finelog `(timestamp, sequence)`.
Task/Job follow continues across later Attempts of the same exact UID; Attempt
follow ends when that Attempt is terminal and drained. Replacement of the
logical Job or Task ends follow with `ResourceReplaced`.

The controller authorizes the exact resource and translates it to finelog keys
before proxying the query. Finelog never authorizes a caller from a raw source
string supplied by the client.

## Backend and resolver protocols

`lib/iris/src/iris/cluster/backends/protocol.py` contains plain data and
protocols. It imports resource models but no controller database or persistence
module.

```python
@dataclass(frozen=True, slots=True)
class SourceSnapshot[T]:
    items: tuple[T, ...]
    status: ResourceSourceStatus


@dataclass(frozen=True, slots=True)
class ExactAttemptTarget:
    identity: AttemptIdentity
    execution_cluster_id: str
    backend_id: str
    node_uid: str | None
    node_address: str | None
    runtime: AttemptRuntimeObject | None


@dataclass(frozen=True, slots=True)
class PendingTaskInput:
    identity: TaskIdentity
    job: JobIdentity
    task_index: int
    owner_id: str
    spec: JobSpec
    current_attempt: AttemptIdentity | None
    failure_count: int
    preemption_count: int
    submitted_at: Timestamp


@dataclass(frozen=True, slots=True)
class BudgetInput:
    owner_id: str
    priority_band: PriorityBand
    accelerator_kind: str
    limit: float
    usage: float


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    backend_id: str
    pending_tasks: tuple[PendingTaskInput, ...]
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    budgets: tuple[BudgetInput, ...]
    now: Timestamp
    trace: bool


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    task: TaskIdentity
    backend_id: str
    scaling_group_id: str | None
    node: NodeIdentity | None
    slice: SliceIdentity | None


@dataclass(frozen=True, slots=True)
class PreemptionDecision:
    attempt: AttemptIdentity
    reason: str


@dataclass(frozen=True, slots=True)
class UnschedulableDecision:
    task: TaskIdentity
    reason: str


@dataclass(frozen=True, slots=True)
class CapacityDemandInput:
    accelerator_kind: str
    accelerator_variant: str
    accelerator_count: int
    priority_band: PriorityBand
    task_count: int


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    placements: tuple[PlacementDecision, ...]
    preemptions: tuple[PreemptionDecision, ...]
    unschedulable: tuple[UnschedulableDecision, ...]
    residual_demand: tuple[CapacityDemandInput, ...]


@dataclass(frozen=True, slots=True)
class DesiredAttemptInput:
    identity: AttemptIdentity
    job: JobIdentity
    spec: JobSpec
    backend_id: str
    node: NodeIdentity | None
    desired_state: TaskState


@dataclass(frozen=True, slots=True)
class ActionTargetInput:
    action_id: str
    kind: ActionKind
    attempt: ExactAttemptTarget


@dataclass(frozen=True, slots=True)
class ReconcileRequest:
    backend_id: str
    desired_attempts: tuple[DesiredAttemptInput, ...]
    action_targets: tuple[ActionTargetInput, ...]
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    now: Timestamp


@dataclass(frozen=True, slots=True)
class AttemptObservation:
    identity: AttemptIdentity
    state: TaskState
    node: NodeIdentity | None
    runtime: AttemptRuntimeObject | None
    started_at: Timestamp | None
    finished_at: Timestamp | None
    exit_code: int | None
    status_message: str
    error_message: str
    terminal_reason: str


class RuntimeTargetState(StrEnum):
    ACTIVE = "active"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ActionObservation:
    action_id: str
    attempt: AttemptIdentity
    state: RuntimeTargetState
    reason: str


@dataclass(frozen=True, slots=True)
class NodeRetirementDecision:
    node: NodeIdentity
    reason: str


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    attempts: tuple[AttemptObservation, ...]
    actions: tuple[ActionObservation, ...]
    retired_nodes: tuple[NodeRetirementDecision, ...]
    activity: tuple[ActivityEntry, ...]


@dataclass(frozen=True, slots=True)
class AutoscaleRequest:
    backend_id: str
    demand: tuple[CapacityDemandInput, ...]
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    now: Timestamp


@dataclass(frozen=True, slots=True)
class AutoscaleResult:
    nodes: SourceSnapshot[NodeSummary]
    slices: SourceSnapshot[SliceSummary]
    retired_nodes: tuple[NodeRetirementDecision, ...]
    activity: tuple[ActivityEntry, ...]


class NodeReader(Protocol):
    def snapshot_nodes(self) -> SourceSnapshot[NodeSummary]:
        """Return the last complete bounded snapshot without provider I/O."""

    def describe_node(
        self,
        identity: NodeIdentity,
        *,
        deadline: Deadline,
    ) -> NodeDetail:
        """Return cached detail with at most one deadline-bounded live enrichment."""


class SliceReader(Protocol):
    def snapshot_slices(self) -> SourceSnapshot[SliceSummary]:
        """Return the last complete bounded projection without provider I/O."""

    def describe_slice(
        self,
        identity: SliceIdentity,
        *,
        deadline: Deadline,
    ) -> SliceDetail:
        """Return cached membership with at most one deadline-bounded refresh."""


class AttemptRuntime(Protocol):
    def describe_attempt(
        self,
        target: ExactAttemptTarget,
        *,
        deadline: Deadline,
    ) -> AttemptDetail:
        """Describe only the exact runtime identity carried by target."""

    def exec_attempt(
        self,
        target: ExactAttemptTarget,
        command: Sequence[str],
        *,
        deadline: Deadline,
    ) -> ExecResult: ...

    def profile_attempt(
        self,
        target: ExactAttemptTarget,
        profile: ProfileType,
        *,
        duration: Duration,
        deadline: Deadline,
    ) -> ProfileResult: ...


class TaskBackend(Protocol):
    backend_id: str
    name: str
    capabilities: frozenset[BackendCapability]
    attempts: AttemptRuntime

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Make a pure placement decision from controller-owned input and backend snapshots."""

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        """Perform bounded effects and return exact observations for controller commit."""

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """Perform one bounded capacity cycle and return its projection."""

    def close(self) -> None:
        """Release backend-owned resources without writing controller state."""


@dataclass(frozen=True, slots=True)
class BackendBinding:
    tasks: TaskBackend
    nodes: NodeReader
    slices: SliceReader | None
```

The controller reads scheduling, action, worker-registration, and capacity rows,
then constructs these complete immutable requests. Backends receive no database,
transaction, persistence protocol, SQLAlchemy table, or callable that can reach
controller state. They return plain decisions and observations; only the
controller commits those results. In particular, node retirement is a returned
decision rather than a backend callback into persistence.

`BackendBinding` keeps resource reads separate from task effects. The K8s Node
and Slice readers expose its bounded poll caches. The RPC readers are
controller-owned adapters over `persistence/node.py` and `persistence/slice.py`;
they are composed beside the RPC Task backend and are never passed into it. List
requests read those snapshots. Only a single-item describe may perform
deadline-bounded live provider I/O.

`controller/resources/attempts.py` owns the only runtime resolver:

```python
def resolve_attempt_target(
    snapshot: ReadSnapshot,
    locator: AttemptLocator,
    backends: BackendResolver,
) -> ExactAttemptTarget:
    """Resolve one persisted Attempt UID and exact backend coordinates in one snapshot."""
```

`BackendResolver.require(backend_id)` performs an exact dictionary lookup and
raises `BackendIdentityUnknown`; it has no representative, default, kind, or
capability fallback. Every composition root requires nonblank
`ControllerConfig.cluster_id` and asserts
`set(config.backend_configs) == set(live_backends)`. Local mode supports its one
constructed RPC backend only; explicit or multi-backend local topology fails at
configuration time.

Logs are not a backend protocol. Controller-authorized log queries resolve the
resource to finelog keys and call finelog directly, so K8s and RPC cannot diverge
on log semantics.

## Python client surface

`lib/iris/src/iris/cluster/client/protocol.py`, `remote_client.py`, and
`lib/iris/src/iris/client/client.py` expose the same typed methods:

```python
def submit_job(self, spec: JobSpec, *, bundle: bytes | None = None) -> JobIdentity: ...
def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]: ...
def describe_job(self, key: ResourceKey) -> JobDetail: ...
def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]: ...
def describe_task(self, key: ResourceKey) -> TaskDetail: ...
def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail: ...
def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]: ...
def describe_node(self, locator: NodeLocator) -> NodeDetail: ...
def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]: ...
def describe_slice(self, locator: SliceLocator) -> SliceDetail: ...
def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]: ...
def describe_endpoint(self, key: ResourceKey) -> EndpointDetail: ...
def mint_endpoint_token(
    self,
    key: ResourceKey,
    *,
    ttl: Duration,
) -> EndpointToken: ...
def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]: ...
def fetch_job_logs(self, identity: JobIdentity, query: LogQuery = LogQuery()) -> LogPage: ...
def fetch_task_logs(self, identity: TaskIdentity, query: LogQuery = LogQuery()) -> LogPage: ...
def fetch_attempt_logs(
    self,
    identity: AttemptIdentity,
    query: LogQuery = LogQuery(),
) -> LogPage: ...
def stream_job_logs(self, identity: JobIdentity, query: LogQuery = LogQuery()) -> Iterator[LogEntry]: ...
def stream_task_logs(self, identity: TaskIdentity, query: LogQuery = LogQuery()) -> Iterator[LogEntry]: ...
def stream_attempt_logs(
    self,
    identity: AttemptIdentity,
    query: LogQuery = LogQuery(),
) -> Iterator[LogEntry]: ...
def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt: ...
def retry_task(
    self,
    identity: TaskIdentity,
    *,
    expected_attempt_uid: str,
    idempotency_key: str,
) -> ActionReceipt: ...
def terminate_attempt(
    self,
    identity: AttemptIdentity,
    *,
    idempotency_key: str,
) -> ActionReceipt: ...
def get_action_receipt(self, action_id: str) -> ActionReceipt: ...
def wait_for_action(self, action_id: str, *, timeout: Duration) -> ActionReceipt: ...
def exec_attempt(
    self,
    locator: AttemptLocator,
    *,
    command: Sequence[str],
    timeout: Duration,
) -> ExecResult: ...
def profile_attempt(
    self,
    locator: AttemptLocator,
    *,
    profile: ProfileType,
    duration: Duration,
) -> ProfileResult: ...
```

`wait_for_action` may time out while the durable receipt remains nonterminal.
Log helpers continue to use finelog after resolving and authorizing the typed
resource. Exec and profile accept `AttemptLocator`, resolve it once to an exact
Attempt, and refuse a replacement before contacting a backend.
`get_action_receipt` is available only to the accepting principal or an admin.
Node detail returns at most the 50 newest Attempts; longer history remains in
Activity and stats.

## Protobuf contract

Create `lib/iris/src/iris/rpc/resource.proto` with this service and message
surface. Generated `resource_pb2.py` and `resource_pb2.pyi` follow the existing
protobuf build.

```proto
edition = "2023";

package iris.resource;

import "job.proto";
import "iris_logging.proto";
import "time.proto";

option features.field_presence = EXPLICIT;

enum ResourceKind {
  RESOURCE_KIND_UNSPECIFIED = 0;
  RESOURCE_KIND_JOB = 1;
  RESOURCE_KIND_TASK = 2;
  RESOURCE_KIND_ATTEMPT = 3;
  RESOURCE_KIND_ENDPOINT = 4;
  RESOURCE_KIND_NODE = 5;
  RESOURCE_KIND_SLICE = 6;
}

enum SourceState {
  SOURCE_STATE_UNSPECIFIED = 0;
  SOURCE_STATE_AVAILABLE = 1;
  SOURCE_STATE_UNAVAILABLE = 2;
  SOURCE_STATE_UNSUPPORTED = 3;
}

enum Freshness {
  FRESHNESS_UNSPECIFIED = 0;
  FRESHNESS_CURRENT = 1;
  FRESHNESS_STALE = 2;
  FRESHNESS_UNKNOWN = 3;
}

enum NodeHealth {
  NODE_HEALTH_UNSPECIFIED = 0;
  NODE_HEALTH_READY = 1;
  NODE_HEALTH_DEGRADED = 2;
  NODE_HEALTH_UNAVAILABLE = 3;
  NODE_HEALTH_RETIRED = 4;
}

enum SliceLifecycle {
  SLICE_LIFECYCLE_UNSPECIFIED = 0;
  SLICE_LIFECYCLE_CREATING = 1;
  SLICE_LIFECYCLE_READY = 2;
  SLICE_LIFECYCLE_DELETING = 3;
  SLICE_LIFECYCLE_FAILED = 4;
}

enum MembershipState {
  MEMBERSHIP_STATE_UNSPECIFIED = 0;
  MEMBERSHIP_STATE_UNKNOWN = 1;
  MEMBERSHIP_STATE_OBSERVED = 2;
}

enum ActionKind {
  ACTION_KIND_UNSPECIFIED = 0;
  ACTION_KIND_CANCEL_JOB = 1;
  ACTION_KIND_RETRY_TASK = 2;
  ACTION_KIND_TERMINATE_ATTEMPT = 3;
}

enum ActionState {
  ACTION_STATE_UNSPECIFIED = 0;
  ACTION_STATE_ACCEPTED = 1;
  ACTION_STATE_VERIFYING = 2;
  ACTION_STATE_SUCCEEDED = 3;
  ACTION_STATE_FAILED = 4;
}

enum ActionResult {
  ACTION_RESULT_UNSPECIFIED = 0;
  ACTION_RESULT_NONE = 1;
  ACTION_RESULT_SATISFIED = 2;
  ACTION_RESULT_TARGET_ABSENT = 3;
  ACTION_RESULT_PROVIDER_REJECTED = 4;
  ACTION_RESULT_INTERNAL_ERROR = 5;
}

enum EndpointAccess {
  ENDPOINT_ACCESS_PRIVATE = 0;
  ENDPOINT_ACCESS_LINK = 1;
}

message ResourceKey {
  string cluster_id = 1;
  ResourceKind kind = 2;
  string resource_id = 3;
}

message JobIdentity {
  ResourceKey key = 1;
  string job_uid = 2;
}

message TaskIdentity {
  ResourceKey key = 1;
  string task_uid = 2;
}

message AttemptLocator {
  ResourceKey task = 1;
  int32 attempt_number = 2;
}

message AttemptIdentity {
  ResourceKey task = 1;
  int32 attempt_number = 2;
  string attempt_uid = 3;
}

message NodeIdentity {
  ResourceKey key = 1;
  string backend_id = 2;
  string node_uid = 3;
}

message NodeLocator {
  ResourceKey key = 1;
  string backend_id = 2;
  string node_uid = 3;
}

message SliceIdentity {
  ResourceKey key = 1;
  string backend_id = 2;
  string slice_uid = 3;
}

message SliceLocator {
  ResourceKey key = 1;
  string backend_id = 2;
  string slice_uid = 3;
}

message ResourceSourceStatus {
  string source_id = 1;
  string backend_id = 2;
  SourceState state = 3;
  Freshness freshness = 4;
  iris.time.Timestamp observed_at = 5;
  string error_code = 6;
  string error_message = 7;
}

message PageRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message PageInfo {
  string next_page_token = 1;
  repeated ResourceSourceStatus source_statuses = 2;
}

message JobSpec {
  uint32 version = 1;
  string name = 2;
  iris.job.RuntimeEntrypoint entrypoint = 3;
  iris.job.ResourceSpecProto resources = 4;
  iris.job.EnvironmentConfig environment = 5;
  string bundle_id = 6;
  iris.time.Duration scheduling_timeout = 7;
  repeated string ports = 8;
  int32 max_task_failures = 9;
  int32 max_retries_failure = 10;
  int32 max_retries_preemption = 11;
  repeated iris.job.Constraint constraints = 12;
  iris.job.CoschedulingConfig coscheduling = 13;
  int32 replicas = 14;
  iris.time.Duration timeout = 15;
  bool fail_if_exists = 16;
  iris.job.JobPreemptionPolicy preemption_policy = 17;
  iris.job.ExistingJobPolicy existing_job_policy = 18;
  iris.job.PriorityBand priority_band = 19;
  string task_image = 20;
  repeated string submit_argv = 21;
  string client_revision_date = 22;
  iris.job.ContainerProfile container_profile = 23;
}

message SubmitJobRequest {
  JobSpec spec = 1;
  bytes bundle_blob = 2;
}

message SubmitJobResponse {
  JobIdentity job = 1;
}

message JobQuery {
  string owner_id = 1;
  ResourceKey parent = 2;
  string job_id_prefix = 3;
  repeated iris.job.JobState states = 4;
  string backend_id = 5;
  string execution_cluster_id = 6;
  PageRequest page = 7;
}

message JobSummary {
  JobIdentity identity = 1;
  string owner_id = 2;
  JobIdentity parent = 3;
  iris.job.JobState state = 4;
  string execution_cluster_id = 5;
  string backend_id = 6;
  int32 num_tasks = 7;
  iris.time.Timestamp submitted_at = 8;
  iris.time.Timestamp started_at = 9;
  iris.time.Timestamp finished_at = 10;
  string error_message = 11;
}

message JobDetail {
  JobSummary summary = 1;
  JobSpec spec = 2;
}

message ListJobsRequest {
  JobQuery query = 1;
}

message ListJobsResponse {
  repeated JobSummary jobs = 1;
  PageInfo page = 2;
}

message DescribeJobRequest {
  ResourceKey job = 1;
}

message DescribeJobResponse {
  JobDetail job = 1;
}

message TaskQuery {
  ResourceKey job = 1;
  string job_id_prefix = 2;
  repeated iris.job.TaskState states = 3;
  string backend_id = 4;
  string authority_cluster_id = 5;
  string execution_cluster_id = 6;
  PageRequest page = 7;
}

message TaskSummary {
  TaskIdentity identity = 1;
  JobIdentity job = 2;
  int32 task_index = 3;
  iris.job.TaskState state = 4;
  string execution_cluster_id = 5;
  string backend_id = 6;
  AttemptIdentity current_attempt = 7;
  NodeIdentity current_node = 8;
  int32 failure_count = 9;
  int32 preemption_count = 10;
  iris.time.Timestamp submitted_at = 11;
  iris.time.Timestamp started_at = 12;
  iris.time.Timestamp finished_at = 13;
  string status_message = 14;
  string error_message = 15;
}

message AttemptSummary {
  AttemptIdentity identity = 1;
  iris.job.TaskState state = 2;
  string execution_cluster_id = 3;
  string backend_id = 4;
  NodeIdentity node = 5;
  iris.time.Timestamp created_at = 6;
  iris.time.Timestamp started_at = 7;
  iris.time.Timestamp finished_at = 8;
  int32 exit_code = 9;
  string error_message = 10;
  string terminal_reason = 11;
}

message AttemptRuntimeObject {
  string provider_kind = 1;
  string namespace = 2;
  string name = 3;
  string provider_uid = 4;
  string provider_node_id = 5;
  string provider_node_uid = 6;
  string container_id = 7;
  iris.time.Timestamp observed_at = 8;
}

message TaskDetail {
  TaskSummary summary = 1;
  repeated AttemptSummary attempts = 2;
  repeated ResourceSourceStatus source_statuses = 3;
  repeated string root_cause_highlights = 4;
}

message AttemptDetail {
  AttemptSummary summary = 1;
  AttemptRuntimeObject runtime = 2;
  repeated ResourceSourceStatus source_statuses = 3;
}

message ListTasksRequest {
  TaskQuery query = 1;
}

message ListTasksResponse {
  repeated TaskSummary tasks = 1;
  PageInfo page = 2;
}

message DescribeTaskRequest {
  ResourceKey task = 1;
}

message DescribeTaskResponse {
  TaskDetail task = 1;
}

message DescribeAttemptRequest {
  AttemptLocator attempt = 1;
}

message DescribeAttemptResponse {
  AttemptDetail attempt = 1;
}

message NodeQuery {
  string backend_id = 1;
  string contains = 2;
  repeated NodeHealth health = 3;
  PageRequest page = 4;
}

message NodeCapacity {
  int64 cpu_millicores = 1;
  int64 memory_bytes = 2;
  int64 disk_bytes = 3;
  string accelerator_kind = 4;
  string accelerator_variant = 5;
  int32 accelerator_count = 6;
}

message NodeSummary {
  NodeIdentity identity = 1;
  NodeHealth health = 2;
  bool schedulable = 3;
  NodeCapacity capacity = 4;
  string scaling_group_id = 5;
  SliceIdentity slice = 6;
  int32 running_task_count = 7;
  iris.time.Timestamp observed_at = 8;
}

message NodeAttribute {
  string key = 1;
  oneof value {
    string string_value = 2;
    int64 integer_value = 3;
    double float_value = 4;
  }
}

message NodeDetail {
  NodeSummary summary = 1;
  string address = 2;
  repeated NodeAttribute attributes = 3;
  repeated AttemptSummary recent_attempts = 4;
  string bootstrap_log_key = 5;
  repeated ResourceSourceStatus source_statuses = 6;
}

message ListNodesRequest {
  NodeQuery query = 1;
}

message ListNodesResponse {
  repeated NodeSummary nodes = 1;
  PageInfo page = 2;
}

message DescribeNodeRequest {
  NodeLocator node = 1;
}

message DescribeNodeResponse {
  NodeDetail node = 1;
}

message SliceQuery {
  string backend_id = 1;
  string scaling_group_id = 2;
  PageRequest page = 3;
}

message SliceSummary {
  SliceIdentity identity = 1;
  string scaling_group_id = 2;
  SliceLifecycle lifecycle = 3;
  MembershipState membership_state = 4;
  int32 observed_member_count = 5;
  iris.time.Timestamp observed_at = 6;
  string error_message = 7;
}

message SliceMember {
  string provider_node_id = 1;
  NodeIdentity node = 2;
  iris.time.Timestamp observed_at = 3;
}

message SliceDetail {
  SliceSummary summary = 1;
  repeated SliceMember members = 2;
  repeated ResourceSourceStatus source_statuses = 3;
}

message ListSlicesRequest {
  SliceQuery query = 1;
}

message ListSlicesResponse {
  repeated SliceSummary slices = 1;
  PageInfo page = 2;
}

message DescribeSliceRequest {
  SliceLocator slice = 1;
}

message DescribeSliceResponse {
  SliceDetail slice = 1;
}

message EndpointQuery {
  string name_prefix = 1;
  ResourceKey task = 2;
  PageRequest page = 3;
}

message EndpointSummary {
  ResourceKey key = 1;
  string endpoint_id = 2;
  string name = 3;
  ResourceKey task = 4;
  string execution_cluster_id = 5;
  EndpointAccess access = 6;
  iris.time.Timestamp lease_deadline = 7;
}

message EndpointDetail {
  EndpointSummary summary = 1;
  string address = 2;
  map<string, string> metadata = 3;
}

message ListEndpointsRequest {
  EndpointQuery query = 1;
}

message ListEndpointsResponse {
  repeated EndpointSummary endpoints = 1;
  PageInfo page = 2;
}

message DescribeEndpointRequest {
  ResourceKey endpoint = 1;
}

message DescribeEndpointResponse {
  EndpointDetail endpoint = 1;
}

message MintEndpointTokenRequest {
  ResourceKey endpoint = 1;
  iris.time.Duration ttl = 2;
}

message MintEndpointTokenResponse {
  string token = 1;
  iris.time.Timestamp expires_at = 2;
  string capability_url = 3;
}

message ActivityQuery {
  ResourceKey target = 1;
  string attempt_uid = 2;
  iris.time.Timestamp after = 3;
  PageRequest page = 4;
}

message ActivityEntry {
  string entry_id = 1;
  iris.time.Timestamp occurred_at = 2;
  string source = 3;
  string severity = 4;
  string kind = 5;
  string message = 6;
  ResourceKey target = 7;
  string attempt_uid = 8;
  string correlation_id = 9;
  map<string, string> attributes = 10;
}

message ListActivityRequest {
  ActivityQuery query = 1;
}

message ListActivityResponse {
  repeated ActivityEntry entries = 1;
  PageInfo page = 2;
}

message LogQuery {
  iris.time.Timestamp after = 1;
  int64 cursor = 2;
  int32 max_lines = 3;
  string substring = 4;
  iris.logging.LogLevel minimum_level = 5;
  bool tail = 6;
}

message LogTarget {
  oneof target {
    JobIdentity job = 1;
    TaskIdentity task = 2;
    AttemptIdentity attempt = 3;
  }
}

message FetchLogsRequest {
  LogTarget target = 1;
  LogQuery query = 2;
}

message FetchLogsResponse {
  repeated iris.logging.LogEntry entries = 1;
  int64 next_cursor = 2;
  repeated ResourceSourceStatus source_statuses = 3;
}

message ActionReceipt {
  string action_id = 1;
  ActionKind kind = 2;
  ResourceKey target = 3;
  string expected_target_uid = 4;
  string expected_attempt_uid = 5;
  ActionState state = 6;
  ActionResult result_code = 7;
  string result_message = 8;
  iris.time.Timestamp created_at = 9;
  iris.time.Timestamp updated_at = 10;
  iris.time.Timestamp completed_at = 11;
}

message CancelJobRequest {
  JobIdentity job = 1;
  string idempotency_key = 2;
}

message RetryTaskRequest {
  TaskIdentity task = 1;
  string expected_attempt_uid = 2;
  string idempotency_key = 3;
}

message TerminateAttemptRequest {
  AttemptIdentity attempt = 1;
  string idempotency_key = 2;
}

message ActionResponse {
  ActionReceipt receipt = 1;
}

message GetActionReceiptRequest {
  string action_id = 1;
}

message ExecAttemptRequest {
  AttemptLocator attempt = 1;
  repeated string command = 2;
  iris.time.Duration timeout = 3;
}

message ExecAttemptResponse {
  int32 exit_code = 1;
  string stdout = 2;
  string stderr = 3;
  string error_message = 4;
}

message ProfileAttemptRequest {
  AttemptLocator attempt = 1;
  iris.job.ProfileType profile = 2;
  iris.time.Duration duration = 3;
}

message ProfileAttemptResponse {
  bytes profile_data = 1;
  string error_message = 2;
}

service ResourceService {
  rpc SubmitJob(SubmitJobRequest) returns (SubmitJobResponse);
  rpc ListJobs(ListJobsRequest) returns (ListJobsResponse);
  rpc DescribeJob(DescribeJobRequest) returns (DescribeJobResponse);
  rpc ListTasks(ListTasksRequest) returns (ListTasksResponse);
  rpc DescribeTask(DescribeTaskRequest) returns (DescribeTaskResponse);
  rpc DescribeAttempt(DescribeAttemptRequest) returns (DescribeAttemptResponse);
  rpc ListNodes(ListNodesRequest) returns (ListNodesResponse);
  rpc DescribeNode(DescribeNodeRequest) returns (DescribeNodeResponse);
  rpc ListSlices(ListSlicesRequest) returns (ListSlicesResponse);
  rpc DescribeSlice(DescribeSliceRequest) returns (DescribeSliceResponse);
  rpc ListEndpoints(ListEndpointsRequest) returns (ListEndpointsResponse);
  rpc DescribeEndpoint(DescribeEndpointRequest) returns (DescribeEndpointResponse);
  rpc MintEndpointToken(MintEndpointTokenRequest) returns (MintEndpointTokenResponse);
  rpc ListActivity(ListActivityRequest) returns (ListActivityResponse);
  rpc FetchLogs(FetchLogsRequest) returns (FetchLogsResponse);
  rpc CancelJob(CancelJobRequest) returns (ActionResponse);
  rpc RetryTask(RetryTaskRequest) returns (ActionResponse);
  rpc TerminateAttempt(TerminateAttemptRequest) returns (ActionResponse);
  rpc GetActionReceipt(GetActionReceiptRequest) returns (ActionResponse);
  rpc ExecAttempt(ExecAttemptRequest) returns (ExecAttemptResponse);
  rpc ProfileAttempt(ProfileAttemptRequest) returns (ProfileAttemptResponse);
}
```

`controller.proto` retains checkpoint, authentication, budgets, scheduler,
backend summaries, endpoint registration, worker registration, and federation
sync. It removes `LaunchJob`, `GetJobStatus`, `GetJobState`, `TerminateJob`,
`ListJobs`, `GetTaskStatus`, the old `ListTasks`, `KickTasks`, `ListWorkers`,
`GetWorkerStatus`, `GetKubernetesClusterStatus`, `ExecInContainer`, and
`ProfileTask`. First-party callers use `ResourceService`; there are no adapter
RPCs.

`EndpointService` retains registration and unregistration for task runtimes,
uses `iris.resource.EndpointAccess`, and no longer lists endpoints. The nested
controller Endpoint enum and list messages are deleted.

Federation deltas carry `JobIdentity`, `TaskIdentity`, Attempt summaries, and
the canonical backend/execution coordinates from `resource.proto`. Federation
action envelopes carry the authority action ID, action kind, exact target UID,
expected Attempt UID, principal, idempotency key, and payload hash. The peer
deduplicates the authority action ID and reports the same receipt state through
sync.

Federation authority is fixed as follows:

- authority identity comes from the retained handoff plus authenticated peer;
- execution cluster is the authenticated peer that accepted the handoff;
- promotion backend is capacity advice only and is not persisted as placement;
- the execution peer is authoritative for actual backend placement;
- a pending parent Job may adopt the first authenticated non-empty backend only
  when every non-empty Task assertion in the same delta agrees;
- that transaction fills the Job, every retained Task, and every mirrored
  Attempt backend, including unchanged Tasks; omitted Task coordinates inherit
  the canonical Job backend;
- a Task Attempt has no independent backend assertion on the wire and inherits
  its validated Task backend; and
- an intra-delta mismatch or any later replacement rejects the whole sync
  transaction without changing cursor, Job, Task, Attempt, or endpoint mirrors.

The public query budget is retained by bulk-loading all affected root Tasks once;
federation apply never reads one Task per delta row.

## Persisted schema

`lib/iris/src/iris/cluster/controller/persistence/schema/` owns the final DDL.
The schema remains noun-specific; there is no `resources` table.

`schema/base.py` declares:

```sql
CREATE TABLE schema_migrations (
    name TEXT PRIMARY KEY,
    source_fingerprint TEXT NOT NULL,
    applied_at_ms INTEGER NOT NULL
);

CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

### Workloads

`schema/workloads.py` declares:

```sql
CREATE TABLE jobs (
    job_uid TEXT PRIMARY KEY,
    authority_cluster_id TEXT NOT NULL CHECK (authority_cluster_id <> ''),
    job_id TEXT NOT NULL,
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    backend_id TEXT NOT NULL,
    placement_state TEXT NOT NULL CHECK (
        placement_state IN ('pending', 'known') AND
        ((placement_state = 'pending' AND backend_id = '') OR
         (placement_state = 'known' AND backend_id <> ''))
    ),
    owner_id TEXT NOT NULL,
    submitting_principal TEXT NOT NULL,
    parent_job_uid TEXT REFERENCES jobs(job_uid) ON DELETE CASCADE,
    root_job_uid TEXT NOT NULL REFERENCES jobs(job_uid),
    depth INTEGER NOT NULL CHECK (depth >= 0),
    state INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    root_submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    scheduling_deadline_at_ms INTEGER,
    error_message TEXT NOT NULL DEFAULT '',
    exit_code INTEGER,
    num_tasks INTEGER NOT NULL CHECK (num_tasks >= 0),
    name TEXT NOT NULL,
    UNIQUE (authority_cluster_id, job_id)
);

CREATE INDEX jobs_parent ON jobs(parent_job_uid);
CREATE INDEX jobs_state_submitted ON jobs(state, submitted_at_ms DESC);
CREATE INDEX jobs_owner_state ON jobs(owner_id, state);
CREATE INDEX jobs_backend_state ON jobs(backend_id, state);
CREATE INDEX jobs_execution_state ON jobs(execution_cluster_id, state);

CREATE TABLE job_specs (
    job_uid TEXT PRIMARY KEY REFERENCES jobs(job_uid) ON DELETE CASCADE,
    spec_version INTEGER NOT NULL CHECK (spec_version = 1),
    resources_json TEXT NOT NULL CHECK (json_valid(resources_json)),
    entrypoint_json TEXT NOT NULL CHECK (json_valid(entrypoint_json)),
    environment_json TEXT NOT NULL CHECK (json_valid(environment_json)),
    constraints_json TEXT NOT NULL CHECK (json_valid(constraints_json)),
    coscheduling_json TEXT NOT NULL CHECK (json_valid(coscheduling_json)),
    bundle_id TEXT NOT NULL,
    ports_json TEXT NOT NULL CHECK (json_valid(ports_json)),
    scheduling_timeout_ms INTEGER,
    max_task_failures INTEGER NOT NULL CHECK (max_task_failures >= 0),
    max_retries_failure INTEGER NOT NULL CHECK (max_retries_failure >= 0),
    max_retries_preemption INTEGER NOT NULL CHECK (max_retries_preemption >= 0),
    replicas INTEGER NOT NULL CHECK (replicas > 0),
    timeout_ms INTEGER,
    fail_if_exists INTEGER NOT NULL CHECK (fail_if_exists IN (0, 1)),
    preemption_policy INTEGER NOT NULL,
    existing_job_policy INTEGER NOT NULL,
    priority_band INTEGER NOT NULL,
    task_image TEXT NOT NULL,
    submit_argv_json TEXT NOT NULL CHECK (json_valid(submit_argv_json)),
    client_revision_date TEXT NOT NULL,
    container_profile INTEGER NOT NULL
);

CREATE TABLE job_workdir_files (
    job_uid TEXT NOT NULL REFERENCES jobs(job_uid) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (job_uid, filename)
);

CREATE TABLE tasks (
    task_uid TEXT PRIMARY KEY,
    authority_cluster_id TEXT NOT NULL CHECK (authority_cluster_id <> ''),
    task_id TEXT NOT NULL,
    job_uid TEXT NOT NULL REFERENCES jobs(job_uid) ON DELETE CASCADE,
    task_index INTEGER NOT NULL CHECK (task_index >= 0),
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    backend_id TEXT NOT NULL,
    placement_state TEXT NOT NULL CHECK (
        placement_state IN ('pending', 'known') AND
        ((placement_state = 'pending' AND backend_id = '') OR
         (placement_state = 'known' AND backend_id <> ''))
    ),
    state INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    error_message TEXT NOT NULL DEFAULT '',
    status_message TEXT NOT NULL DEFAULT '',
    exit_code INTEGER,
    max_retries_failure INTEGER NOT NULL CHECK (max_retries_failure >= 0),
    max_retries_preemption INTEGER NOT NULL CHECK (max_retries_preemption >= 0),
    current_attempt_uid TEXT,
    current_node_uid TEXT,
    priority_band INTEGER NOT NULL,
    priority_neg_depth INTEGER NOT NULL,
    priority_root_submitted_ms INTEGER NOT NULL,
    priority_insertion INTEGER NOT NULL,
    UNIQUE (authority_cluster_id, task_id),
    UNIQUE (job_uid, task_index),
    FOREIGN KEY (current_attempt_uid) REFERENCES attempts(attempt_uid)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX tasks_job_state ON tasks(job_uid, state);
CREATE INDEX tasks_backend_state ON tasks(backend_id, state);
CREATE INDEX tasks_execution_state ON tasks(execution_cluster_id, state);
CREATE INDEX tasks_current_attempt ON tasks(current_attempt_uid)
    WHERE current_attempt_uid IS NOT NULL;
CREATE INDEX tasks_pending ON tasks(
    state,
    priority_band,
    priority_neg_depth,
    priority_root_submitted_ms,
    submitted_at_ms,
    priority_insertion
);

CREATE TABLE attempts (
    attempt_uid TEXT PRIMARY KEY,
    task_uid TEXT NOT NULL REFERENCES tasks(task_uid) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 0),
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    backend_id TEXT NOT NULL CHECK (backend_id <> ''),
    node_uid TEXT,
    state INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    exit_code INTEGER,
    error_message TEXT NOT NULL DEFAULT '',
    terminal_reason TEXT NOT NULL DEFAULT '',
    UNIQUE (task_uid, attempt_number)
);

CREATE INDEX attempts_task_state ON attempts(task_uid, state, started_at_ms);
CREATE INDEX attempts_backend ON attempts(backend_id);
CREATE INDEX attempts_node ON attempts(node_uid) WHERE node_uid IS NOT NULL;

CREATE TABLE attempt_runtime_objects (
    attempt_uid TEXT PRIMARY KEY REFERENCES attempts(attempt_uid) ON DELETE CASCADE,
    provider_kind TEXT NOT NULL CHECK (provider_kind IN ('kubernetes', 'rpc')),
    namespace TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    provider_uid TEXT NOT NULL DEFAULT '',
    provider_node_id TEXT NOT NULL DEFAULT '',
    provider_node_uid TEXT NOT NULL DEFAULT '',
    container_id TEXT NOT NULL DEFAULT '',
    observed_at_ms INTEGER NOT NULL,
    CHECK (
        (
            provider_kind = 'kubernetes' AND namespace <> '' AND
            name <> '' AND provider_uid <> ''
        ) OR (
            provider_kind = 'rpc' AND provider_node_uid <> '' AND container_id <> ''
        )
    )
);

CREATE INDEX attempt_runtime_provider_uid
    ON attempt_runtime_objects(provider_kind, provider_uid)
    WHERE provider_uid <> '';

CREATE TABLE endpoints (
    endpoint_id TEXT PRIMARY KEY,
    authority_cluster_id TEXT NOT NULL CHECK (authority_cluster_id <> ''),
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    owner_job_id TEXT NOT NULL,
    owner_task_id TEXT,
    owner_job_uid TEXT REFERENCES jobs(job_uid) ON DELETE CASCADE,
    owner_task_uid TEXT REFERENCES tasks(task_uid) ON DELETE CASCADE,
    peer_id TEXT,
    metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
    access INTEGER NOT NULL CHECK (access IN (0, 1)),
    registered_at_ms INTEGER NOT NULL,
    lease_deadline_at_ms INTEGER
);

CREATE INDEX endpoints_name ON endpoints(name);
CREATE INDEX endpoints_owner_task ON endpoints(authority_cluster_id, owner_task_id);
CREATE INDEX endpoints_peer ON endpoints(peer_id) WHERE peer_id IS NOT NULL;
```

The `tasks.current_attempt_uid` foreign key is deferred because a control-loop
transaction may insert an Attempt and update its Task in either statement order.
`current_node_uid` is the exact current-Attempt hot-path mirror. It has no foreign
key because live Kubernetes Nodes do not have `rpc_nodes` rows.

`placement_state='pending'` is legal only for a SENT federated Job and its Tasks
before the authenticated execution peer reports actual placement. Such rows have
no Attempts. Local submissions resolve a configured backend before inserting
the Job, and every Attempt has a known non-empty backend. Describe remains
durable for pending placement, but logs, exec, profile, retry, and terminate fail
with `BackendIdentityUnknown` until adoption completes.

Runtime observations merge only when `observed_at_ms` increases. An equal-time,
equal-identity observation may fill an empty optional Node ID, but may not
replace a non-empty field. Older observations are ignored. A provider UID, RPC
Node UID, or container conflict for one exact Attempt is reported as unavailable
and does not mutate the sidecar. Attempt state/times and count projections are
never updated by runtime enrichment. An identical observation issues no Attempt,
Task, or sidecar DML. For the exact current Attempt, the same
transaction mirrors Kubernetes or RPC `provider_node_uid` into
`tasks.current_node_uid`; non-current observations never touch the Task.

### Nodes and capacity

`schema/execution.py` declares only durable RPC registrations and provider Slice
projections. Kubernetes Nodes remain cached backend observations.

```sql
CREATE TABLE rpc_nodes (
    node_uid TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    backend_id TEXT NOT NULL CHECK (backend_id <> ''),
    scaling_group_id TEXT,
    registered_at_ms INTEGER NOT NULL,
    last_seen_at_ms INTEGER NOT NULL,
    retired_at_ms INTEGER
);

CREATE UNIQUE INDEX current_rpc_node_logical_id
    ON rpc_nodes(execution_cluster_id, backend_id, node_id)
    WHERE retired_at_ms IS NULL;
CREATE INDEX rpc_nodes_scaling_group
    ON rpc_nodes(execution_cluster_id, backend_id, scaling_group_id);

CREATE TABLE rpc_node_details (
    node_uid TEXT PRIMARY KEY REFERENCES rpc_nodes(node_uid) ON DELETE CASCADE,
    address TEXT NOT NULL,
    hostname TEXT NOT NULL,
    ip_address TEXT NOT NULL,
    provider_instance_id TEXT NOT NULL,
    provider_zone TEXT NOT NULL,
    provenance_json TEXT NOT NULL CHECK (json_valid(provenance_json))
);

CREATE TABLE node_capacity (
    node_uid TEXT PRIMARY KEY REFERENCES rpc_nodes(node_uid) ON DELETE CASCADE,
    cpu_millicores INTEGER NOT NULL CHECK (cpu_millicores >= 0),
    memory_bytes INTEGER NOT NULL CHECK (memory_bytes >= 0),
    disk_bytes INTEGER NOT NULL CHECK (disk_bytes >= 0),
    accelerator_kind TEXT NOT NULL,
    accelerator_variant TEXT NOT NULL,
    accelerator_count INTEGER NOT NULL CHECK (accelerator_count >= 0)
);

CREATE TABLE node_attributes (
    node_uid TEXT NOT NULL REFERENCES rpc_nodes(node_uid) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (value_type IN ('str', 'int', 'float')),
    str_value TEXT,
    int_value INTEGER,
    float_value REAL,
    PRIMARY KEY (node_uid, key),
    CHECK (
        (value_type = 'str' AND str_value IS NOT NULL AND int_value IS NULL AND float_value IS NULL) OR
        (value_type = 'int' AND str_value IS NULL AND int_value IS NOT NULL AND float_value IS NULL) OR
        (value_type = 'float' AND str_value IS NULL AND int_value IS NULL AND float_value IS NOT NULL)
    )
);

CREATE TABLE scaling_groups (
    execution_cluster_id TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    scaling_group_id TEXT NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    backoff_until_ms INTEGER NOT NULL DEFAULT 0,
    last_scale_up_at_ms INTEGER NOT NULL DEFAULT 0,
    last_scale_down_at_ms INTEGER NOT NULL DEFAULT 0,
    quota_exceeded_until_ms INTEGER NOT NULL DEFAULT 0,
    quota_reason TEXT NOT NULL DEFAULT '',
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (execution_cluster_id, backend_id, scaling_group_id)
);

CREATE TABLE slices (
    slice_uid TEXT PRIMARY KEY,
    slice_id TEXT NOT NULL,
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    backend_id TEXT NOT NULL CHECK (backend_id <> ''),
    scaling_group_id TEXT NOT NULL,
    management_mode TEXT NOT NULL CHECK (management_mode IN ('autoscaled', 'manual')),
    lifecycle TEXT NOT NULL CHECK (
        lifecycle IN ('creating', 'ready', 'deleting', 'failed')
    ),
    membership_state TEXT NOT NULL CHECK (
        membership_state IN ('unknown', 'observed')
    ),
    created_at_ms INTEGER NOT NULL,
    observed_at_ms INTEGER,
    error_message TEXT NOT NULL DEFAULT '',
    UNIQUE (execution_cluster_id, backend_id, slice_id),
    FOREIGN KEY (execution_cluster_id, backend_id, scaling_group_id)
        REFERENCES scaling_groups(execution_cluster_id, backend_id, scaling_group_id)
);

CREATE INDEX slices_scaling_group
    ON slices(execution_cluster_id, backend_id, scaling_group_id);

CREATE TABLE slice_members (
    slice_uid TEXT NOT NULL REFERENCES slices(slice_uid) ON DELETE CASCADE,
    provider_node_id TEXT NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    PRIMARY KEY (slice_uid, provider_node_id)
);
```

`membership_state='unknown'` plus no rows is unknown. `observed` plus no rows is
known empty. Registered Node links are derived by backend-qualified matching of
`slice_members.provider_node_id` to current `rpc_nodes.node_id`.

### Actions

`schema/operations.py` declares:

```sql
CREATE TABLE action_receipts (
    action_id TEXT PRIMARY KEY,
    authority_cluster_id TEXT NOT NULL CHECK (authority_cluster_id <> ''),
    authority_action_id TEXT NOT NULL,
    action_kind TEXT NOT NULL CHECK (
        action_kind IN ('cancel_job', 'retry_task', 'terminate_attempt')
    ),
    target_kind TEXT NOT NULL CHECK (
        target_kind IN ('job', 'task', 'attempt')
    ),
    target_id TEXT NOT NULL,
    expected_target_uid TEXT NOT NULL CHECK (expected_target_uid <> ''),
    expected_attempt_uid TEXT NOT NULL DEFAULT '',
    backend_id TEXT NOT NULL DEFAULT '',
    execution_cluster_id TEXT NOT NULL CHECK (execution_cluster_id <> ''),
    principal_id TEXT NOT NULL,
    client_idempotency_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('accepted', 'verifying', 'succeeded', 'failed')
    ),
    result_code TEXT NOT NULL CHECK (
        result_code IN (
            'none', 'satisfied', 'target_absent',
            'provider_rejected', 'internal_error'
        )
    ),
    result_message TEXT NOT NULL DEFAULT '',
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    completed_at_ms INTEGER,
    UNIQUE (authority_cluster_id, authority_action_id),
    UNIQUE (principal_id, action_kind, client_idempotency_key),
    CHECK (
        (action_kind = 'cancel_job' AND target_kind = 'job' AND expected_attempt_uid = '') OR
        (action_kind = 'retry_task' AND target_kind = 'task' AND expected_attempt_uid <> '') OR
        (action_kind = 'terminate_attempt' AND target_kind = 'attempt' AND expected_attempt_uid <> '')
    ),
    CHECK (action_kind = 'cancel_job' OR backend_id <> ''),
    CHECK (
        (state IN ('accepted', 'verifying') AND result_code = 'none' AND completed_at_ms IS NULL) OR
        (state = 'succeeded' AND result_code IN ('satisfied', 'target_absent') AND completed_at_ms IS NOT NULL) OR
        (state = 'failed' AND result_code IN ('provider_rejected', 'internal_error') AND completed_at_ms IS NOT NULL)
    )
);

CREATE INDEX action_receipts_state
    ON action_receipts(state, updated_at_ms);
CREATE INDEX action_receipts_target
    ON action_receipts(target_kind, target_id, updated_at_ms DESC);
CREATE INDEX action_receipts_principal
    ON action_receipts(principal_id, updated_at_ms DESC);
```

Receipts retain logical target data without foreign keys so pruning a completed
Job does not erase its audit result. Completed receipts are retained for 30 days.
Nonterminal receipts are not removed by age.

### Existing domains

`schema/federation.py` declares:

```sql
CREATE TABLE federated_jobs (
    job_uid TEXT PRIMARY KEY REFERENCES jobs(job_uid) ON DELETE CASCADE,
    direction TEXT NOT NULL CHECK (direction IN ('sent', 'received')),
    peer_id TEXT NOT NULL,
    owner_principal TEXT NOT NULL,
    handoff_state TEXT CHECK (
        (direction = 'sent' AND handoff_state IS NOT NULL AND handoff_state IN
            ('queued', 'pending', 'handed_off', 'rejected')) OR
        (direction = 'received' AND handoff_state IS NULL)
    ),
    cancel_intent_version INTEGER NOT NULL DEFAULT 0,
    handoff_nonce TEXT NOT NULL,
    UNIQUE (direction, peer_id, handoff_nonce)
);

CREATE INDEX federated_jobs_direction_peer
    ON federated_jobs(direction, peer_id, handoff_state);

CREATE TABLE federation_sync_state (
    peer_id TEXT PRIMARY KEY,
    cursor TEXT NOT NULL
);

CREATE TABLE federated_tasks (
    task_uid TEXT PRIMARY KEY REFERENCES tasks(task_uid) ON DELETE CASCADE,
    peer_node_label TEXT NOT NULL DEFAULT ''
);

CREATE TABLE federation_changelog (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    authority_cluster_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    job_uid TEXT NOT NULL,
    task_uid TEXT,
    requester_id TEXT NOT NULL,
    tombstone INTEGER NOT NULL CHECK (tombstone IN (0, 1)),
    written_at_ms INTEGER NOT NULL
);

CREATE INDEX federation_changelog_requester
    ON federation_changelog(requester_id, seq);
```

The changelog intentionally has no Job/Task foreign keys because a tombstone
must outlive pruned resource rows. Migration maps direction `0/1` to
`sent/received` and handoff state `0/1/2/3` to
`pending/handed_off/rejected/queued`; every other value fails preflight.

`schema/operations.py` also retains budgets with exact final DDL:

```sql
CREATE TABLE user_budgets (
    owner_id TEXT PRIMARY KEY,
    budget_limit INTEGER NOT NULL DEFAULT 0,
    max_band INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
```

There are no other persisted projection or rollout tables. Projection caches are
reconstructed from the noun tables. Rollout image/source/schema information
remains in the existing file-backed rollout record, and the `meta` table retains
only named integer/string values including the checkpoint epoch.

The final database contains one SQLite file. `auth.sqlite3`, `ATTACH auth`, auth
backup/restore handling, and migrations whose only purpose was the removed auth
schema are deleted.

### Checkpoint layout

Resource-schema checkpoints use a new prefix so an old image cannot select a
main-only v2 checkpoint:

```text
<remote_state_dir>/checkpoints/v2/<checkpoint_epoch_ms>/controller.sqlite3.zst
<remote_state_dir>/checkpoints/v2/<checkpoint_epoch_ms>/manifest.json
```

`manifest.json` is written last and contains exactly:

```json
{
  "format_version": 2,
  "schema_epoch": 2,
  "checkpoint_epoch_ms": 0,
  "controller_sha256": "<64 lowercase hex>",
  "controller_size_bytes": 0
}
```

The latest selector considers only numeric v2 directories with both objects, a
valid manifest, matching size/hash, and epoch equal to the directory name. A
failed main upload or missing manifest is undiscoverable. The live database's
checkpoint epoch advances only after the complete pair is readable and verified.
Restore requires the same checks and never creates a missing sidecar.

Pre-migration v1 checkpoints remain under the old numeric prefix and are
complete only when main and auth backups are both present, compressed or legacy
uncompressed. The v2 controller does not select them for ordinary restore. The
migration preflight may inspect an explicitly named v1 checkpoint; disaster
rollback uses the old image and that complete v1 pair. After a successful v2
migration and health check, startup removes the stale local `auth.sqlite3`.
Restoring the v1 disaster checkpoint recreates it.

## Migration contract

`lib/iris/src/iris/cluster/controller/persistence/schema/version.py` defines:

```python
RESOURCE_SCHEMA_EPOCH: Final[int] = 2
RESOURCE_SCHEMA_NAME: Final[str] = "resource_schema_v2"
MERGE_BASE_SCHEMA_FINGERPRINT: Final[str]
MERGE_BASE_MIGRATION_NAMES: Final[tuple[str, ...]]
```

`lib/iris/src/iris/cluster/controller/persistence/migrate.py` exposes:

```python
@dataclass(frozen=True, slots=True)
class SchemaStatus:
    epoch: int | None
    schema_fingerprint: str
    migration_names: tuple[str, ...]
    accepted: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationContext:
    cluster_id: str
    backend_kinds: Mapping[str, str]
    scale_group_to_backend: Mapping[str, str]
    backend_namespaces: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MigrationProblem:
    name: str
    count: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationReport:
    schema: SchemaStatus
    problems: tuple[MigrationProblem, ...]

    @property
    def accepted(self) -> bool: ...


def inspect_schema(connection: sqlite3.Connection) -> SchemaStatus:
    """Inspect tables, normalized SQL, indexes, foreign keys, and migration ledger without writing."""


def preflight_database(
    database_path: Path,
    *,
    context: MigrationContext,
) -> MigrationReport:
    """Run the complete bounded row and schema validation without source writes."""


def initialize_or_upgrade_database(
    database_path: Path,
    *,
    context: MigrationContext,
) -> None:
    """Create resource schema v2 or atomically upgrade the one accepted source schema."""
```

A fresh database has no user tables and receives the final v2 schema directly.
An existing database must match the physical fingerprint and the exact ordered
merge-base migration names. The fingerprint
includes normalized table and index SQL, columns without relying on CIDs,
foreign-key actions, CHECK predicates, partial-index predicates, and every
user-defined index. Migration source bytes were never stored in the legacy
database and are build provenance, not database evidence.

Before its first DDL statement, the upgrader validates every retained row in
bounded keyset batches. Mapping state lives in temporary SQLite tables rather
than unbounded Python dictionaries. It rejects malformed JSON,
orphan relationships, non-integer Task indexes, duplicate identities, unknown
backends outside a valid pending SENT handoff, ambiguous federation direction,
and provider runtime evidence that cannot be tied to an exact Attempt.
Validation is read-only.

The UUIDv5 namespace is
`2c72b7f4-a156-5d27-8b58-7de28d5ec4cc`. Names are UTF-8 strings joined with a
single NUL byte and prefixed by `iris-resource-v2`. Decimal integers have no
leading zero. The accepted mapping is:

- retained Job UIDs use `job\0<authority_cluster_id>\0<job_id>`;
- retained Task UIDs use `task\0<job_uid>\0<task_index>`;
- existing Attempt UIDs remain unchanged;
- current Worker rows become RPC Node registrations using
  `node\0<execution_cluster_id>\0<backend_id>\0<worker_id>`;
- retained Slice UIDs use
  `slice\0<execution_cluster_id>\0<backend_id>\0<slice_id>`;
- legacy K8s Attempt fields become an `attempt_runtime_objects` row only when
  configured backend namespace, pod name, and pod UID are all available. Its
  conservative `observed_at_ms` is the first non-null of
  Attempt finished, started, and created time, in that order. Task
  `container_id` alone is never historical Attempt evidence;
- Endpoint access maps legacy NULL/PRIVATE `0` to PRIVATE `0` and legacy LINK
  `2` to LINK `1`; every other stored value fails preflight;
- authority and execution coordinates come from the Job/Task federation handle
  plus authenticated peer, while backend coordinates come from the retained
  Task/Attempt placement and must agree within each Job.

New Job, Task, RPC Node, and Slice UIDs are lowercase RFC 4122 UUIDv4 strings.
Migrated UUIDv5 values use the same lowercase hyphenated representation. Existing
Attempt UIDs remain lowercase 16-hex values; new Attempts use 32 lowercase hex
characters from `secrets.token_hex(16)`. Parsers accept exactly those two Attempt
lengths. A random collision retries three times before failing the transaction.
A deterministic UID collision between distinct source tuples fails preflight
and reports both logical IDs.

After validation, one SQLite transaction creates the final tables, copies the
validated rows, checks counts and relationships, drops the old tables, and
installs the final names and indexes. The final `schema_migrations` table has one
`resource_schema_v2` row with its source fingerprint and application time.
Fresh and upgraded databases must produce the same normalized final
fingerprint. A failed validation or transaction leaves the source database
schema and migration ledger unchanged and usable by the merge-base image.

`lib/iris/src/iris/cluster/controller/migrations/0027_*.py` through `0050_*.py`
and the source-file migration runner are deleted. The v2 upgrader is not a
general historical migration system and never executes migration code from a
database ledger.

`python -m iris.cluster.controller.persistence.migrate preflight --config
<cluster.yaml> --database <controller.sqlite3>` runs `preflight_database` and
emits bounded problem counts and sample IDs. Rollout runs it against a fresh
checkpoint from every controller before scheduling an outage. Startup reruns the
same validation to close the gap between checkpoint and cutover.
Each named problem carries an exact count and at most 20 stable, sorted sample
IDs.

The rollout tool computes connected components from the configured federation
graph. It preflights every member, quiesces Job submission and action acceptance
on every member, checkpoints every member, stops the old controllers, and starts
the new immutable image across the component. A component is reopened only when
every member reports schema epoch 2 and the new federation protocol. Independent
components may roll separately. No new controller communicates with an old peer.

## Read and mutation semantics

### Reads

`controller/resources/jobs.py`, `tasks.py`, `attempts.py`, `nodes.py`,
`slices.py`, `endpoints.py`, and `activity.py` authorize the typed parent before
assembling a response. Job, Task, Attempt, and Endpoint summaries come from one
SQLite snapshot. Node and Slice lists merge bounded backend snapshots and emit
one `ResourceSourceStatus` per backend.

List endpoints have fixed SQL statement budgets independent of returned row
count. Global Task federation queries pass large ID sets through one JSON bind
and `json_each`, keeping each statement under SQLite's 32,767 bind limit.
Provider list failures are represented in `source_statuses`; they do not erase
healthy sources or become empty success.

Task and Attempt describe share one resolver. It reads the current Attempt UID,
backend, execution cluster, and runtime identity in one snapshot, then calls at
most one exact backend. If any coordinate changes before the live call, the
backend receives no request and the response reports replacement or stale
detail. Logs, exec, and profile use the same resolver.

### Actions

`controller/operator_commands.py` provides bounded pre-acceptance ingress. The
service authorizes and validates an envelope, wakes the controller, and waits
for the control-loop transaction. It does not write resource state or report
acceptance itself.

At the start of a tick, the single writer exact-compares the target, performs
the state transition, and inserts the receipt in one transaction before taking
the scheduling snapshot. A crash before this commit means no acceptance. A
lost response after the commit is recovered by the idempotency key.

Invalid identity, stale UID, policy rejection, exhausted budget, unauthorized
target, and incompatible peer fail before acceptance and create no receipt.
Accepted receipts start `ACCEPTED`, then move to `VERIFYING`, `SUCCEEDED`, or
`FAILED`; `VERIFYING` may repeat and may move only to a terminal state. Terminal
receipts are immutable. `SATISFIED` means the requested durable state and runtime
postcondition were observed. `TARGET_ABSENT` is also successful when the exact
runtime disappeared after acceptance. Only a provider's definitive rejection or
an unrecoverable internal invariant can produce a failed receipt.

- `CancelJob` exact-compares `job_uid`, applies the existing cancellation
  transition to the Job, every authority-owned descendant Job, and all active
  Tasks in that subtree, and reuses federation cancel redrive. Acceptance also
  prevents later child submission beneath the cancelled Job. An already terminal
  matching subtree returns a succeeded receipt.
- `RetryTask` requires a nonterminal exact Task, its exact current active
  Attempt, and remaining preemption retry budget. It makes that Attempt
  preempted and the Task retry-eligible through existing retry and coscheduling
  policy. It does not reopen arbitrary terminal failed Tasks or promise that a
  replacement Attempt will be scheduled.
- `TerminateAttempt` exact-compares the current Attempt, makes the Attempt and
  Task terminal with `OPERATOR_TERMINATED`, and disables automatic retry.

Backend teardown occurs only through normal reconcile. A receipt succeeds when
the durable transition is present and the exact old runtime is observed absent.
Provider timeout or source unavailability leaves it `VERIFYING`. A replacement
Attempt, pod, container, or Node is never contacted.

Federated actions retain one authority receipt and one execution receipt keyed
by `(authority_cluster_id, authority_action_id)`. Redelivery is idempotent. An
incompatible peer rejects before authority acceptance; no old RPC fallback is
provided.

## Errors

`lib/iris/src/iris/cluster/resources/errors.py` defines these public errors and
the controller maps them to the listed gRPC status:

| Error | Trigger | gRPC status |
| --- | --- | --- |
| `InvalidResourceKey` | Empty cluster/ID, wrong kind, malformed Attempt number | `INVALID_ARGUMENT` |
| `InvalidPageToken` | Malformed token or token/query/principal mismatch | `INVALID_ARGUMENT` |
| `ResourceNotFound` | Authorized logical or exact target does not exist | `NOT_FOUND` |
| `ResourceReplaced` | Logical ID exists with a different exact UID | `FAILED_PRECONDITION` |
| `BackendIdentityUnknown` | Exact execution backend cannot be resolved | `FAILED_PRECONDITION` |
| `ActionPolicyRejected` | Retry budget, state, or action precondition rejects acceptance | `FAILED_PRECONDITION` |
| `ActionIdempotencyConflict` | Same principal/kind/key with a different payload | `ALREADY_EXISTS` |
| `ResourceSourceUnavailable` | Required exact live source is unavailable | `UNAVAILABLE` |
| `UnsupportedResourceVerb` | Backend does not implement the requested verb | `UNIMPLEMENTED` |
| `UnsupportedResourceSchema` | Database is neither fresh, v2, nor the exact accepted source | startup failure |
| `AmbiguousResourceMigration` | Source rows have conflicting identities or coordinates | startup failure |

List operations do not raise `ResourceSourceUnavailable` for one failed backend;
they return partial items and a failed source status. Mutations and exact live
operations fail because they cannot prove the target or postcondition.

## CLI contract

The final command tree is:

```text
iris job run
iris job list
iris job describe JOB
iris job spec JOB
iris job logs JOB
iris job activity JOB
iris job cancel JOB [--wait]
iris job wait JOB

iris task list [--job JOB] [--state STATE] [--backend BACKEND]
iris task describe TASK
iris task logs TASK
iris task activity TASK
iris task retry TASK [--wait]

iris attempt describe TASK[:ATTEMPT]
iris attempt logs TASK[:ATTEMPT]
iris attempt activity TASK[:ATTEMPT]
iris attempt exec TASK[:ATTEMPT] -- COMMAND...
iris attempt profile TASK[:ATTEMPT] --type TYPE
iris attempt terminate TASK[:ATTEMPT] [--wait]

iris node list [--backend BACKEND] [--health HEALTH]
iris node describe NODE --backend BACKEND

iris slice list [--backend BACKEND] [--scaling-group GROUP]
iris slice describe SLICE --backend BACKEND

iris endpoint list [--prefix PREFIX] [--task TASK]
iris endpoint describe ENDPOINT
iris endpoint mint ENDPOINT [--ttl DURATION]
```

`job run` remains the familiar submit-and-optionally-wait command. `activity`
replaces `task events`. `job stop`, `job kill`, `job kick`, `job summary`, public
Worker commands, and provider-specific Kubernetes status commands are removed.
Backend and peer diagnostic commands remain under `iris cluster` because they
describe controller topology rather than resources.

Endpoint describe and mint accept an endpoint ID or an exact name. Name lookup
must resolve one live registration; multiple matches require the operator to
select an endpoint ID.

`task retry` means “preempt and retry the current active Attempt.” It consumes
the configured preemption retry budget and may leave the Task pending. It rejects
a terminal failed Task; rerunning terminal work is a new Job submission.

Every list supports `--json`, uses the same filter names as its RPC query, and
prints an unavailable-source warning without discarding healthy rows. Action
commands generate an idempotency key when none is supplied, print the action ID,
and return success after durable acceptance unless `--wait` requests convergence.

## Dashboard contract

The controller dashboard has Jobs, Tasks, Nodes, Capacity, Backends, Endpoints,
and Account routes. Tasks is a global inventory. Nodes merges RPC registrations
and Kubernetes observations. Capacity renders Slices where the backend supports
them and keeps Kubernetes NodePool/Kueue information as a Backend detail, not a
Slice.

Job, Task, and Attempt detail pages use the typed resource responses. Their
history component is named `ActivityTimeline.vue` and displays task events plus
action receipts. Job cancel, Task retry, and Attempt terminate buttons show the
exact target and action state. A stale page receives `ResourceReplaced` and must
refresh before presenting another action.

`FleetTab.vue`, `FleetOverview.vue`, `WorkerDetail.vue`, and
`KubernetesClusterDetail.vue` are removed. The worker-daemon's own local
dashboard remains an implementation diagnostic and is not renamed Node.

## Final file ownership

### New files

| Path | Contract |
| --- | --- |
| `lib/iris/src/iris/rpc/resource.proto` | Complete public resource wire surface above |
| `lib/iris/src/iris/cluster/resources/identity.py` | Logical and exact identities |
| `lib/iris/src/iris/cluster/resources/source.py` | Pages, freshness, partial-source status |
| `lib/iris/src/iris/cluster/resources/job.py` | Job query, spec, summary, detail |
| `lib/iris/src/iris/cluster/resources/task.py` | Task query, summary, detail |
| `lib/iris/src/iris/cluster/resources/attempt.py` | Attempt locator, summary, detail, runtime object |
| `lib/iris/src/iris/cluster/resources/node.py` | Node query, locator, identity, summary, and detail records |
| `lib/iris/src/iris/cluster/resources/slice.py` | Slice query, locator, identity, summary, and detail records |
| `lib/iris/src/iris/cluster/resources/endpoint.py` | Endpoint query and records |
| `lib/iris/src/iris/cluster/resources/action.py` | Action enums, requests, receipts |
| `lib/iris/src/iris/cluster/resources/activity.py` | Activity query and entries |
| `lib/iris/src/iris/cluster/resources/log.py` | Exact Job/Task/Attempt log queries and pages |
| `lib/iris/src/iris/cluster/resources/errors.py` | Public typed failures |
| `lib/iris/src/iris/cluster/controller/resources/jobs.py` | Job authorization and response assembly |
| `lib/iris/src/iris/cluster/controller/resources/tasks.py` | Global Task service |
| `lib/iris/src/iris/cluster/controller/resources/attempts.py` | One exact Attempt resolver |
| `lib/iris/src/iris/cluster/controller/resources/nodes.py` | Backend-neutral Node merge |
| `lib/iris/src/iris/cluster/controller/resources/slices.py` | Cached Slice assembly |
| `lib/iris/src/iris/cluster/controller/resources/endpoints.py` | Endpoint resource reads |
| `lib/iris/src/iris/cluster/controller/resources/actions.py` | Action authorization and convergence |
| `lib/iris/src/iris/cluster/controller/resources/activity.py` | Authorized finelog/receipt merge |
| `lib/iris/src/iris/cluster/controller/operator_commands.py` | Bounded pre-acceptance ingress |
| `lib/iris/src/iris/cluster/controller/persistence/schema/base.py` | Metadata and SQLite value types |
| `lib/iris/src/iris/cluster/controller/persistence/schema/workloads.py` | Job, Task, Attempt, Endpoint DDL |
| `lib/iris/src/iris/cluster/controller/persistence/schema/execution.py` | Node, capacity, scaling-group, Slice DDL |
| `lib/iris/src/iris/cluster/controller/persistence/schema/federation.py` | Federation DDL |
| `lib/iris/src/iris/cluster/controller/persistence/schema/operations.py` | Receipts, budgets, rollout DDL |
| `lib/iris/src/iris/cluster/controller/persistence/schema/version.py` | Epoch and accepted fingerprints |
| `lib/iris/src/iris/cluster/controller/persistence/migrate.py` | Fresh creation and exact-source upgrade |
| `lib/iris/src/iris/cluster/controller/persistence/job.py` | Typed Job/spec reads and writes |
| `lib/iris/src/iris/cluster/controller/persistence/task.py` | Typed Task reads and writes |
| `lib/iris/src/iris/cluster/controller/persistence/attempt.py` | Typed Attempt/runtime reads and writes |
| `lib/iris/src/iris/cluster/controller/persistence/endpoint.py` | Typed Endpoint reads and writes |
| `lib/iris/src/iris/cluster/controller/persistence/node.py` | RPC Node registration and reads |
| `lib/iris/src/iris/cluster/controller/persistence/slice.py` | Slice projection reads and writes |
| `lib/iris/src/iris/cluster/controller/persistence/action.py` | Receipt reads, writes, retention |
| `lib/iris/src/iris/cluster/controller/persistence/scheduling.py` | Cross-resource scheduling queries |
| `lib/iris/src/iris/cluster/controller/persistence/control.py` | Reconcile, timeout, prune queries |
| `lib/iris/src/iris/cluster/controller/persistence/federation.py` | Federation persistence |
| `lib/iris/src/iris/cluster/controller/persistence/budget.py` | Budget persistence |
| `lib/iris/src/iris/cluster/controller/persistence/consistency.py` | Final-schema invariants and status |
| `lib/iris/src/iris/cluster/backends/protocol.py` | DB-free TaskBackend, runtime, Node, and Slice contracts |
| `lib/iris/src/iris/cluster/backends/k8s/backend.py` | K8s TaskBackend orchestration and effects |
| `lib/iris/src/iris/cluster/backends/k8s/manifests.py` | Pure K8s object construction |
| `lib/iris/src/iris/cluster/backends/k8s/node.py` | Cached Node inventory |
| `lib/iris/src/iris/cluster/backends/k8s/task.py` | Workload observation normalization |
| `lib/iris/src/iris/cluster/backends/k8s/telemetry.py` | K8s task stats and activity production |
| `lib/iris/src/iris/cluster/backends/k8s/gc.py` | K8s garbage collection |
| `lib/iris/src/iris/cluster/backends/rpc/node.py` | RPC registration to Node projection |
| `lib/iris/src/iris/cluster/backends/rpc/slice.py` | Autoscaler Slice projection |
| `lib/iris/src/iris/cli/node.py` | Node list/describe |
| `lib/iris/src/iris/cli/slice.py` | Slice list/describe |
| `lib/iris/src/iris/cli/endpoint.py` | Singular Endpoint command group |
| `lib/iris/dashboard/src/components/controller/TasksTab.vue` | Global Task inventory |
| `lib/iris/dashboard/src/components/controller/NodesTab.vue` | Unified Node inventory |
| `lib/iris/dashboard/src/components/controller/NodeDetail.vue` | Node detail |
| `lib/iris/dashboard/src/components/shared/ActivityTimeline.vue` | Events and receipt outcomes |
| `lib/iris/tests/journeys/test_resource_views.py` | Global Task/Node/Slice and partial-source journeys |
| `lib/iris/tests/journeys/test_resource_actions.py` | Cancel/retry/terminate restart and replacement journeys |
| `lib/iris/tests/journeys/test_resource_federation.py` | Federated views, outage, redelivery, and convergence journeys |
| `lib/iris/tests/cluster/controller/test_resource_schema_v2.py` | Fresh/upgrade/rejection/rollback schema contract |
| `lib/iris/tests/cluster/controller/test_resource_query_plans.py` | Statement, bind, and semantic plan budgets |
| `lib/iris/tests/cluster/controller/test_resource_service.py` | RPC authorization, paging, partial-source, and exact-target behavior |
| `lib/iris/tests/cluster/controller/test_resource_actions.py` | Receipt constraints, idempotency, retention, and single-writer behavior |
| `lib/iris/tests/cluster/controller/test_resource_imports.py` | Persistence/backend dependency gates and retired symbol scan |
| `lib/iris/tests/cli/test_task_activity.py` | Authorized Activity command and partial finelog behavior |
| `lib/iris/tests/cli/test_attempt.py` | Attempt describe/logs/activity/exec/profile/terminate behavior |

Package `__init__.py` files expose only intentionally public record and protocol
types; they do not mirror every private helper.

### Existing files with semantic changes

| Paths | Final responsibility |
| --- | --- |
| `rpc/controller.proto`, `rpc/job.proto` and generated bindings | Internal controller operations and shared scheduling types; old operator RPCs removed |
| `cluster/controller/service.py` | Thin RPC dispatch into resource and operational services |
| `cluster/controller/controller.py`, `reconcile/*` | Single writer, exact actions, scheduling/reconcile/autoscale tick |
| `cluster/controller/db.py`, `main.py`, `checkpoint.py` | One-database startup, v2 validation, checkpointing |
| `cluster/controller/auth.py` | Resource-parent authorization and action principal checks |
| `cluster/controller/autoscaler/*` | Typed Node/Slice persistence and capacity projections |
| `cluster/controller/federation_proxy.py`, `cluster/federation/*` | Exact resource deltas and action redelivery |
| `cluster/backends/rpc/backend.py` | Cohesive RPC TaskBackend without controller DB access |
| `cluster/composer.py`, `cluster/config.py`, `cluster/local_cluster.py` | Explicit cluster identity and backend construction |
| `scripts/iris/rollout_controllers.py` | Federation-component preflight, quiesce, cutover, and health gating |
| `cluster/client/*`, `client/client.py` | Typed client methods in this spec |
| `cli/job.py`, `cli/task.py`, `cli/attempt.py`, `cli/main.py` | Final noun/verb command tree |
| `lib/iris/tests/cli/test_job.py`, `lib/iris/tests/cli/test_task_describe.py` | Final Job and Task command behavior |
| `dashboard/src/App.vue`, `router.ts`, `types/rpc.ts` | Resource routes and generated wire types |
| `dashboard/src/components/controller/JobDetail.vue`, `TaskDetail.vue`, `CapacityTab.vue`, `BackendsTab.vue`, `EndpointsTab.vue` | Resource responses and actions |
| `lib/iris/README.md`, `lib/iris/OPS.md`, `lib/iris/docs/architecture.md`, `lib/iris/docs/multi_backend.md`, `lib/iris/docs/federation.md` | Operator vocabulary, architecture, migration, and rollback procedure |

### Deleted production files and surfaces

- `cluster/controller/schema.py`, `reads.py`, and `writes.py` after every caller
  moves to a typed persistence module.
- `cluster/controller/backend_store.py` after the controller owns RPC Node
  persistence and passes immutable inputs to the backend.
- `cluster/controller/backend.py` after plain backend contracts move to
  `cluster/backends/protocol.py` and scheduling helpers move under
  `controller/scheduling/`.
- `cluster/backends/k8s/tasks.py` after its responsibilities move to the six K8s
  modules above.
- `cluster/controller/ops/worker.py`; internal registration uses
  `persistence/node.py` and public reads use the Node service.
- `cli/endpoints.py`, replaced by singular `cli/endpoint.py`.
- historical migrations `0027` through `0050`, the source-file migration runner,
  auth database attachment, and auth checkpoint code.
- the volatile `PendingKick` queue and every `KickTasks` caller/RPC.
- public Worker and standalone Kubernetes-cluster status RPCs and dashboard
  components.
- direct CLI finelog SQL for `task events`.
- `lib/iris/tests/cli/test_task_events.py`, replaced by the Activity behavior
  tests above.

## Test contract

Journey coverage is the merge-level behavior proof. Add concise scenarios under
`lib/iris/tests/journeys/` for:

- Job submission, Task 7 failure, exact Attempt inspection, Activity, and Job
  terminal behavior;
- active Task retry, preemption-budget exhaustion, delayed replacement
  scheduling, and replacement safety;
- Job cancellation before scheduling, while running, during federation outage,
  and after controller restart;
- exact Attempt termination for RPC and K8s, including provider timeout and a
  replacement runtime;
- federation to cluster B, B unavailable, partial activity, action redelivery,
  and eventual convergence;
- global Task and Node lists across multiple Jobs and backends with one source
  unavailable;
- RPC Node A retiring and B registering under the same logical ID without
  rewriting historical Attempts;
- Slice membership unknown, known empty, partial registration, and stale source;
  and
- fresh resource-schema startup plus an upgraded database running the same
  submission/restart/checkpoint journey.

Narrow tests remain for properties journeys should not inspect:

- normalized schema fingerprint, exact-source rejection, malformed row
  preflight, transaction rollback, and fresh/upgraded equality;
- SQL statement budgets, SQLite semantic query plans, 32,767-ID federation
  binds, and high-cardinality keyset paging;
- protobuf serialization, client conversion, authorization, idempotency
  conflicts, receipt retention, and federation envelope replay;
- K8s manifest construction/observation and RPC provider protocol contracts;
  and
- import gates proving backends do not import controller persistence and the
  retired schema/query bundles are absent.

Tests assert public responses, persisted rows, provider-boundary calls, and
journey state. They do not assert private helper names, annotations, or call
order. The full safe Iris suite, Pyrefly, changed-files precommit, dashboard
typecheck/build, and stable replay corpus are merge gates.

## Out of scope

- Manual or autoscaled Slice deletion.
- A universal Resource base class, generic CRUD repository, or polymorphic
  resource table.
- Persisting live Kubernetes Node heartbeats in SQLite.
- Replacing the scheduler, Kueue, autoscaler, finelog, or federation transport.
- A second resource-event database or a claim of total ordering across activity
  sources.
- Historical schema support beyond the exact merge-base fingerprint.
- Old-image operation against the resource schema, dual writes, repair-on-open,
  compatibility RPCs, CLI aliases, or telemetry waiting periods.
- Time-series utilization data in Node responses; Node detail links to the
  existing stats views.
- Renaming the RPC worker daemon or its local diagnostic dashboard to Node.
