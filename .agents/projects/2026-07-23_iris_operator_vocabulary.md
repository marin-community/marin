# A resource model for Iris: operator vocabulary and internal structure

> Status: **proposal, for peer review** — written before implementation, against
> the code as of this branch. GitHub issue
> [#7543](https://github.com/marin-community/marin/issues/7543); motivated by the
> `cw-us-east-02a` job-loss incident
> [#7542](https://github.com/marin-community/marin/issues/7542). The code is the
> source of truth; every claim below is cited to `file:line` so a reviewer can
> check it. Nothing here is implemented yet.

## Scope

Two parts, at two horizons.

- **Part I — the operator CLI vocabulary (this PR).** A fixed noun set and verb
  set for `iris`, each verb meaning one thing everywhere. This is what #7543 asks
  for and what the incident needed; it is buildable now against the current RPCs
  with one net-new backend read.
- **Part II — evolving the internals around the same resource model (a plan, not
  this PR).** @rjpower asked to use this as an opportunity to normalize the
  internal package structure and naming. Part II is the grounded plan for that:
  what "resource" should mean inside iris, the one abstraction worth extracting (a
  typed `ResourceRef`), and the bounded, phased changes to the DB schema, the
  module layout, and the docstrings — with an explicit list of what the resource
  framing should *not* be forced onto (the pure scheduling/reconcile kernels, and
  the nouns that have no controller identity).

The idea both parts share: iris manages a small set of resource kinds, and a small
set of verbs operates on them. The CLI is where an operator sees that; the
internals are where it is either honored or contradicted. Today they contradict it
in specific, catalogued ways (Part II).

## Peer review response (codex, 2026-07-23)

The proposal was sent to codex (gpt-5.6-sol) for an adversarial read. Verdict:
**request changes.** The operator-facing direction is right — first-class
task/attempt inspection, backend identity, terminal reasons, structured history,
and `describe` crossing the backend boundary are all necessary — but Part I as
scoped bundles too much, and several verb semantics are not implementable as
written. The accepted changes below **supersede** the corresponding sections; where
a section further down still reads the old way, this response is authoritative.

**Ship a smaller incident-driven slice first.** One accepted design, several
implementation PRs, in this order (this replaces the spiral under "Implementation
plan"):

1. Thin connection resolver + clean-install test (the §8 provider-extras fix).
   *Landed on this branch.*
2. Incident slice: **persist** the backend object reference at create success,
   **capture the terminal backend reason** when an attempt finishes, a dedicated
   `DescribeTask` RPC, and `task`/`attempt describe`. No event system, no renames.
3. Event contract + delivery pipeline — only after durability and security are
   settled (below).
4. Stop semantics + a race-safe asynchronous operation API.
5. CLI vocabulary + module moves, with compatibility messaging.

**`describe` (§4) — persist backend identity as the primary mechanism.** A
deterministic pod name plus one live GET is fragile: pods get
GC'd, names are reused (a k8s name is not an object identity — the pod UID is),
namespaces disappear, federated attempts run on a peer, and the derivation
algorithm can drift across controller versions. Change: persist a backend object
reference (`backend_id, home_cluster, namespace, kind, name, object_uid,
attempt_uid, created_at`) when create succeeds, and record a **bounded terminal
reason** (including the init-container failure) on the attempt decision record when
it finishes, so the answer survives pod deletion. Deterministic derivation becomes
a clearly labeled legacy fallback with an algorithm version. Live backend data is
optional enrichment; `DescribeTask` (a **new** RPC, so live I/O never contaminates
the cheap `GetTaskStatus` poll path) returns the registry snapshot with a deadline
and a typed partial-result status (`not-created / observed-at-T / deleted /
unreachable / permission-denied / unsupported / legacy-reference`). One pod GET
does not answer "why pending" on Kueue in general (workload/admission state,
scheduling conditions, k8s events); describe reports what it can and names what it
could not reach.

**`events` (§5) — a real event contract, audit-vs-telemetry decided before build.**
The `{entity_id, ts, action, trigger, noun, details_json}` row is insufficient.
Carry canonical identity and provenance: `event_id, occurred_at,
recorded_at, home_cluster, noun, resource_id, parent_job_id/parent_task_id,
attempt_uid, action, action_schema_version, source, principal, correlation_id,
details`. Reasons: `entity_id` collides across nouns and clusters; attempt events
keyed by `task_id` are not attempt-indexed; job roll-ups over `entity_id` are N+1;
`ORDER BY ts` is not a deterministic total order across four writers; `noun` cannot
be inferred inside `log_event` without caller changes or a brittle table;
`slice_ready` already bypasses `log_event`, so "single chokepoint captures
everything" is false; unversioned JSON is hard to redact and evolve. The stream is
not obviously low-volume (`scheduling_pass_completed` and pending-reason events can
fire every tick — emit pending reasons only on change, or model them as bounded
current state). Name the namespace by its durability contract
(`iris.audit_event`/`iris.operational_event`); `iris.event` is too generic. Before
building,
decide audit-vs-telemetry and answer: bounded vs unbounded buffering, drop
behavior + metrics, shutdown flush, retries/dedup, finelog outage, ordering, and
at-least-once vs best-effort. A background-thread buffer is not by itself an audit
trail. The read RPC needs cursored pagination, rate limits,
partial-result/source-watermark reporting, pushed-down parent indexes, and must not
run finelog queries on the control thread.

**`stop` (§7) — do not default to a "stop" that restarts.** `task stop
--reschedule` as the default violates the plain meaning of stop. Change: default to
no retry; make retry explicit; and prefer distinct domain verbs over one overloaded
`stop` — `job cancel` (prevent new attempts), `task retry` (end the current
attempt, keep the task eligible), `attempt terminate` (end exactly one current
attempt, conditional on `attempt_uid`), `slice delete` (infra teardown — do not
rename it to a benign `stop`). Attempt targeting is a compare-and-stop: the request
carries the exact `attempt_uid` and the control loop terminates only if it still
matches, so a rollover (attempt 3 ends, attempt 4 starts) cannot hit the wrong
attempt, and the same guard makes retries/duplicates idempotent. The CLI exposes
these as **asynchronous** operations (accepted vs completed, with a way to observe
completion), since all three paths queue onto the single control thread. `job kill`
(the literal alias) is still deleted.

**Ownership model (§II.1) — provenance dimensions over exclusive A/B/C classes.**
The A/B/C cut leaks: worker/slice have controller rows *and* live projections;
attempt is record-backed *and* becomes a live pod; endpoint lacks the hierarchical
`JobName` identity its class implies; actor and cluster are promised verbs they
cannot implement. Change: describe each noun by *dimensions* — registry identity
(controller DB), desired state (home controller), runtime identity (backend),
runtime observation (execution cluster), health (worker tracker), history
(finelog) — and let the service layer report provenance per field. Resolve the
overloaded `cluster` noun (a config scope, a federation peer, a k8s execution
cluster, and the controller process are four different things): `controller`,
`backend`, and `peer` are more defensible nouns than `actor`, which the controller
cannot enumerate. A/B/C stays only as informal shorthand.

**`ResourceRef` (§II.2) — a minimal `ResourceKey`, deferring full subsumption.**
Same string grammar does not mean same semantic type: `/user/job/0` is at once a stable
task identity, a current process, a multi-attempt log source, and a live profiling
target; `/user/job/0:3` is a human ordinal while `attempt_uid` is the execution
identity. `TaskTarget` is a *resolved* runtime handle and must be produced by
resolving a stable ref, not used as one. Change Phase A to introduce only a
canonical `ResourceKey` (home_cluster, kind, id) for event addressing, plus
explicit `AttemptLocator` (task + ordinal) and `AttemptIdentity` (`attempt_uid`),
keeping `ExecutionTarget` and `LogSourceRef` distinct. Share parsing behind
adapters; do not converge the typed `List*` RPCs into a generic `List(kind, query)`
(it weakens typing, authz, pagination, evolution).

**Matrix (§3) — remove cells that cannot be implemented.** actor
`list/describe/events` needs controller identity/discovery it lacks; attempt
`profile` is impossible on a completed attempt and race-prone live; `task spec`
returning the parent job's spec is not a task spec; `cluster spec` as config does
not meet "sufficient to resubmit." `spec` output is redacted, so it is a
**sanitized submitted configuration**, not a guaranteed re-runnable spec. Pin the
universal-verb contracts before the CLI move: which attempt `task logs` selects,
whether `job logs` aggregates, `--follow` across a retry, the canonical `list
--state` values per state machine, list pagination/snapshot consistency, and
whether `attempt stop` rejects historical attempts.

**New cross-cutting concerns (were missing).**
- *Federation:* every ref/event carries a home cluster and, where different, an
  execution cluster; `describe`/`events`/`stop` proxy to the authoritative peer or
  return a structured partial/redirect — never silent local best-effort.
- *Security/authorization:* per-resource tenant authz before any DB/finelog/backend
  read; authz for list/roll-up/prefix queries; redaction allowlists for event
  details and k8s messages; principal identity on stop/audit events; separation of
  user-facing vs system worker/cluster events; sanitize terminal control characters
  in rendered backend messages. `**details`-to-JSON must not retain tokens/URLs.
- *Versioning:* client/controller skew is real even under "no aliases" — new proto
  fields/RPCs need a deploy order and capability detection; removed commands should
  fail with a precise replacement message.
- *Performance/isolation:* no per-task live GET for `job describe` unless asked; no
  event queries on the control thread; deadlines + backend rate limiting on live
  describes; event buffer/health metrics visible without the event system itself.

**Part II schema (§II.4) — soften.** Uniform CLI presentation does not need uniform
physical encoding: prefer typed domain adapters over an int→string state migration
(migrate one table only when another change requires it). Name the three event
roles in docs; do not physically rename established finelog namespaces. Decide
free-floating IDs per coordinate by ownership/integrity need, not for relational
aesthetics (a thin anchor table with no lifecycle is a second stale truth).
slice↔worker: measure contradictions in real DBs and document membership lifetime
before dropping either side; if history matters, a junction/history table may be
right. The service-file split, docstring sweep, private-helper rename, and proto
renames are unrelated cleanup and should not ride inside the resource-model program.

The sections below are the original proposal, retained for context and reference.

# Part I — The operator CLI vocabulary (this PR)

## Problem

Debugging the `cw-us-east-02a` job loss (#7542) took about an hour, and almost
none of it happened in the `iris` CLI. The CLI could report that 15 jobs had
failed. It could not say *why* any single one failed, or *what backend object*
it owned. The real cause — `Bundle fetch <id> failed: HTTP Error 404` in a pod's
`stage-workdir` init container — was reachable only by: reading `_pod_name` in
`backends/k8s/tasks.py` to derive the pod name by hand, pulling a kubeconfig, and
running `kubectl logs <pod> -c stage-workdir`. The accounting question "which
attempts failed, when, with what error" was answered with raw SQL over
`task_attempts`.

Three structural gaps produced that hour:

1. **No command shows a single object.** `cluster status`, `process status`, and
   `job summary` are three different words for *describe* at three altitudes, and
   two of the most important nouns have no describe at all: `task` has exactly one
   subcommand (`exec`), and `attempt` is not addressable.

2. **Audit history is grepped, never queried.** Events are structured where they are
   written — `log_event(action, entity_id, trigger=..., **details)`
   (`controller/audit_logging.py:15`) — and then flattened to a text log line and
   shipped to finelog. Retrieval is `iris process logs --substring=<id>`
   (`OPS.md`; `cli/process_status.py:86`), a substring match over opaque text with
   no entity index and no time-range-by-entity query.

3. **`describe` stops at the iris object.** Nothing maps a task attempt to its
   backend pod (name, node, phase, init-container status). This incident was
   precisely an attempt→pod mismatch, and the vocabulary that would have shortcut
   it does not exist.

A fourth gap blocks the CLI from even connecting: `iris --cluster=cw-us-east-02a
cluster status` fails with `ImportError: Install iris[controller] to use
CloudK8sService`, because resolving a controller URL eagerly constructs the full
provider bundle (`cli/connect.py:310`). The incident was worked around with
`kubectl port-forward` and `--controller-url`.

### What exists today, by noun and verb

The filled cells disagree with each other — `summary`/`status` are one verb under
two names, `stop`/`kill` are two names for one verb, and per-task inspection is
split across `job`, `task`, and `process`:

| noun | list | describe | events | logs | exec | profile | stop |
|---|---|---|---|---|---|---|---|
| cluster | `cluster list` | `cluster status` | — | — | — | — | `cluster stop` |
| slice | `cluster vm status` | — | — | — | — | — | `cluster delete-slice` |
| worker | `cluster vm status` | `process status -t` | — | `process logs -t` | — | `process profile -t` | — |
| job | `job list` | `job summary` | — | `job logs` | — | — | `job stop` / `kill` |
| task | — | — | — | — | `task exec` | `process profile -t` | `job kick` |
| attempt | — | — | — | — | — | — | — |
| endpoint | `endpoints list` | — | — | — | — | — | — |
| actor | — | — | — | — | `actor call` | — | — |

## Goals

From the issue's "Done when," restated as the acceptance criteria this proposal
must satisfy:

1. A fixed noun set and a fixed verb set are **written down in `lib/iris/OPS.md`**,
   each verb meaning the same thing for every noun.
2. `describe` and `events` **exist for job, task, and attempt** and report the
   **terminal reason and backend identity** (for a task/attempt: the backing pod
   name, node, phase, and — the incident's blind spot — init-container status).
3. `stop` and `kill` (and the third synonym, `kick`) **collapse to one verb**.
4. The operator CLI **resolves a controller URL and runs read-only commands
   without provider extras installed** (a plain install against a Kubernetes
   cluster).

Plus the maintainer's directive on the issue (@rjpower): *"normalize everything,
use this as an opportunity to clean up the module/package structure as
appropriate to follow best practices."* So the CLI package layout is in scope
too, alongside the command names.

## Principles

- **kubectl is the reference.** `get`, `describe`, `logs`, `exec`, `events`,
  `delete` work across every resource kind, so each new kind is learnable for
  free and `describe` is always the first command you reach for. We adopt the same
  discipline: one verb, one meaning, applied wherever the (noun, verb) pair is
  meaningful.
- **`describe` crosses the backend boundary.** An iris task is a Kubernetes pod or
  a worker process. If `describe` stops at the iris row, operators keep falling
  through to `kubectl` during exactly the incidents the vocabulary is meant to
  serve. `describe` must name and read the backend object.
- **`events` is a queryable resource.** Today it is grep over log lines. The write sites are already
  structured (`log_event`, a single chokepoint). Events become first-class data
  keyed by entity id, so `events` can exist for *every* noun instead of only where
  someone remembered to grep.
- **The CLI works from a plain install.** Read-only inspection must not require
  `iris[controller]`. Provider machinery is constructed only by commands that
  provision or tear down infrastructure.
- **No backward compatibility** (repo policy). Renamed verbs are renamed in place;
  all in-repo call sites (`OPS.md`, skills, docs) are updated in the same change.
  No aliases, no deprecation shims.

## Design

### 1. The noun set

`cluster`, `slice`, `worker`, `job`, `task`, `attempt`, `endpoint`, `actor`.

Two current command groups **dissolve** into this set:

- **`process` disappears.** It is an altitude selector (`--target`), not a noun.
  Its three targets map onto real nouns: `process ... -t /system/worker/<id>` →
  `worker`; `process ... -t /user/job/0` (task container) → `task`; and the
  default (no target = the controller) → `cluster` (the controller is the
  cluster's control-plane head; see §9). So `process status/logs/profile` become
  `{worker,task,cluster} {describe,logs,profile}`.
- **`slice` is promoted** out of `cluster vm`/`cluster {create,delete}-slice` into
  a first-class noun.

Nouns are **singular** (`job`, `task`, `worker`, …); today's plural `endpoints`
group is renamed to `endpoint` for consistency.

Out of scope as nouns (kept as-is): `iris query` (raw admin SELECT — an escape
hatch we want to *depend on less*, not remove), `iris rpc` (generic RPC
passthrough), `iris build` (dev/image builds), `iris login` / `iris user budget`
(auth/admin). These are power tools and lifecycle actions, not the inspection
vocabulary this proposal normalizes.

### 2. The verb set

Eight verbs. Each means the same thing for every noun; a noun implements a verb
only where it is meaningful (§3).

| verb | meaning |
|---|---|
| `list` | Enumerate objects of this kind, with consistent filters (`--state`, `--prefix`, `--limit`). |
| `describe` | **One object**: current state, *why* it is in that state, its parent and children, and its **backend identity** (the pod/worker it runs as, incl. node/phase/init-container status). The first command you reach for. |
| `spec` | The definition the object was submitted with, sufficient to resubmit. |
| `events` | The ordered, structured, queryable history of *this object*, keyed by its id. |
| `logs` | Log lines for the object (unchanged in meaning). |
| `exec` | Run a command in the object's container (unchanged). |
| `profile` | Capture a CPU/memory/thread profile of the object (unchanged). |
| `stop` | Terminate the object. **One name** — replaces `stop`, `kill`, and `kick`. |

Noun-specific **lifecycle** verbs that do not generalize stay, clearly separated
from the universal inspection verbs above: `cluster {start,restart,dashboard}`,
`cluster init-keys`, `cluster controller {serve,checkpoint,restart}`, `slice
create`, `job run` (submit), `endpoint mint`, `actor call`. These are actions on a
noun, not one of the eight cross-cutting verbs, and the proposal leaves their
names alone except where they collide with a universal verb (`cluster stop` →
still `stop`; `delete-slice` → `slice stop`).

### 3. The target matrix (the normalization)

Target state. `✓` = implemented and uniform; `L` = lifecycle/other verb covers it;
`—` = not meaningful.

| noun | list | describe | spec | events | logs | exec | profile | stop |
|---|---|---|---|---|---|---|---|---|
| cluster | ✓ | ✓ | ✓ (config) | ✓ | ✓ (controller) | — | ✓ (controller) | ✓ |
| slice | ✓ | ✓ | — | ✓ | — | — | — | ✓ |
| worker | ✓ | ✓ | — | ✓ | ✓ | — | ✓ | L (drain) |
| job | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ |
| task | ✓ | ✓ | ✓ (of its job) | ✓ | ✓ | ✓ | ✓ | ✓ |
| attempt | ✓ | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| endpoint | ✓ | ✓ | — | ✓ | — | — | — | L (unregister) |
| actor | ✓ | ✓ | — | ✓ | — | ✓ (call) | — | — |

**Old → new command mapping** (the concrete rename list):

| today | becomes |
|---|---|
| `cluster status` | `cluster describe` |
| `cluster stop` | `cluster stop` (unchanged) |
| `cluster vm status` | `slice list` (+ `worker list` for the per-worker view) |
| `cluster create-slice` | `slice create` |
| `cluster delete-slice` | `slice stop` |
| `job summary` | `job describe` |
| `job stop` / `job kill` | `job stop` (drop `kill`) |
| `job kick /u/j/0` | `task stop /u/j/0` |
| `job kick /u/j/0:3` | `attempt stop /u/j/0:3` |
| `job kick /u/j` (whole job) | `job stop --tasks /u/j` (or `task stop` per id) |
| `process status` (controller) | `cluster describe` (control-plane view) + controller ProcessInfo |
| `process status -t /system/worker/<id>` | `worker describe <id>` |
| `process logs` (controller) | `cluster logs` |
| `process logs -t /system/worker/<id>` | `worker logs <id>` |
| `process profile [-t <worker>]` | `cluster profile` / `worker profile <id>` |
| `process profile -t /user/job/0` | `task profile /user/job/0` |
| `task exec /u/j/0` | `task exec /u/j/0` (unchanged) |
| `endpoints list` | `endpoint list` |
| `endpoints mint` | `endpoint mint` |

New commands with no predecessor (the gaps the incident hit): `task describe`,
`task list`, `task events`, `task logs`, `attempt describe`, `attempt events`,
`attempt logs`, `attempt profile`, `job events`, `{cluster,slice,worker,endpoint}
events`, `slice describe`.

The rest of the design settles how the four load-bearing verbs are built.
`describe` and `events` are the two that would have mattered in #7542.

### 4. `describe` — composition, and crossing the backend boundary

`describe` is mostly *composition of RPCs that already exist*, plus one net-new
backend read.

**What is already there** (no server work):

- `job describe` = `GetJobStatus` (returns `JobStatus` + the reconstructed
  original `LaunchJobRequest`, `service.py:1807`) + `ListTasks` (each `TaskStatus`
  carries its full `attempts` list, `service.py:2145`). Parent/children:
  `JobStatus.parent_job_id` (33), `has_children` (32).
- `task describe` = `GetTaskStatus` (`service.py:2073`), which already returns
  `TaskStatus task` **with the attempt chain** (`TaskStatus.attempts`,
  `job.proto:187`), `root_cause_highlights` (distilled failure log lines), and
  `job_resources`. `TaskStatus.container_id` (19) is documented as "Docker
  container ID, **K8s pod name**" — so the current attempt's pod name is *already*
  on the wire; today's `job summary` just never prints it.
- `attempt describe` addressing: the attempt chain is in `TaskStatus.attempts`
  (`TaskAttempt`: `attempt_id`, `worker_id`, `state`, `exit_code`, `error`,
  `started_at`, `finished_at`, `is_worker_failure`, `attempt_uid`,
  `job.proto:249`).

**What is net-new** — and it is exactly the incident's blind spot. `TaskAttempt`
carries `worker_id` but **no pod name, node, or phase**; only the *current*
attempt's pod name is stored (`tasks.container_id`). And init-container status is
**never observed for task pods** — the only read of `initContainerStatuses` in the
k8s subtree is for the controller's own pod (`platforms/k8s/controller.py:867`),
never in the task observation path (`_task_container_status`,
`backends/k8s/tasks.py:964`; `_poll_pods`, `:2890`). So a pod wedged in
`Init:Error` on a `stage-workdir` 404 surfaces only as a generic `Pending` phase,
and the task-container extractors return nothing.

Two facts make this cheap to close:

1. **The pod name is derivable offline.** `_pod_name(task_id, attempt_id,
   attempt_uid)` (`backends/k8s/tasks.py:348`) is pure and deterministic —
   `sha256` + string munging, no live state. All three inputs live in
   `task_attempts` (`attempt_uid` is controller-minted, `writes.py:530`, read via
   `attempt_uid_for`, `reads.py:1142`). So `describe` can name the pod for *any*
   attempt in the chain — including a past, failed one — with no kubectl.
2. **Node/phase/init-container status need exactly one live pod GET.** These are
   scheduler-assigned and only in live state (`pod.spec.nodeName`,
   `pod.status.phase`, `pod.status.initContainerStatuses`). The k8s backend
   already fetches the managed-pod list every reconcile (`sync`, `:2307`); reading
   init-container state is a sibling of the existing `_container_state_reason`
   (`:1254`) — iterate `initContainerStatuses` alongside `containerStatuses`.

**Proposed mechanism.** Add one per-target backend method to the `TaskBackend`
contract (`controller/backend.py:464`), alongside the existing `TaskTarget`
family (`get_process_status`/`profile_task`/`exec_in_container`, which already
take a `TaskTarget` carrying `task_id, attempt_id, worker_id, address,
attempt_uid` — `backend.py:153`):

```
describe_task(target: TaskTarget) -> TaskBackendDetail
```

- **K8s** (`K8sTaskProvider`): compute `_pod_name(target...)`, GET that one pod,
  and return `{pod_name, node_name, phase, container_states, init_container_states}`
  — the last parsed from `initContainerStatuses` (net-new, ~20 lines mirroring
  `_container_state_reason`). Target resolution is already built:
  `service.py:_resolve_task_target` (`:2473`) constructs the `CLUSTER_VIEW`
  `TaskTarget` with `attempt_uid` for exactly this case.
- **RPC backend** (`RpcTaskBackend`): backend identity is `worker_id` + `address`
  (host); `pid`/`hostname` come from the existing `get_process_status` forward to
  the worker (`backends/rpc/backend.py:454`). Return
  `{worker_id, address, pid, hostname}`.

The controller wraps this in a new `DescribeTask` RPC (or extends
`GetTaskStatusResponse` with an optional `backend_detail`). The CLI's `task
describe` and `attempt describe` render: state → terminal reason → attempt chain
(each attempt's pod name computed offline) → for the current/selected attempt, the
live backend detail. `attempt describe /u/j/0:3` prints the pod that attempt 3
owned, its node, and — if it died in init — the `stage-workdir` container's
`reason`/`message`. That single command replaces the source-reading, the SQL, and
most of the kubectl from #7542.

### 5. `events` — a queryable `iris.event` resource

Today the only record of a `log_event` call is a flattened text line on the
`/system/controller` log source; the `event=… entity=… trigger=… k=v` structure
is discarded by the log formatter at write time (`controller.py:428`), and
recovered only by substring-matching (`process logs --substring`).

**Two facts decide the design:**

1. **`log_event` is a single chokepoint** (`audit_logging.py:15`). Every direct
   caller (~13 sites: `ops/worker.py:103`, `ops/task.py:112`, `ops/job.py:124`,
   `pruner.py:71`, `backend_store.py:203`, …) *and* the reconcile kernel's buffered
   `LogEvent`s (drained through `reconcile/commit.py:242`) funnel through it. A
   dual-write added **inside `log_event`** captures every site with **zero
   call-site changes**.
2. **The pattern already exists in finelog.** `iris.task_event` / `TaskEventRow`
   (`stats/tables.py:179`) is described verbatim as *"the event log for every
   job"* — one row per admission verdict, keyed by `task_id`, written by
   `TaskEventLog` (`backends/k8s/tasks.py:1899`). And the repo's stated
   architecture rule (`AGENTS.md`, "Decisions vs measurements") puts append-only
   time-series streams in **finelog namespaces, not the controller SQLite DB**.

**Proposal: a new finelog stats namespace `iris.event`**, keyed by `entity_id`.

- Add an `IrisEvent` row dataclass to `stats/tables.py`:
  `{entity_id (key_column), ts, action, trigger, noun, details_json}`. The
  heterogeneous `**details` kwargs collapse into one JSON text column (the one
  schema wrinkle vs. the typed namespaces).
- Register it in `LogStack` (`controller/log_stack.py:108`) and thread the `Table`
  into `audit_logging`; add one non-blocking `table.write([row])` in `log_event`
  (`Table.write` buffers to a background flush thread — `log_client.py:259` — so it
  never touches the control hot path). This mirrors how `iris.task_state` is
  produced controller-side (`task_state_stats.py:116`).
- Set a **dedicated `StoragePolicy` with forensic retention** (e.g. 30 days). This
  is the one place we deviate from an existing policy on purpose:
  `iris.task_event` is capped ~1h/100MiB (`tables.py:61`), far too short for
  post-incident debugging. `iris.event` is low-volume, high-value.

`key_column="entity_id"` gives parquet row-group pruning for the exact access
pattern the verb needs: `SELECT ts, action, trigger, details_json FROM
"iris.event" WHERE entity_id = '<id>' ORDER BY ts DESC`.

**The `events` verb presents a unified, ordered timeline** per object, drawn from
the sources that apply to that noun:

- `task events` / `attempt events`: attempt state-transitions (authoritative, from
  `task_attempts` via `TaskStatus.attempts`) + audit events (`iris.event`, keyed by
  `task_id`) + k8s admission events (`iris.task_event`).
- `job events`: job audit events + a roll-up of its tasks' events.
- `worker events`: worker audit events (`worker_registered`, `worker_failing`,
  `worker_pruned`, `worker_failed`, `reconcile_rpc_failed`) + provisioning
  (`iris.provisioning`).
- `slice events`: `slice_pruned` (+ `slice_ready`, which bypasses `log_event` via
  the autoscaler's `_log_action`, `autoscaler/runtime.py:359`, and already writes
  structured `IrisProvisioning` rows — so it is queryable without change).
- `cluster events`: the singleton-entity actions
  (`scheduling_pass_completed`, `checkpoint_written`, `dispatch_updates_applied`).

A new controller RPC (`GetEvents(entity_id, [since], [limit])`) fronts the finelog
query so the CLI needs neither a finelog tunnel nor schema knowledge — the same
reason `describe`/`spec` go through the controller, so operators never have to
learn internal tables.

### 6. `spec` — reuse the reconstruction that already exists

The issue's premise that `jobs.request_proto` is a serialized blob is **stale**:
there is no `request_proto` column. The submission spec is stored *decomposed* in
the `job_config` table (`schema.py:285`) and rebuilt into a complete
`LaunchJobRequest` by `reconstruct_launch_job_request(job, *, workdir_files)`
(`controller/codec.py:153`). `GetJobStatus` **already returns it** as
`GetJobStatusResponse.request` (`service.py:1880`, redacted via
`redact_request_env_vars`). The federation handoff path already round-trips it to
a re-runnable job (`federation_store.py:69`).

So `job spec <id>` prints `GetJobStatusResponse.request` (as YAML/JSON), and `task
spec` prints the same for its parent job — **no new RPC, no new storage**. A
future `resubmit` verb is a natural follow-on (reconstruct + `LaunchJob`, mirroring
`federation_store`) but is out of scope here.

### 7. `stop` — collapse three names into one

`stop`, `kill`, and `kick` are the same verb at different scopes:

- `job kill` is a literal alias of `job stop` (`cli/job.py:1155`, both call
  `_stop_jobs`). **Delete `kill`.**
- `job kick` (`cli/job.py:1176`) forces a *task attempt* terminal via
  `kick_tasks` (`service.py:2174`), with `--state preempted|failed`. This is
  `stop` at the task/attempt scope. **Rename to `task stop` / `attempt stop`**,
  with the retry semantics as a flag: `--reschedule` (default; `preempted`, retries
  if budget remains) vs `--fail` (terminal, no retry). The stdin/`--dry-run`
  query→act bridge (`OPS.md` "Bulk actions") is preserved on the new spelling.

Result: `stop` means "terminate this object" at every level — `cluster stop`,
`slice stop`, `job stop`, `task stop`, `attempt stop` — with the task/attempt
variant carrying the reschedule choice. `worker`/`endpoint` keep their existing
lifecycle spellings (`drain`, `unregister`) since "terminate a worker" is a
different, heavier operation than terminating a workload (noted `L` in §3).

### 8. Read-only CLI without provider extras

The `ImportError: Install iris[controller]` is incidental, not essential. Tracing
the failure: `cli/connect.py:_resolve_controller_url` calls `provider_bundle(config)`
(`:310`) for a non-IAP cluster → `factory.create_provider_bundle` constructs
`K8sControllerProvider` (`factory.py:100`) → whose `__init__` **eagerly**
constructs `CloudK8sService(...)` (`platforms/k8s/controller.py:345`) → whose
`__post_init__` raises unless the optional `kubernetes` python client is installed
(`platforms/k8s/service.py:199`).

But URL resolution only needs two things, **neither of which uses the `kubernetes`
python client**:

- the controller address — `config.controller_address()` (pure config read) or
  `discover_controller` (pure string formatting, `controller.py:373`); and
- the tunnel — `tunnel` → `port_forward`, which **shells out to the `kubectl`
  binary** via `subprocess.Popen` (`service.py:752`).

The `kubernetes` package is required only by the `DynamicClient`-backed CRUD/exec
methods, which read-only URL resolution never calls.

**Fix: make `CloudK8sService`'s `DynamicClient` construction lazy.** Build the
`kubernetes`-backed client on first CRUD use, not in `__post_init__`, so
`discover_controller` and `port_forward` (kubectl subprocess) work with only the
`kubectl` binary present. A read-only `cluster describe` / `task describe` then
resolves the URL and tunnels on a plain `iris` install. (Alternative considered: a
kubectl-subprocess-only `K8sService` variant used for the tunnel path; the lazy
approach is smaller and keeps one class.) This is independently landable and has
no dependency on the vocabulary work — a good first stage.

### 9. Module / package restructure

Today `cli/` is two grab-bag megafiles — `cluster.py` (1615 lines: cluster
lifecycle + `vm` + `controller` + `log-server`) and `job.py` (1501 lines: submit +
list/stop/kill/kick/summary/logs + all the `ResourceSpec`/constraint building) —
plus thin per-concern files. The verbs for one noun are scattered across files
(`task exec` in `task.py`, `task`'s profile in `process_status.py`, `task`'s stop
in `job.py`).

**Target: one module per noun, a shared verb/render layer.**

```
cli/
  main.py            # the iris group, global options, subcommand registration
  connect.py         # controller URL resolution, clients (unchanged leaf)
  render.py          # table/state/duration/memory formatting (from proto_display + the
                     #   _render_* helpers now inline in job.py/process_status.py)
  targets.py         # id + stdin parsing, the query→act bridge (_collect_targets, dry-run)
  cluster.py         # cluster: list/describe/events/logs/profile/stop + start/restart/dashboard
  controller.py      # cluster controller {serve,checkpoint,restart}, log-server  (lifecycle)
  slice.py           # slice: list/describe/events/create/stop
  worker.py          # worker: list/describe/events/logs/profile + drain
  job.py             # job: list/describe/spec/events/logs/stop   (inspection only)
  submit.py          # job run + ResourceSpec/constraint/topology building (the ~1000 lines out of job.py)
  task.py            # task: list/describe/spec/events/logs/exec/profile/stop
  attempt.py         # attempt: describe/events/logs/profile/stop
  endpoint.py        # endpoint: list/describe/events/mint
  actor.py           # actor: list/describe/events/call
  query.py rpc.py build.py user.py   # power tools / lifecycle, unchanged
```

Rationale: each noun's file *is* the answer to "what can I do to a task?"; the
universal verbs share `render.py`/`targets.py` so `describe`/`events`/`list`/`stop`
render and parse identically everywhere (the uniformity is enforced by shared code,
not convention). `process_status.py` is deleted — its verbs move onto `cluster`,
`worker`, and `task`. The submit machinery leaves `job.py` so the inspection verbs
are readable. This is a mechanical move-and-split; no behavior changes beyond the
renames in §3.

## Migration (no backward compatibility)

Renames land with their call sites, in the same change:

- `lib/iris/OPS.md` (the primary surface — `cluster status`, `process *`, `job
  summary`, `job kick`/`kill`, `cluster vm status`, the SQL-for-attempt-history
  and `process logs --substring` sections all get rewritten to the new verbs;
  `iris query` stays but the doc points at `events`/`describe` first).
- Skills that call the old commands: `debug`, `triage-canary`,
  `recover-stuck-k8s-pod`, `babysit-job`, `restart-iris` (grep: ~7 skill files).
- `lib/iris/docs/*` live docs (`federation.md`).
- `iris.cluster` OPS references in `lib/zephyr/OPS.md` (shared-infra commands).

Historical `.agents/ops/*` postmortems are **not** rewritten — they are dated
records of what was run at the time.

## Implementation plan (spiral — each stage independently testable)

Ordered so the two highest-value, lowest-risk pieces land first and the vocabulary
is usable incrementally.

1. **Provider-extras fix (§8).** Lazy `CloudK8sService`. Test: read-only
   `cluster describe` against a k8s cluster from an install without
   `iris[controller]`; unit test that `discover_controller`/`tunnel` don't
   construct the `DynamicClient`. *No vocabulary dependency — land first.*
2. **`stop` collapse (§7).** Delete `job kill`; rename `job kick` → `task stop` /
   `attempt stop` with `--reschedule/--fail`; preserve stdin/`--dry-run`. Update
   `OPS.md`.
3. **`describe` backend crossing (§4).** `describe_task` on the `TaskBackend`
   contract; k8s pod GET + init-container parsing; RPC backend worker identity; the
   `DescribeTask` RPC; `task describe` / `attempt describe` render the pod
   name/node/phase/init-status and the attempt chain. This is the core incident
   fix — build and test it end to end on a real failed-in-init pod.
4. **`iris.event` namespace + `events` verb (§5).** `IrisEvent` row, `LogStack`
   wiring, the one dual-write in `log_event`, forensic `StoragePolicy`,
   `GetEvents` RPC, `{job,task,attempt,worker,slice,cluster} events`.
5. **Fill the matrix (§3) + package restructure (§9).** The remaining renames
   (`cluster status`→`describe`, `job summary`→`describe`, `slice`/`worker`/
   `endpoint` promotion, `process` dissolution) and the noun-per-module split with
   shared `render.py`/`targets.py`.
6. **`spec` verb (§6).** `job spec` / `task spec` over `GetJobStatus.request`.
7. **Write the vocabulary into `OPS.md`** as the canonical reference (Goal #1) and
   sweep the skill/doc call sites (Migration).

Stages 1–2 are small and land immediately. Stage 3 is the incident fix. Stages
4–7 complete the normalization. Each stage keeps the CLI shippable.

## Testing

- Unit: `_pod_name` round-trip is already deterministic; add a test that
  `describe`'s offline pod-name computation matches what the k8s backend applies.
  Init-container parsing gets a table-driven test over synthetic pod dicts
  (`Init:Error`, `ImagePullBackOff`, running, completed).
- Unit: `log_event` dual-write emits an `IrisEvent` row for a representative
  action from each noun; the render layer formats an attempt chain / event timeline
  from fixture protos with no cluster (mirroring `build_job_summary`, which is
  already a pure, unit-tested function — `cli/job.py:1331`).
- Unit: provider-extras — `_resolve_controller_url` for a k8s config resolves an
  address and builds a tunnel without importing/constructing the `kubernetes`
  client.
- Integration (existing k8s smoke, CoreWeave CI): `task describe` on a pod that
  failed in `stage-workdir` prints the init-container reason; `events` returns the
  ordered history for a preempted task.

# Part II — Evolving the internals around the resource model (plan)

Part I normalizes what the operator types. This part asks whether the code behind
those commands is organized around the same resources, and where it should move if
not. It is a plan for follow-on work; the CLI in Part I ships first and depends on
none of it. The grounding is a four-seam sweep of the current code — the persistent
stores (`cluster/controller/schema.py`, `cluster/stats/tables.py`), the scheduler
and reconcile kernels (`cluster/controller/scheduling/`, `.../reconcile/`), the
per-noun type layer (protos, dataclasses, RPCs), and the package layout. Every
claim is cited so a reviewer can check it against the code.

## II.1 — What a "resource" is in iris, and what it is not

The eight nouns do not share a lifecycle. They fall into three ownership classes,
and the class decides which verbs are real and where the code for them lives.

- **Class A — controller-owned records with a strong hierarchical id: job, task,
  attempt, endpoint.** Each is a row the controller writes (`jobs`/`tasks`/
  `task_attempts`/`endpoints`, `schema.py:243,327,423,498`) with a proto view and,
  for job/task/attempt, a `JobName`-based id that already encodes parent/child
  (`cluster/types.py:132`). `list`/`describe`/`spec`/`events`/`stop` all mean
  something concrete here. This is where a resource abstraction earns its keep.
- **Class B — live infra projections: worker, slice, cluster-as-K8s.** The
  authoritative state is not a controller record; it is a `RemoteWorkerHandle`/
  `SliceHandle` (`cluster/platforms/types.py:220,324`) or a kubectl/autoscaler
  snapshot. `list`/`describe` are read projections; `stop` is infra teardown routed
  through the platform layer, not a controller state write; `spec` does not exist
  (they are provisioned from a `ScaleGroup`, not submitted). A uniform model must
  tolerate a member with no submit spec and, for the K8s backend, no controller
  worker rows at all (`health = None`, `backends/k8s/tasks.py:2188`).
- **Class C — non-records: cluster-identity, actor.** "cluster" is a config `name`
  plus three unrelated wire types (`PeerSummary`/`BackendSummary`/
  `KubernetesClusterStatus`); it has no single object. "actor" has no controller
  identity — the controller sees an `Endpoint` row, and `RegisteredActor` exists
  only inside the task process (`actor/server.py:45`). A Resource type here would
  invent identity that is not there.

The recommendation that follows: do not introduce a single `Resource` base class
over all eight nouns. Model Class A as resources; keep Class B as a fleet/infra
read surface; leave Class C as scope (cluster) and as a sub-view of an endpoint
(actor). The uniformity the operator sees in Part I comes from the shared verb
layer in the CLI, not from a forced common supertype in the domain model.

## II.2 — The one abstraction worth extracting: a typed `ResourceRef`

There is no shared `Resource`/`Entity`/`Addressable` type today. The nearest thing
that already works is a string path grammar, used in three places that do not know
about each other:

- profiling / process-status `target` — `/user/job/0`, `/user/job/0:3`,
  `/system/worker/<id>`, `/system/controller` (`rpc/job.proto:82`; dispatch
  `service.py:2615`);
- finelog log `source`/`key` — the identical grammar (`cluster/log_keys.py:61,67`);
- `JobName`/`TaskAttempt`, which already parse and format the job/task/attempt
  portion with ancestry helpers (`cluster/types.py:132,336`).

So job, task, attempt, the controller process, and a worker process are already
addressable by one path string; slice, endpoint, actor, and cluster are not yet in
the grammar. Extracting a typed `ResourceRef` — one parser/formatter that produces
a tagged reference and subsumes `target`, `source`, and `JobName`/`TaskAttempt` —
is the single highest-leverage internal change. It is what lets `describe(ref)`,
`events(ref)`, and `stop(ref)` dispatch by ref-kind in one place; today each RPC
re-parses the string itself. It should reuse the existing `TaskTarget`
(`controller/backend.py:153`) for the attempt case, so no parallel addressing
scheme appears. It is also the natural point to give slice and attempt the
first-class `list` they lack, and to converge the six near-identical `*Query`
messages (`ListJobs/Tasks/Workers/Endpoints/Backends/Peers`, `service.py:2023,
2145,2330,2429,3220,3344`) toward one `List(kind, query)` shape over time.

## II.3 — The resource surface lives at the service layer, above the pure kernels

Two kernels are pure by construction and must stay that way. The scheduling kernel
(`scheduling/scheduler.py:696`, "does not dispatch tasks, modify state, or run
threads", `:678`) and the reconcile kernel (`controller/reconcile/`, "a pure state
machine over a closed snapshot, plus thin I/O") have zero `db`/`schema` imports,
enforced structurally and by the AGENTS.md dependency rule. A resource surface
"backed by the scheduler" must not reach into them; it belongs at the
controller/service layer, where the DB, the backends, and finelog already meet.

Two consequences for the resource verbs:

- **`stop` is not a synchronous mutating call.** It fans into three
  control-loop-mediated paths: job stop is a cascade-kill through the reconcile
  kernel (`ReconcileState.cancel_job`, `reconcile/batches.py:273`); task/attempt
  stop is a queued kick resolved in-tick (`request_task_kicks` →
  `_resolve_pending_kicks`, `controller.py:576,1576`); worker stop is a queued
  eviction plus slice/sibling teardown routed to the owning backend
  (`request_worker_eviction`, `controller.py:562`). All three stay on the single
  control thread. A uniform `stop` verb wraps these; it does not replace the
  queue-onto-the-tick discipline.
- **`describe`/`events` legitimately cross stores.** A task's full description
  spans the DB registry (`tasks`/`task_attempts`), finelog measurements
  (`iris.task`/`iris.profile`), and the live backend object (a pod GET or a worker
  RPC). This is the decisions-vs-measurements split (`AGENTS.md:85-89`) working as
  intended; it is not a defect to normalize away.

## II.4 — DB schema evolution

The schema is sound for what it does; the resource model asks for consistency, not
a rewrite. Ordered by value over risk:

1. **Uniform state typing.** `state` is an integer enum on jobs/tasks/attempts
   (`schema.py:259,333,429`) but a `lifecycle` free string on slices
   (`schema.py:549`) and a `handoff_state` int on federation (`schema.py:594`).
   Converge on a typed state per resource (a `StrEnum` persisted consistently) so
   `describe`/`list --state` read the same way for every noun. Keep the deliberate
   non-storage of derived attempt failure/preemption counts (recomputed by
   `AttemptCountsProjection`, `schema.py:341-345`); the resource model reads that
   projection, it does not persist a count.
2. **Name the three event streams so they stop colliding.**
   `federation_changelog` (SQLite, control-plane change events for peers to poll,
   `schema.py:634`) and `iris.task_event` (finelog, admission telemetry,
   `stats/tables.py:179`) both call themselves "the event log," and Part I adds a
   third, `iris.event` (audit). These are a decision-change feed, a measurement
   stream, and an audit trail. The plan names them by role; it does not merge them.
   The decisions-vs-measurements rule already assigns audit/measurement events to
   finelog namespaces and control-plane change feeds to controller tables.
3. **Decide what the free-floating strings are.** `backend_id`, `scale_group`,
   `user_id`, and `cluster` are referenced across many tables with no anchor table
   — the `backends` and `users` tables were dropped (migrations 0042, 0040), and
   `scaling_groups` keys on `name` while everything else says `scale_group`
   (`schema.py:530` vs `workers.scale_group:480`). Either re-introduce thin
   registry rows for the ones that are genuinely resources (backend, scale-group)
   or document them as external coordinates; align `scaling_groups.name` with the
   `scale_group` column name. This dovetails with the maintainer's flat-namespace
   direction of making `cluster` a first-class column.
4. **Pick one direction for slice↔worker.** The relation is denormalized both ways:
   `slices.worker_ids` is a JSON array (`schema.py:550`) and `workers.slice_id` is
   a column (`schema.py:479`). Keep `workers.slice_id` as the source of truth and
   derive membership, or add a junction; drop the JSON array as authoritative.
5. **Document the attempt dual identity once.** `attempt_id` (ordinal, in
   `/user/job/0:3`) and `attempt_uid` (16-hex routing key that drives the pod name,
   unique index `schema.py:435,444`) are both real and both needed. Keep both;
   state the invariant in one place; do not re-explain it at each use.

Every item stays inside the decisions-vs-measurements rule: no new measurement
columns land on controller tables; they go to finelog namespaces.

## II.5 — Scheduling and constraints: preserve, do not restructure

The scheduling path is already the closest thing iris has to a clean resource
model. The plan treats it as the reference to preserve:

- The constraint/attribute model is uniform — one `ConstraintIndex`
  (`constraints.py:1148`) matches workers in the scheduler, routes jobs to backends
  in the meta-scheduler, and routes demand in the autoscaler. Keep it.
- The `TaskBackend` contract (`controller/backend.py:464`) is a clean per-resource
  driver: declared capabilities (`BackendCapability`, `:85`), three uniform phases
  (schedule/reconcile/autoscale), plain-data frozen results dispatched on non-empty
  field. This is the seam the uniform verbs hang on (§II.6); it should not grow
  domain logic.
- The 4-scalar resource model (cpu/gpu/tpu/memory) plus constraint-matched device
  type/variant (`scheduler.py:61,250`), gang/coscheduling, priority bands, and
  Kueue-on-K8s all stay.

The one scheduling-adjacent addition the resource model wants is on the read side,
not the decision side: `task describe` should surface *why* a task is pending. The
scheduler already computes that verdict (unschedulable reasons and gate results,
`scheduling/policy.py:840`), but it is not persisted as a queryable reason.
Capturing the latest placement verdict for a pending task — as an event on the
`iris.event` stream (§II.2, §5) — answers "why is this stuck" in the
resource-model-shaped way and keeps the kernel pure: the controller writes the
event after the pass; the kernel returns it as data.

## II.6 — Backend-crossing is a capability branch, reused not reinvented

Part I §4 adds `describe_task` to the `TaskBackend` contract. Generalized: uniform
verbs that touch a live object branch on `BackendCapability` (`backend.py:85`) and
route through the existing `TaskTarget`, because the two backends diverge — the
worker-daemon backend owns workers, slices, liveness, and an autoscaler, while the
K8s `CLUSTER_VIEW` backend owns none of them and reconstructs a pod name from
`attempt_uid` plus one live GET. Worker liveness is in-memory and per-backend
(`WorkerHealthTracker`, unioned by `Controller.all_liveness`, `controller.py:601`),
so a uniform `worker list --health` is a join across backends; no single table
holds it. The plan reuses this dispatch path for every backend-crossing verb; it
adds no parallel router.

## II.7 — Package and module naming

Mechanical renames toward the vocabulary, no behavior change. Ranked by leverage:

1. **Fold the `vm` proto noun into `worker`/`slice`.** `rpc/vm.proto` models
   `VmInfo`/`VmState` nested under `SliceInfo` (`vm.proto:52,84`) while the Python
   type layer already says `WorkerStatus`/`SliceInfo` (`cluster/types.py:497`). This
   proto-vs-types split is the largest naming inconsistency; the CLI's `cluster vm
   status` (`cli/cluster.py:934`) is its surface symptom.
2. **Rename `platforms/vm_lifecycle.py` → `controller_lifecycle.py`.** Its docstring
   says "Controller lifecycle as free functions" and it exports `start_controller`/
   `stop_controller` (`vm_lifecycle.py:16,444`); the filename is wrong.
3. **Split the CLI grab-bags** (Part I §9): `cli/cluster.py` (1615 lines, six nouns)
   into per-noun modules; delete the synthetic `process` group
   (`cli/process_status.py`), moving `status`→`describe`/`logs`/`profile` onto
   `worker`, `task`, and the controller view.
4. **Align the controller-side stop vocabulary.** `kick_tasks`/`terminate_job`/
   `request_task_kicks` (`service.py:2174`, `controller.py:576`) and
   `Worker.KillTask` are the server-side counterpart of the CLI's stop/kill/kick
   sprawl; converge on stop/terminate as Part I collapses the CLI verbs.
5. **`endpoints` group → `endpoint`** for singular-noun consistency
   (`cli/endpoints.py:23`).
6. **Make the shared CPU-sampling helper public.** `cluster/process_status.py`
   imports `_read_proc_cpu_millicores` from `cluster/runtime/process.py` across a
   module boundary; a `_private` name reused elsewhere should be public.
7. **Optional, later:** `controller/service.py` is 3524 lines; if the resource
   surface is added, split the one monolithic RPC impl along the resource nouns.

## II.8 — Docstrings

Standardize on the pattern `cluster/controller/reads.py` already uses, and make it
the reference: state a shared cross-module mechanism once in the module docstring,
then a one-line imperative Google-style summary on every public function whose
behavior is not obvious from a return-type-reflecting name; skip trivial one-line
accessors; drop `__all__` blocks that only mirror the import list. Concrete first
targets from the sweep: missing summaries on `worker/worker.py:241 start()` /
`:400 wait()`, `controller/controller.py:1027 owns_scale_group()` / `:1137
resolve()`, `types.py:506 is_idle()` / `:843 is_job_finished()`; redundant
`__all__` re-export blocks in `cluster/runtime/__init__.py:24` and
`client/__init__.py:29`; a bulleted mechanism-restating module docstring in
`cluster/runtime/process.py:4`. Run this alongside the naming changes in §II.7,
documenting each shared mechanism once; do not repeat it per call site.

## II.9 — Phasing, and what this plan will not do

The internal work is sequenced after Part I and split so each phase is
independently reviewable:

- **Phase A (shared with Part I):** the typed `ResourceRef` at the service layer,
  `describe_task` on `TaskBackend`, and the `iris.event` namespace. Part I already
  needs all three, so they land with the CLI and are the foundation the rest builds
  on.
- **Phase B (mechanical, no behavior change):** the naming sweep and package splits
  (§II.7) and the docstring sweep (§II.8). Pure refactors, landable in small PRs,
  no migration.
- **Phase C (migration-gated, last):** the DB schema alignments (§II.4) — uniform
  state typing, event-stream naming, anchoring or documenting the free-floating
  strings, the slice↔worker direction. Highest risk, done incrementally behind
  migrations, each paired with a `describe`/`list` read that proves the shape
  before and after.

Explicitly out of scope, to keep the framing honest:

- No single `Resource` base class over all eight nouns (§II.1); the classes do not
  share a lifecycle.
- No resource logic inside the scheduling or reconcile kernels (§II.3); they stay
  pure.
- No merging of the three event streams (§II.4); they differ in kind.
- No persisting of derived state — attempt counts, per-cycle capacity, and worker
  liveness stay computed, not stored (§II.4, §II.5).
- No new addressing scheme: `ResourceRef` subsumes `target`/`source`/`TaskTarget`
  (§II.2, §II.6).

## Open questions for review

Part I (the CLI, this PR):

1. **`describe` transport:** new `DescribeTask` RPC, or extend
   `GetTaskStatusResponse` with an optional `backend_detail` populated on request?
   The latter is fewer moving parts; the former keeps `GetTaskStatus` cheap
   (no live pod GET unless asked).
2. **`events` retention:** is 30 days the right forensic window for `iris.event`,
   and is finelog the agreed home (vs. a controller table)? The "decisions vs
   measurements" rule and the `iris.task_event` precedent both point to finelog,
   but audit events arguably want stronger durability than measurements.
3. **`worker`/`endpoint` stop:** keep `drain`/`unregister` as distinct lifecycle
   spellings, or force them under `stop` for strict uniformity? I lean toward
   keeping them — "terminate a worker" is not "terminate a workload."
4. **`process` dissolution:** does folding controller inspection into `cluster`
   (`cluster describe`/`logs`/`profile`) read well, or should the controller be its
   own noun despite not being in the issue's set?
5. **Scope of one PR:** land stages 1–2 (small, immediate) separately from the
   larger 3–7? The maintainer asked to "normalize everything"; I propose one
   design, delivered as the spiral above, but the split point for PRs is worth
   agreeing on.

Part II (the internal model, plan):

6. **Resource classes:** is the A/B/C split (§II.1) the right cut? The debatable
   case is worker/slice — they have DB rows (Class-A-like) but their authoritative
   state lives in backend handles and their `stop` is infra teardown (Class B). I
   put them in B; the alternative is a fourth "record-backed infra" class.
7. **`ResourceRef` timing:** extract the typed ref in Phase A (shared with the CLI),
   or defer until the CLI has proven the vocabulary? Extracting early avoids each
   new verb re-parsing `target`/`source`; deferring keeps this PR smaller.
8. **DB state typing (§II.4.1):** is converging slice `lifecycle` and federation
   `handoff_state` onto a persisted `StrEnum` worth a migration, or is the current
   int-enum/string split tolerable given how rarely those tables change?
9. **Free-floating strings (§II.4.3):** re-introduce thin anchor tables for
   `backend`/`scale_group` (dropped in migrations 0042/0040), or formally document
   them as external coordinates and only align the `scaling_groups.name` naming?
10. **Service split (§II.7.7):** is splitting the 3524-line `service.py` along the
   resource nouns in scope for this effort, or a separate refactor to keep the
   vocabulary PRs reviewable?
