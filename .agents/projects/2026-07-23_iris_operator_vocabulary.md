# A uniform noun-verb vocabulary for the Iris operator CLI

> Status: **proposal, for peer review** — written before implementation, against
> the code as of this branch. GitHub issue
> [#7543](https://github.com/marin-community/marin/issues/7543); motivated by the
> `cw-us-east-02a` job-loss incident
> [#7542](https://github.com/marin-community/marin/issues/7542). The code is the
> source of truth; every claim below is cited to `file:line` so a reviewer can
> check it. Nothing here is implemented yet.

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

## Open questions for review

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
