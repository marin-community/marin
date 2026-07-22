# Harbor and Daytona scripts port scope

## Decision

Port maintained capabilities, not the OpenThoughts-Agent directory layout. The two
source directories contain 21 executable Python/shell scripts plus two package
markers. Harbor has a mix of reusable data/trace capabilities and historical
repairs. Daytona is supported Marin infrastructure for container workloads
beyond Harbor, so its lifecycle, snapshot, health, inspection, and configuration
capabilities must become a small generic Marin surface rather than be discarded.

The Harbor task packaging, trace export, literal-token conversion, and deterministic
result repair belong in marin-core adjacent to the Harbor evaluator. Daytona client
operations also belong in marin-core as an optional dependency: they cannot be
made Iris code because they are a distinct provider and need not route through an
Iris controller. Iris Harbor sandboxes remain an additional supported backend, not
a replacement for Daytona. No ported module may import OpenThoughts-Agent, read a
DC_AGENT_SECRET_ENV/.claude/secret.md file, rely on a
/Users/benjaminfeuer/... path, or write to the OpenThoughts Supabase schema.

This scope was prepared from Marin main at b22baf7b5bd5298faa2e944d8072ffed260f4561
and OpenThoughts-Agent 5b1f8d94 (2026-07-22).

## Current assets to reuse

| Need | Existing Marin asset | Porting consequence |
|---|---|---|
| Harbor jobs and result layout | marin.evaluation.evaluators.harbor_evaluator.HarborEvaluator | Consolidate job configuration, execution, resume, and result parsing here rather than add a second runner. |
| Iris-hosted sandbox environment | marin.harbor.iris_environment.IrisEnvironment and marin.harbor.sandbox.iris_sandbox | A supported Harbor/backend implementation and a useful reference for provider-neutral sandbox contracts. |
| Local credential conventions | iris CLI's environment forwarding and rigging credential/config patterns | Daytona needs an explicit non-secret profile and reads its key only from a named process environment variable. |
| Pinned Harbor API | root pyproject source pin and marin-core harbor extra | Use the pinned API directly; do not perpetuate a cross-version compatibility shim. |
| HF upload | marin.export.hf_upload.UploadToHfConfig | Extend its public API only where task-parquet streaming needs a capability it lacks. |
| Filesystems and object storage | fsspec, rigging.filesystem.StoragePath, GCS/S3 support in marin-core | Make all source/destination locations explicit URIs; do not recreate OpenThoughts gcs_cache. |
| Dataset transforms | marin.transform, marin.datakit, marin-levanter dataset helpers | Keep task archive formats under a small Harbor-specific module; do not introduce reverse imports. |
| Controller/database access | marin-iris client and diagnostics work scoped in the Iris port | Harbor analysis consumes local evidence bundles / typed APIs, never OpenThoughts database helpers. |

datasets, pyarrow, tqdm, rich, and python-dotenv already resolve in the workspace.
daytona is not currently a Marin dependency and should become a narrow
marin-core[daytona] extra. supabase remains out of scope because the one source
script is tied to an OpenThoughts-only reporting schema.

## Source inventory and disposition

### Harbor

| OpenThoughts source | Responsibility and coupling | Target | Disposition |
|---|---|---|---|
| _harbor_compat.py | Legacy/unified Harbor config, event, hook, and factory shims. Imported by 12 OpenThoughts callers. | none | Drop. Marin pins one Harbor revision; update future Marin callers to its API. Keeping a compatibility layer violates the no-backwards-compatibility policy. |
| count_snapshots_from_tasks.py | Counts unique Dockerfile environment hashes from local, HF-parquet, or registry tasks. Imports Harbor cache internals. | marin.harbor.task_images; thin scripts/harbor/count_task_images.py | Port, redesign. Report image identity/count for planning, but name it after task images rather than Daytona snapshots. Support task directories and HF-parquet first; registry download follows only with a stable Harbor public API. |
| job_config_utils.py | YAML/JSON JobConfig parsing, agent kwargs normalization, local dataset rewrite, legacy metric filtering. Imported by four OpenThoughts callers. | marin.harbor.job_config | Merge. Expose typed load/dump/override operations; delete unsupported legacy metric filtering after configs migrate. |
| literal_correlator.py | Pure literal-record parsing, chain reconstruction, conservative token/logprob injection; thin UPath discovery I/O. Tests cover matching rules. | marin.harbor.literal_records | Port. Preserve the pure data API and split URI discovery from matching. |
| literal_traces_to_sft.py | Converts literal-token HF trace rows to SFT conversations/text, resolves tokenizer provenance, uploads to HF. | marin.harbor.literal_sft; scripts/harbor/literal_traces_to_sft.py | Port. Reuse literal-record types and the Marin HF upload boundary; retain strict mismatch-dropping behavior. |
| make_and_upload_trace_dataset.py | Streams completed Harbor trials to sharded HF traces, optional literal enrichment, tokenizer provenance, and optional OpenThoughts Supabase registration. | marin.harbor.trace_export; scripts/harbor/export_traces.py | Port exporter only. Replace Supabase registration with an explicit output manifest; do not port database write. |
| pin_task_env.py | Ad hoc package diff/pinning and Docker-vs-Daytona snapshots for one task. | none | Drop as a one-off data repair. Task Dockerfiles should be corrected/versioned at dataset construction; reusable provider diagnostics belong in marin.daytona, not a dataset-specific patcher. |
| recompute_metrics_from_db.py | Supabase job/metric recomputation and CSV reporting. | none | Drop. It is coupled to the OpenThoughts unified-db schema and old metrics. Recompute future records from Harbor results/evidence bundles. |
| recompute_result_json.py | Rebuilds one Harbor result.json from trial result files using Harbor metrics. | marin.evaluation.harbor_results; scripts/harbor/recompute_result.py | Port. Useful deterministic repair tool; accepts explicit local/fsspec job URI and writes only with --write (report/dry-run is default). |
| run_and_export_traces.py | Programmatic Harbor local-dataset runner plus trace serialization/export. Imports OpenThoughts cache and task-archive helpers; 92 OpenThoughts callers import it. | marin.harbor.trace_runner | Merge into new runner/exporter surface. Largest migration risk; split before movement. |
| tasks_parquet_converter.py | Deterministically archives Harbor task dirs into path/task_binary Parquet, extracts local/HF Parquet, uploads to HF. Imported by 16 callers. | marin.harbor.task_archive; scripts/harbor/task_archive.py | Port. This is foundational standalone format logic. Reuse fsspec/HF APIs and eliminate OpenThoughts cache dependence. |
| __init__.py | Package marker only. | package markers as needed | No standalone migration. |

### Daytona

| OpenThoughts source | Responsibility and coupling | Target | Disposition |
|---|---|---|---|
| daytona_client.py | Resolves API key from argv/env/secrets file and creates Daytona client. | marin.daytona.client and marin.daytona.config | Port the client boundary, not the secret-file fallback. Client construction takes typed config and a key resolved only from an explicit process-environment variable. |
| cleanup_stale_sandboxes.py | Lists/deletes stale Daytona sandboxes; imports daytona_client. | marin.daytona.sandboxes; scripts/daytona/sandboxes.py | Port. Keep default read-only inventory, exact typed state/age selection, dry-run, and explicit confirmation before deletion. |
| daytona_snapshot_manager.py | Read-only/default-safe snapshot audit and destructive stale deletion. Imports daytona_client. | marin.daytona.snapshots; scripts/daytona/snapshots.py | Port. Preserve paging, protected transitional states, JSON audit, and confirmation-gated deletion; remove organization-specific hard caps, prefix assumptions, and secrets-file handling. |
| health_check.py | Direct Daytona create/exec/delete latency and concurrency benchmark, with Jupiter proxy instructions. | marin.daytona.health; scripts/daytona/health.py | Port as generic provider health probe. Preserve structured per-phase timing and bounded concurrent probes; remove Jupiter paths/proxychains instructions and let normal process networking/configuration apply. |
| inspect_daytona_data.py | Builds Daytona sandbox and dumps staged task files for debug. | marin.daytona.inspect; scripts/daytona/inspect.py | Port. Generalize to explicit image plus optional local upload roots and remote destinations; Harbor task-layout staging becomes an adapter, not hard-coded generic behavior. |
| patch_freelancer_testbed.py | One-off rewrite of named Freelancer task dataset/Dockerfile. | versioned dataset transform only if still needed | Defer/drop. It must not become a generic operation. If dataset remains active, create a transform under experiments/datasets with explicit source/target IDs. |
| search_sandbox_jobs.py | Queries OpenThoughts Supabase sandbox_jobs/benchmark/model tables and emits CSV matrices. | none | Drop. Its schema and credentials do not exist in Marin. Query W&B/eval records or typed Harbor evidence bundles instead. |
| validate_and_upload_from_hf.py | Extracts HF task Parquet; validates buildability via Daytona, Harbor smoke, and oracle; writes survivors to HF. Imports task archive, OpenThoughts upload helper, Harbor compatibility shim. | marin.harbor.task_validation; scripts/harbor/validate_tasks.py | Port. Preserve Daytona build validation as one backend, add Iris/prebuilt-image validation as another, and keep Harbor smoke/oracle as separately selectable stages. Inputs/outputs use task_archive and UploadToHfConfig. |
| batch_validate_from_md.sh | User-path-specific shell loop over unpublished Markdown list and validator. | none | Drop. Generic validator gains --input-manifest (JSONL/Parquet); personal Markdown parser and absolute defaults do not belong in Marin. |

## Target API and layout

Harbor library code belongs in marin-core because it owns Harbor/evaluation and
training-data semantics. Daytona is likewise a marin-core provider integration:
it is useful outside Harbor, so it must not live under marin.harbor or
marin-iris. Harbor may import marin.daytona through an explicit backend factory;
neither generic provider module may import Harbor. Small command wrappers remain
under scripts/harbor or scripts/daytona and contain argument parsing only.

~~~
lib/marin/src/marin/
  harbor/
    task_archive.py       # archive_tasks(), extract_task_archive(), list_task_dirs()
    task_images.py        # task_image_inventory()
    job_config.py         # load_job_config(), write_job_config(), JobOverrides
    literal_records.py    # parse/reconstruct/bind/inject plus URI adapters
    trace_runner.py       # run_local_harbor_job(), read_trial_records()
    trace_export.py       # export_trial_traces(), write_trace_dataset()
    literal_sft.py        # convert_literal_trace_rows()
    task_validation.py    # validate_task_set() with Daytona/Iris/Harbor/oracle stages
  daytona/
    config.py             # DaytonaConfig and resolve_daytona_credentials()
    client.py             # typed Daytona SDK factory/protocol boundary
    sandboxes.py          # inventory, age/state selection, guarded deletion
    snapshots.py          # paged audit, protected-state selection, guarded deletion
    health.py              # create/exec/delete probe and concurrency summaries
    inspect.py             # sandbox staging/remote-file inspection
  evaluation/
    harbor_results.py     # summarize/recompute Harbor result.json

scripts/harbor/
  task_archive.py
  count_task_images.py
  export_traces.py
  literal_traces_to_sft.py
  validate_tasks.py
  recompute_result.py
scripts/daytona/
  sandboxes.py
  snapshots.py
  health.py
  inspect.py
~~~

Key contracts:

~~~
@dataclass(frozen=True)
class TaskArchive:
    root: StoragePath
    task_count: int
    parquet_uri: str

@dataclass(frozen=True)
class TraceExport:
    output_uri: str
    completed_trials: int
    skipped_trials: int
    literal_enrichment: LiteralEnrichmentStats

@dataclass(frozen=True)
class TaskValidationResult:
    task_id: str
    environment_build: ValidationOutcome
    harbor_smoke: ValidationOutcome | None
    oracle: ValidationOutcome | None

@dataclass(frozen=True)
class DaytonaConfig:
    endpoint: str
    target: str | None
    api_key_env: str

@dataclass(frozen=True)
class SandboxAuditRow:
    sandbox_id: str
    state: str
    created_at: datetime
    last_activity_at: datetime | None
    delete_eligible: bool
~~~

DaytonaConfig holds no token. resolve_daytona_credentials(config) reads only the
specified api_key_env from the current process and fails before network I/O if it
is unset. The first port should require --api-key-env (with DAYTONA_API_KEY as
the documented conventional value), --endpoint, and --target explicitly or from
a non-secret named profile. The profile contains endpoint/target/key-variable name
only; no command accepts a literal token or a secret-file path. Iris jobs pass a
selected key through EnvironmentSpec.env_vars/inject_env under operator control.
The client factory never logs headers, token values, or its config repr.

task_validation receives an explicit SandboxFactory/Harbor environment
configuration. DaytonaSandboxFactory builds Dockerfile-backed tasks; IrisSandboxFactory
supports prebuilt images. It distinguishes task failures from provider
infrastructure failures, preserving the IrisSandboxError classifier for Iris and
introducing an analogous typed Daytona infrastructure outcome.

The generic task archive is deliberately independent of job launch, Datagen, and
RL code. Trace export can call HarborEvaluator helpers, but job execution must
not import experiments/ or old OpenThoughts launch configuration.

## Dependency and coupling assessment

There are five direct OpenThoughts package boundaries in retained Harbor
capabilities:

1. data.gcs_cache in task archive/runner;
2. data.commons task-archive extraction and HF upload;
3. database.unified_db registration and metrics recomputation;
4. the Harbor cross-version shim; and
5. implicit package-root mutation in the Daytona validator.

All five can disappear. marin-core already supplies fsspec storage, HF upload,
Harbor as an optional extra, and Iris sandbox execution. Retained code has zero
required OpenThoughts imports after the split.

The port adds two deliberate boundaries: a small stable task-archive module and
a marin-core[daytona] optional dependency on the official Daytona SDK. Do not add
supabase: sandbox inventory/deletion comes from the Daytona API, while the old
Supabase job-matrix report is an application-specific historical view.

The material migration scope is run_and_export_traces.py: 92 source-tree imports
reference it. This is a caller count, not a reason to retain its API. Move it
only after public inputs are explicit and migrate active callers in batches. Task
archive has 16 known imports; compatibility shim has 12. Other retained scripts
have at most five direct callers.

## Incremental implementation plan

### 1. Establish the Daytona provider foundation

Add marin-core[daytona] and a small marin.daytona.config/client package. The
config is explicit: endpoint, target, and the name of an environment variable
holding the API key. It does not parse dotenv files, inspect an OpenThoughts
secret path, or accept a token in argv. Keep the Daytona SDK behind narrow
protocols for sandbox and snapshot operations so tests can use fakes.

Port the default-safe parts of cleanup_stale_sandboxes and
daytona_snapshot_manager first: paged inventory, state normalization, stale
selection, JSON/human reports, and guarded deletion. The delete command must
require both --delete and --yes (or an interactive exact-count confirmation);
--dry-run is the default. Its scope is selected by explicit target/profile and
optional exact prefix, never by an implicit organization-wide default.

Add fixture/fake tests for environment resolution without printing the token,
pagination, protected snapshot states, stale-age boundaries, dry-run output, and
the refusal to delete without confirmation.

### 2. Establish tests and task-archive primitive

Add marin.harbor.task_archive, porting only deterministic archive creation, safe
extraction, task discovery, and HF Parquet load/upload. Use a local temporary
directory and a memory/fake filesystem in tests; assert that path-traversal
members are rejected and archive/extract round trips preserve task content.
Reuse marin.export.hf_upload for actual HF publication instead of another
uploader.

This phase carries no caller migration and no launch behavior.

### 3. Move pure Harbor result and literal logic

Port literal_correlator as a pure module plus minimal URI adapters and port
recompute-result algorithm into marin.evaluation.harbor_results. Port existing
literal-correlation, literal-rescue, and literal-to-SFT behavior tests as
fixture-based Marin tests. Add result fixtures for successful, exceptional, and
missing-trial outcomes. Repair CLI remains dry-run by default.

### 4. Build trace runner/exporter on task archive

Split monolith into (a) JobConfig loading/overrides, (b) programmatic Harbor
execution boundary, and (c) bounded trace-to-Parquet/HF export. Replace direct
GCS cache helpers with explicit StoragePath inputs; replace Supabase registration
with emitted JSON manifest next to output. Test observable dataset rows,
resume/idempotency, literal enrichment omission on ambiguous match, and manifest
counts. Do not test Harbor command-string assembly.

After controlled parity run against one small local task set, migrate currently
active source callers. Do not mass-edit historical OpenThoughts data generators
until their use is confirmed; they may use their copy during transition.

### 5. Port provider diagnostics and task validation

Port Daytona health_check and inspect_daytona_data on the provider foundation.
Health takes an explicit image or snapshot and reports create, exec, and delete
latencies plus per-probe failures; it has bounded concurrency and writes JSON
when requested. Inspection takes explicit local source-to-remote-destination
mappings, so Harbor's data/tests/solution layout is an adapter rather than an
assumption. These commands are read-only except the short-lived sandbox they
create and always clean up that owned sandbox in a finally path.

Build task_validation from task archive and a backend enum/factory. Preserve the
Daytona Dockerfile build stage, add an Iris prebuilt-image stage, and retain
Harbor smoke/oracle stages as separately selectable. Results record the backend,
provider operation IDs, timing, and a typed task-versus-infrastructure outcome.

Tests use fake Daytona and Iris factories plus fixture tasks to verify stage
ordering, survivor selection, retry classification, cleanup after probe failure,
and no upload on zero survivors. A manually approved integration/reference run
should cover one Daytona Dockerfile task and one Iris prebuilt-image task.

### 6. Add wrappers and retire only obsolete scripts

Wire six thin scripts/harbor entry points and four scripts/daytona entry points
to library. Update Harbor and operations docs with both supported sandbox
backends, the explicit Daytona configuration interface, and safe deletion rules.
Delete the secrets-file client and replace the personal Markdown batch wrapper;
retain no compatibility aliases. Keep the Freelancer patch as a separately
versioned data transform if it is still active, and retire the Supabase matrix
report rather than presenting it as provider job reporting.

Cutover requires a Daytona audit and guarded deletion dry run agreeing with the
current tools, a task archive round trip, and one small trace export agreeing on
task count, trial count, trace count, and mean reward. Only then delete retained
OpenThoughts copies and migrate active callers; historical generators can be
removed in a later separate cleanup.

## Deferred decisions and blockers

1. Daytona SDK contract/version. Marin does not yet pin the official SDK, and
   its pagination, delete, and async sandbox APIs must be verified against one
   selected version before designing the protocol. This is a bounded dependency
   addition, not a reason to keep OpenThoughts secret-file code.
2. Credential injection policy. The CLI can safely resolve a named process
   environment variable now, but shared named profiles and Iris job injection
   need an owner-approved non-secret config location and allowed variable names.
   No token value needs to be committed or placed on a command line.
3. Backend capability mismatch. Daytona builds Dockerfile-backed tasks while
   IrisEnvironment supports prebuilt images. Task validation must expose this as
   a backend capability decision, not treat either backend as a fallback for the
   other.
4. Trace-runner callers. The 92 imports are mostly old dataset generators. We
   need the currently supported-generator list before changing their API;
   migration should be staged by active workload, not global search/replace.
5. Supabase historical reporting. There is no Marin-equivalent schema. Retain
   historical reports outside port and design new reporting on W&B or typed
   Harbor evaluation records.
6. Registry task-image inventory. Current code reaches Harbor cache internals.
   Local/HF-parquet inventory is sufficient initially; registry support needs
   upstream-stable Harbor method or pinned adapter with compatibility test.

## Acceptance criteria

- New code has no OpenThoughts-Agent, DC_AGENT_SECRET_ENV, .claude/secret.md,
  user-home absolute path, or supabase import/reference. Daytona appears only
  behind marin-core[daytona]'s typed provider boundary.
- Daytona configuration carries endpoint, target, and API-key environment-variable
  name, never an API key. Inventory is read-only by default; sandbox/snapshot
  deletion requires explicit scope plus confirmation and records selected IDs.
- Task directory archive round-trips deterministically and safely through
  Parquet; resulting archive is consumed by trace export and validation.
- Literal traces preserve only verified token/logprob alignments and omit
  ambiguous/mismatched alignments.
- Completed Harbor job exports/recomputes from local or fsspec URI without
  database write.
- Validation reports structured per-stage outcome, backend/provider operation
  IDs, and distinguishes task failure from sandbox infrastructure failure.
- Small Daytona-backed and Iris-backed validation/reference runs match current
  completed/failed counts and do not expose credentials.
- Completed cutover removes/replaces all reusable source scripts; each deferred
  data repair is either retired or recorded as a standalone experiment, with no
  compatibility aliases.
