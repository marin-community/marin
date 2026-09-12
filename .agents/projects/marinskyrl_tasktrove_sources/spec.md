# TaskTrove Clean source contract

## Marin public API

File: `lib/marin/src/marin/rl/skyrl.py`

```python
class TaskTroveTagMatch(StrEnum):
    ALL = "all"
    ANY = "any"


@dataclass(frozen=True)
class TaskTroveSelection:
    sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    tag_match: TaskTroveTagMatch = TaskTroveTagMatch.ALL
    limit: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class TaskTroveDataSource:
    step: ArtifactStep[Artifact]
    selection: TaskTroveSelection
    relative_path: str = "tasks/part-00000.parquet"
    manifest_path: str = "manifest.json"
```

`TaskTroveSelection` rejects an entirely empty predicate, blank or duplicate values, and a non-positive limit. It sorts source, tag, and mode values during normalization so semantically equal selections have equal fingerprints. `SkyRLSpec.train_data` accepts the data-source protocol implemented by `ArtifactDataSource` and `TaskTroveDataSource`.

`TaskTroveDataSource.resolve` reads `manifest.json`'s `verify_tool_ref` during a real launch. It emits a stable placeholder during fingerprint construction; the producer artifact dependency already includes the real tool ref in its fingerprint.

## Wire types

```python
@dataclass(frozen=True)
class DirectoryDataSource:
    kind: Literal["directory"]
    uri: str
    identity: str
    local_path: str
    relative_path: str


@dataclass(frozen=True)
class TaskTroveSelectionSnapshot:
    count: int
    digest: str
    distinct_environment_count: int


@dataclass(frozen=True)
class TaskTroveParquetSource:
    kind: Literal["tasktrove_parquet"]
    uri: str
    identity: str
    local_path: str
    relative_path: str
    verifier_ref: str
    selection: TaskTroveSelection
    snapshot: TaskTroveSelectionSnapshot | None = None
```

The JSON parser requires `kind` and rejects unknown fields. Before launch, MarinSkyRL fills one typed selection snapshot containing the count, digest, and distinct environment count, then persists the resolved request. Directory sources retain their behavior. Packed sources stage only the exact Parquet object under `local_path/relative_path`; they are never expanded by task runtime.

Example:

```json
{
  "kind": "tasktrove_parquet",
  "uri": "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9/tasks/part-00000.parquet",
  "identity": "tasktrove/clean@2026.09.10.9:<fingerprint>",
  "local_path": "/tmp/marinskyrl/data/tasktrove-clean",
  "relative_path": "part-00000.parquet",
  "verifier_ref": "git+https://github.com/marin-community/marin.git@<commit>#subdirectory=lib/tasktrove-verify",
  "selection": {
    "sources": ["DCAgent2__nl2bash-tasks-cleaned-oracle-v2"],
    "tags": ["bash", "terminal"],
    "modes": ["script"],
    "tag_match": "all",
    "limit": 160,
    "seed": 17
  }
}
```

## Selection

Required columns are validated by name and Arrow type; extra columns are allowed. Metadata selection reads only `source`, `tags`, `mode`, `path`, and `dockerfile_id`.

- `sources`: exact, case-sensitive equality; values are OR'd.
- `tags` with `ALL`: every requested tag occurs in the row's `list<string>`.
- `tags` with `ANY`: at least one requested tag occurs.
- `modes`: exact, case-sensitive equality; values are OR'd.
- Non-empty source, tag, and mode clauses combine with AND.
- Every requested source and tag must occur somewhere in the release.
- `limit`: keep the lexicographically smallest `sha256(f"{seed}\0{source}\0{path}".encode("utf-8")).digest()` values.
- Membership follows hash order; returned references follow `(row_group, row)` order.

```python
@dataclass(frozen=True)
class PackedTaskReference:
    dataset_path: str
    dataset_identity: str
    row_group: int
    row: int
    source: str
    path: str
    dockerfile_id: str
    mode: str

    def stable_uri(self) -> str: ...


@dataclass(frozen=True)
class TaskSelectionSummary:
    references: tuple[PackedTaskReference, ...]
    digest: str
    distinct_environment_count: int
```

The digest hashes the canonical JSON of each ordered reference's dataset identity, row group, row, source, path, `dockerfile_id`, and mode. The launch-host summary is authoritative; the Ray-side local scan must equal it.

## Node cache and lazy extraction

The exact Parquet object is cached once per allocated node at `local_path/relative_path` using its immutable identity and a size-checked atomic rename. For `.9`, this may copy all 3,837,032,177 bytes, but it creates one local file and no task directories.

`PackedTaskMaterializer` owns an identity-keyed task cache at `tasks/<dataset_identity_hash>/<source_path_hash>/`. It groups a rollout batch by row group, reads the identity columns and `task_binary` once for each required group, validates reference identity, and extracts each unique assigned task once. An inter-process lock plus atomic rename protects the shared task cache. `solution_binary` is not requested. A separate row-group cache or prefetch stage can be added after profiling; the first implementation relies on the already node-local Parquet file.

Archive validation accepts regular files only and rejects links, absolute paths, `..`, duplicate normalized paths, and every `solution/` member. It requires `instruction.md`, `task.toml`, and `environment/Dockerfile`. The materializer compares the row's source, path, mode, and published `dockerfile_id` with the selected reference before atomic publication.

`TerminalBenchTaskDataset` returns:

```python
{
  "prompt": reference.stable_uri(),
  "uid": f"{dataset_identity}/{source}/{path}",
  "env_class": None,
  "env_extras": {
    "data_source": reference.stable_uri(),
    "packed_task": dataclasses.asdict(reference),
  },
}
```

The Harbor runner is the only consumer that interprets `packed_task`. It substitutes the materialized local directory for `task_path` immediately before `HarborConfigBuilder.build_trial_config`. A data mismatch or unsafe archive aborts the run as task-data corruption; it does not synthesize a partial GRPO group.

## Provenance

`resolved-skyrl.json` and `terminal.json` retain the source URI, artifact identity, verifier ref, normalized selector, selected count, selection digest, distinct environment count, and MarinSkyRL runtime commit. Retained per-task metadata carries dataset identity, source, path, mode, and `dockerfile_id`. Harbor remains authoritative for the materialized task checksum and environment digest.

## Repository scope

| Repository | Path | Contract |
|---|---|---|
| Marin | `lib/marin/src/marin/rl/skyrl.py` | public selection and tagged source types |
| Marin | `experiments/post_training/tasktrove/rl_smoke.py` | direct packed-source smoke |
| MarinSkyRL | `cloud/iris/protocol.py` | tagged request and preflight summary |
| MarinSkyRL | `cloud/iris/artifacts.py`, `task_runtime.py` | exact-file node cache |
| MarinSkyRL | `marinskyrl/packed_tasks.py` | selector, digest, grouped row reads, extraction |
| MarinSkyRL | `skyrl-train/skyrl_train/trajectory_runners/harbor/dataset.py` | packed dataset items |
| MarinSkyRL | `skyrl-train/skyrl_train/trajectory_runners/harbor/runner.py` | rollout-batch materialization |

Harbor does not change for the first slice. A generic admission-time async task loader is a possible follow-up if batch-level unpacking proves materially early.

## Deferred contracts

- Snapshot reservation, protected LRU cleanup, cap-full telemetry, and derived runtime-interface salting.
- Publisher sorting and smaller row groups for a later release.
- Family/language filters and disjoint train/validation splits.
- Capability declarations, judge credentials, and judge taxonomy.
