# Per-User GCS Storage Prefix — Spec

Contract layer for the design in [`design.md`](./design.md). Defines the `UserSharedFS` class, the `users://` URL grammar, the identity resolver, error types, layer-resolution semantics, persisted layout, and the wiring surface in `experiments/defaults.py`.

## Executor changes

Two small changes to `lib/marin/src/marin/execution/executor.py` are required so that an individual step can opt out of the executor's global prefix.

### `ExecutorStep.output_path_prefix` (new field)

```python
@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT_co]):
    name: str
    fn: ExecutorFunction
    config: ConfigT_co
    description: str | None = None
    override_output_path: str | None = None
    resources: ResourceConfig | None = None
    output_path_prefix: str | None = None  # NEW
    """Per-step prefix override. If set, takes precedence over the Executor's
    global ``self.prefix`` when computing this step's hash-keyed output path.
    The ``{name}-{hash}`` suffix is preserved either way. Used to route
    specific step types (e.g. training checkpoints) into a different storage
    namespace such as ``users://``. None means "use the Executor's prefix"
    (current behavior)."""
```

### `Executor.compute_version` line 1504

The line that joins prefix and hashed name changes from:

```python
output_path = os.path.join(self.prefix, step.name + "-" + hashed_version)
```

to:

```python
prefix = step.output_path_prefix or self.prefix
output_path = os.path.join(prefix, step.name + "-" + hashed_version)
```

`os.path.join` handles the trailing-`://` of URL schemes correctly: `os.path.join("users://", "name-hash")` yields `"users://name-hash"`. No additional normalization is needed at this site.

### `step_spec.py:114` and `:116` — fix slash count

`StepSpec.output_path` currently uses f-string concatenation: `f"{prefix}/{x}"`. With `prefix="users://"` this yields `"users:///x"` (triple slash). Replace both lines with `os.path.join(prefix, x)` to match the executor's join semantics and produce `"users://x"` consistently.

### `compute_output_path` signature

```python
def compute_output_path(
    name: str,
    config: Any,
    *,
    override_output_path: str | None = None,
    prefix: str | None = None,
    output_path_prefix: str | None = None,  # NEW
) -> str:
    """Compute the concrete output path a step with this name+config will produce.

    ``prefix`` continues to control the throwaway executor's ``self.prefix``
    (and its ``executor_info_base_path``); ``output_path_prefix`` is forwarded
    to the constructed ``ExecutorStep`` and takes precedence over ``prefix``
    when set. Callers wanting per-step routing pass ``output_path_prefix``;
    callers that just want a different global prefix pass ``prefix``.
    """
```

## `UserSharedFS`

A new fsspec filesystem registered under the `users` protocol. Lives in `lib/rigging/src/rigging/filesystem.py` alongside `MirrorFileSystem`. Sibling implementation; no shared base class.

```python
class UserSharedFS(fsspec.AbstractFileSystem):
    """Fsspec filesystem with layered read resolution and writes-to-personal.

    Reads probe three layers in order:
      1. ``{prefix}/users/{user}/<path>`` — the caller's personal dir
      2. ``{prefix}/users/*/<path>`` — single GCS glob across other users
      3. ``{prefix}/<path>`` — the shared root

    The first layer where the artifact exists is the concrete URL returned
    to callers (``_info``, ``_ls``, ``_exists``, ``_open`` in read modes).
    Writes (``_open`` in write/append modes, ``_makedirs``, ``_rm``) always
    target layer 1 — the personal dir — regardless of which layer a prior
    read resolved to. The other-user and shared layers are structurally
    read-only.

    The base prefix and resolved username are computed lazily on first
    operation, not in ``__init__`` — so constructing the filesystem in an
    environment with no resolvable identity does not raise; only attempting
    to use it does.
    """

    protocol = "users"

    def __init__(
        self,
        *args: Any,
        user: str | None = None,
        prefix: str | None = None,
        storage_options: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct the layered filesystem.

        Args:
            user: Explicit username override. If ``None``, resolved lazily
                on first read/write via ``resolve_marin_user()``.
            prefix: Base GCS prefix (e.g. ``gs://marin-us-central2``). If
                ``None``, resolved lazily via ``marin_prefix()`` on first
                read/write — so ferries that set ``MARIN_PREFIX`` dynamically
                are respected.
            storage_options: Passed through to the underlying ``gcsfs``
                instance constructed lazily on first operation.
        ``*args`` and remaining ``**kwargs`` are forwarded to
        ``AbstractFileSystem.__init__``.
        """

    def _resolve_path(self, path: str, *, for_write: bool = False) -> str:
        """Map a ``users://`` path to a concrete ``gs://`` URL.

        Read mode (``for_write=False``): probes the three layers in order
        and returns the first concrete URL where the artifact exists.
        Raises ``FileNotFoundError`` if no layer has it. Identity and prefix
        are resolved here on first call.

        Write mode (``for_write=True``): returns the personal-layer URL
        unconditionally, without probing.

        Per-step-root atomicity: if ``path`` has a recognized step-root
        first segment (the prefix up to and including ``{name}-{hash}/``),
        the resolved user is cached on the instance for the duration of
        the call chain so a single step's reads stay within one user's dir.
        Cache TTL is per-instance lifetime; constructing a new
        ``UserSharedFS()`` clears it.
        """

    def to_gs_url(self, path: str) -> str:
        """Resolve a ``users://`` path to its concrete ``gs://`` URL.

        Read-mode resolution (probes layers). Public so that callers needing
        to feed the resolved URL into scheme-aware utilities (e.g. the
        executor's cross-region guard at ``executor.py:208``, or
        ``CrossRegionGuardedFS``) can do so explicitly. The cross-region
        guard inside Marin will be updated to call this helper for any
        ``users://`` URL before scheme-matching.
        """

    # fsspec overrides — all forward through _resolve_path with the
    # appropriate for_write flag and then delegate to the underlying
    # filesystem (gcsfs in production, memory:// in tests).

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any: ...
    def _info(self, path: str, **kwargs: Any) -> dict[str, Any]: ...
    def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]: ...
    def _exists(self, path: str, **kwargs: Any) -> bool: ...
    def _rm(self, path: str, recursive: bool = False, **kwargs: Any) -> None: ...
    def _makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None: ...


fsspec.register_implementation("users", UserSharedFS)
```

### Layer resolution table

| Operation | Probe / target |
|---|---|
| `_open(path, "rb")`, `_info`, `_ls`, `_exists` | (1) `{prefix}/users/{me}/{path}` → (2) glob `{prefix}/users/*/{step_root}/.executor_status` for `SUCCESS`, then take that user's `{path}` → (3) `{prefix}/{path}` |
| `_open(path, "wb"\|"ab"\|"r+")`, `_makedirs`, `_rm` | `{prefix}/users/{me}/{path}` (unconditional) |
| Read miss across all three layers | Raise `FileNotFoundError` |
| Glob in layer 2 returns >1 user with `SUCCESS` status | Raise `AmbiguousCrossUserResolutionError` (see Errors) |
| Layer-2 candidate user has `.executor_status` ≠ `SUCCESS` (RUNNING, FAILED, missing) | Skip that user; continue scanning |

The layer-2 glob keys on the step-root's `.executor_status` content, *not* on bare path existence. A teammate's RUNNING or FAILED run does not satisfy the probe; we want only complete artifacts. The step root is `{first_segment_after_users://}` — for `users://train-llama-1b_a1b2c3d4/checkpoints/step_1000`, the step root is `train-llama-1b_a1b2c3d4`. Once a user is selected for a given step root, all subsequent reads under that step root resolve to the same user's dir for the lifetime of the `UserSharedFS` instance (per-step-root atomicity, see `_resolve_path` docstring).

### URL grammar

```
users://<path>
```

`<path>` is one or more `/`-separated non-empty segments. Empty path (`users://`) is invalid and raises `ValueError`. There is no scheme authority — fsspec treats the whole string after `users://` as the path.

Valid examples:
- `users://checkpoints/train-llama-1b_a1b2c3d4/.executor_status`
- `users://checkpoints/train-llama-1b_a1b2c3d4/checkpoints/step_1000/`

Invalid (raise `ValueError`):
- `users://`
- `users://?query`

### Behavioural guarantees

- **Read-resolve-then-open is atomic per call.** A read call resolves the layer once and opens that concrete URL. If the file disappears between resolve and open (e.g. a teammate `gsutil rm`s mid-call), the underlying GCS open raises `FileNotFoundError`; the call does *not* re-probe.
- **Writes never escape the personal layer.** Writing to `users://x/y` always touches `{prefix}/users/{me}/x/y` even if reads of the same URL would resolve to a teammate's dir.
- **Identity is resolved at most once per instance.** The resolved username is cached on the instance after first read/write. Constructing a new `UserSharedFS()` re-resolves. URL parsing (`_strip_protocol`) and `__init__` do *not* trigger identity resolution; only the first `_open`/`_info`/`_ls`/`_exists`/`_rm`/`_makedirs` call does.
- **Per-step-root atomicity for layer 2.** Once layer 2 selects a user for step root `S`, all subsequent reads of `users://S/...` resolve to that user's dir for the instance's lifetime. Without this guarantee, two files in the same logical step output could resolve to different users (alice's `.executor_status` + bob's `checkpoints/step_1000/`).
- **`_ls` does not merge across layers.** It returns the listing of whichever layer hit on probe. Merged listings are explicitly out of scope; reporting/inventory tooling enumerates `users/*/` via the raw `gcsfs`/`gs://` interface, not through `users://`.
- **Cross-user resolution is logged.** Each `_resolve_path` call that resolves to layer 2 (another user) or layer 3 (shared root) emits one `INFO`-level log line with the step root and resolved user. Layer-1 (own personal) resolutions are silent. The executor's job-start preamble can opt to log a summary count.

## Identity resolver

```python
def resolve_marin_user(override: str | None = None) -> str:
    """Resolve the canonical marin user identifier.

    Resolution order:
        1. ``override`` argument (if non-None, non-empty after strip).
        2. ``MARIN_USER`` environment variable (if set, non-empty after strip).
        3. ``iris.cluster.client.job_info.get_job_info().user`` if not None.
        4. ``getpass.getuser()``.

    Returns the resolved username (no normalization, no allow/deny list).

    Raises:
        UserIdentityResolutionError: If no override is passed, ``MARIN_USER``
            is unset, ``get_job_info()`` returns None, and ``getpass.getuser()``
            raises ``OSError`` (the stripped-container case with no
            USER/LOGNAME env vars and a UID not in /etc/passwd).
    """
```

The `MARIN_USER` rung is added specifically for service-account/CI contexts where `getpass.getuser()` returns a shared generic name (`runner`, `nobody`); setting `MARIN_USER` lets those contexts route to a meaningful prefix without code changes.

Location: `lib/rigging/src/rigging/filesystem.py` (same module as `UserSharedFS`).

## Errors

```python
class UserIdentityResolutionError(RuntimeError):
    """Raised when ``resolve_marin_user`` exhausts the resolution chain.

    Triggered only when (a) no explicit override is passed, (b) ``MARIN_USER``
    is unset, (c) ``get_job_info()`` returns None (not running inside an Iris
    job context), and (d) ``getpass.getuser()`` raises ``OSError``. The
    original ``OSError`` is attached via ``__cause__``.
    """


class AmbiguousCrossUserResolutionError(RuntimeError):
    """Raised when the layer-2 glob finds SUCCESS for more than one user.

    Triggered when ``users/*/{step_root}/.executor_status`` yields multiple
    users whose status content reads ``SUCCESS``. Under normal operation only
    one user runs a given hashed step (the second user's cache check
    short-circuits before they write), so multiple SUCCESS indicates either a
    write race (two users started concurrently without seeing each other's
    output) or manual ``override_output_path`` values that landed in distinct
    ``users/`` dirs.

    The exception message lists the conflicting paths. Resolution requires
    explicit user action: ``gsutil rm -r`` all but one of the dirs, bump the
    step hash, or pass an ``override_output_path`` that bypasses ``users://``.
    The filesystem deliberately does *not* pick a winner — picking would mask
    a coordination problem that should surface.
    """
```

No other new error types. Existing fsspec error semantics (`FileNotFoundError`, `IsADirectoryError`, `PermissionError`) are forwarded unchanged from the underlying filesystem.

## Persisted directory layout

```
gs://marin-{region}/
├── users/                                          # NEW top-level dir
│   ├── alice/
│   │   └── checkpoints/
│   │       └── train-llama-1b_a1b2c3d4/
│   │           ├── .executor_status
│   │           └── checkpoints/step_1000/...
│   └── bob/
│       └── ...
├── checkpoints/                                    # legacy + manually-shared
│   └── ...
├── experiments/                                    # unchanged
├── raw/                                            # unchanged
└── tmp/ttl=Nd/                                     # unchanged
```

The `users/` top-level dir is the only addition. No existing paths move. No new bucket-level configuration.

## Wiring in `experiments/defaults.py`

Three call sites change. Each gains a new keyword argument `output_path_prefix: str | None = "users://"`, which is forwarded into the underlying `ExecutorStep` / `compute_output_path` call. Callers opt out by passing `output_path_prefix=None` (preserving the existing `MARIN_PREFIX`-based behavior).

```python
def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    override_output_path: str | None = None,
    output_path_prefix: str | None = "users://",        # NEW
) -> ExecutorStep: ...


def default_dpo(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    dpo_config: SimpleDPOConfig,
    tags: Sequence[str] = (),
    override_output_path: str | None = None,
    output_path_prefix: str | None = "users://",        # NEW
) -> ExecutorStep: ...


def resolve_lm_train_config(
    name: str,
    raw_config: TrainLmConfig,
    override_output_path: str | None,
    resources: ResourceConfig,
    output_path_prefix: str | None = "users://",        # NEW
) -> TrainLmConfig: ...
```

Inside each function:
- `default_train` and `default_dpo` pass `output_path_prefix=output_path_prefix` to the `ExecutorStep(...)` constructor — this is the *new* field defined in the "Executor changes" section above. It is *not* `override_output_path`, which is a full-path override; `output_path_prefix` overrides only the prefix segment and preserves hash-keyed `{name}-{hash}`.
- `resolve_lm_train_config` passes it as the new `output_path_prefix` kwarg to `compute_output_path(name, raw_config, override_output_path=override_output_path, output_path_prefix=output_path_prefix)`.

`default_sft` (`experiments/defaults.py:749`) and `simulated_epoching_train` (line 323) inherit transitively through `default_train` — no changes to their signatures.

## File paths

| Component | Path |
|---|---|
| `UserSharedFS` class | `lib/rigging/src/rigging/filesystem.py` (alongside `MirrorFileSystem`) |
| `resolve_marin_user` function | `lib/rigging/src/rigging/filesystem.py` |
| `UserIdentityResolutionError` | `lib/rigging/src/rigging/filesystem.py` |
| Protocol registration | `lib/rigging/src/rigging/filesystem.py` (module-bottom `fsspec.register_implementation`) |
| Unit tests | `lib/rigging/tests/test_users_filesystem.py` (new file, sibling to `test_mirror_fs.py`) |
| `ExecutorStep.output_path_prefix` field | `lib/marin/src/marin/execution/executor.py` (dataclass at line 691) |
| `Executor.compute_version` one-line change | `lib/marin/src/marin/execution/executor.py:1504` |
| `step_spec.py` slash-count fix | `lib/marin/src/marin/execution/step_spec.py:114,116` |
| `compute_output_path` signature update | `lib/marin/src/marin/execution/executor.py:1778` |
| Cross-region guard update (call `to_gs_url` for `users://`) | `lib/marin/src/marin/execution/executor.py:208` (`_infer_gcs_regions`) |
| Training factory wiring | `experiments/defaults.py` (three signature additions, see above) |

## Out of scope

The following are intentionally not in this spec; reviewers should not push back on their absence:

- Any promotion mechanism (`marin promote` CLI, `gsutil cp` helper, automatic copy-on-completion). Cross-user reuse is implicit via layer 2.
- Lifecycle rules on `users/{user}/` in `infra/configure_buckets.py`. No auto-deletion.
- Hard quota enforcement, write-path interception, or per-user `TransferBudget` analog.
- Routing of non-training executor outputs (datasets, evals, ferries, scratch) through `users://`. Each subsystem opts in by constructing `users://` URLs; none do in v1.
- A `scope` field on `StepSpec` or `ExecutorStep`. Routing is at URL-construction time only.
- Merged-listing semantics for `_ls` (returning union of all layers). Single-layer-resolution only.
- Per-user usage reporting tooling (extension of `egress_report.py`). Follow-up PR.
- `MirrorFileSystem` refactor onto a shared `LayeredReadFS` base.
- Cross-region behavior for `users://`. The filesystem operates on a single regional bucket as determined by `marin_prefix()`; cross-region access requires `mirror://`, which can wrap `users://`-resolved URLs but is not composed automatically.
