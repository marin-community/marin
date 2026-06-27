# buoy — spec

Concrete contracts for the `buoy` design. Pins the public surface (package layout, Python signatures, persisted shapes, HTTP routes, errors, Iris wiring). Not an implementation plan.

Throughout, `run_key = f"{entity}/{project}/{run_id}"` (slashes intended; used as both the cache sub-path and the local-disk sub-path).

## Package layout

```
lib/buoy/
  pyproject.toml                 # name "buoy"; deps: iris, rigging, wandb, starlette,
                                 #   uvicorn, httpx, pyarrow, gcsfs/fsspec; xprof binary at runtime
  src/buoy/
    __init__.py
    config.py                    # BuoyConfig
    mirror.py                    # mirror_run, RunMirror, Manifest, ProfileRef, RunSummary, history normalization, layout helpers
    server.py                    # create_app(config) -> Starlette; MirrorRegistry; route handlers
    xprof.py                     # XprofManager (subprocess lifecycle + reverse proxy)
    main.py                      # CLI entrypoint: `buoy serve --host --port`
    dashboard/                   # Vue SPA; `npm run build` -> dashboard/dist/
  deploy/
    Dockerfile
    k8s/
      01-deployment.yaml.tmpl    # replicas: 1, strategy Recreate, ephemeral-storage limits, emptyDir volume
      02-service.yaml.tmpl       # ClusterIP, finelog-style
  tests/
    test_mirror.py
    test_server.py
```

`buoy` does **not** import `marin`. Dependency direction: `buoy → {iris, rigging, wandb}`; nothing in the repo imports `buoy`. (The `jax_profile` artifact-type filter mirrors `marin/profiling/ingest.py`'s `PROFILE_ARTIFACT_TYPE = "jax_profile"` but is re-stated locally to keep buoy self-contained.)

## Config

```python
@dataclass(frozen=True)
class BuoyConfig:
    host: str                       # bind host, e.g. "0.0.0.0"
    port: int                       # single public port
    wandb_api_key: str              # from env WANDB_API_KEY; never logged
    default_entity: str             # surfaced as the picker default; entity is still per-request
    cache_base: str                 # gs://.../tmp/ttl=30d/buoy  (marin_temp_bucket(30, "buoy"))
    local_cache_dir: str            # ephemeral disk root for xprof logdirs
    xprof_idle_timeout: float       # seconds since last proxied request before an xprof subprocess is reaped
    max_xprof_procs: int            # cap on concurrent xprof subprocesses (eviction by last-request time)

    @staticmethod
    def from_env() -> "BuoyConfig": ...
        # resolves WANDB_API_KEY, WANDB_DEFAULT_ENTITY, cache_base via marin_temp_bucket(30, "buoy");
        # raises ValueError (fail fast) if WANDB_API_KEY is unset.
```

`local_cache_dir` capacity must be `>= max_xprof_procs * largest_expected_logdir` (profiles run ~275 MB); the k8s volume is sized to match.

## Mirror layer — `buoy.mirror`

```python
@dataclass(frozen=True)
class RunSummary:
    """One row in the run picker. Cheap; from wandb.Api list, no download."""
    entity: str
    project: str
    run_id: str
    display_name: str
    state: str                      # wandb run state verbatim: "running"|"finished"|"crashed"|"failed"|"killed"
    created_at: str                 # ISO-8601 UTC
    url: str                        # canonical wandb run URL

@dataclass(frozen=True)
class ProfileRef:
    """A mirrored jax_profile artifact, ready to hand to `xprof --logdir`."""
    artifact_name: str              # e.g. "...-profiler:v0"
    version: int                    # artifact version, for deterministic "latest" selection
    logdir: str                     # GCS path to the artifact root (contains plugins/profile/<ts>/<host>.xplane.pb)
    size_bytes: int

@dataclass(frozen=True)
class Manifest:
    """Written to manifest.json LAST, as the commit marker. Its presence == the run is fully cached."""
    entity: str
    project: str
    run_id: str
    display_name: str
    state: str                      # run state at mirror time; drives re-fetch-on-view
    created_at: str                 # ISO-8601 UTC
    mirrored_at: str                # ISO-8601 UTC, when this mirror was written
    url: str
    files: list[str]               # cache-relative paths written (excl. manifest.json); read can verify presence
    metric_keys: list[str]          # scalar columns available in history.parquet
    history_source: str             # "exports" (download_history_exports) | "scan" (scan_history fallback)
    artifacts: list[str]            # artifact names mirrored under artifacts/
    profiles: list[ProfileRef]      # all jax_profile artifacts, newest version first; [] if none

@dataclass(frozen=True)
class RunMirror:
    root: str                       # gs://.../{entity}/{project}/{run_id}/
    manifest: Manifest

def run_root(cache_base: str, entity: str, project: str, run_id: str) -> str:
    """Pure path join: f'{cache_base}/{run_key}'. No I/O."""

def list_runs(
    api: "wandb.Api",
    entity: str,
    project: str,
    *,
    limit: int = 50,
    cursor: str | None = None,
    search: str | None = None,
) -> tuple[list[RunSummary], str | None]:
    """List runs for the picker, newest-created first; returns (rows, next_cursor).
    Backed by api.runs(path, per_page=, order='-created_at', filters=); `search` matches display_name/run_id.
    Bounded by `limit` so a multi-thousand-run project never loads wholesale (avoids the 30s proxy timeout)."""

def normalize_history(frames: "list[pa.Table] | Iterable[dict]", source: str) -> "pa.Table":
    """Normalize wandb history into one Arrow table for history.parquet.

    Policy (deterministic, testable):
      - Concatenate all export frames / scan rows.
      - Key column is wandb '_step' (int). Rows sharing a _step are merged (last non-null wins).
      - Outer-union columns across frames; missing cells are null.
      - Each metric column is coerced to a single nullable type: numeric -> float64; if a column holds
        non-numeric/non-scalar values (dict, media ref, string) it is DROPPED and omitted from metric_keys.
      - Sort ascending by _step.
    Returns a table whose columns are ['_step', *metric_keys]."""

def mirror_run(
    entity: str,
    project: str,
    run_id: str,
    *,
    api: "wandb.Api",
    cache_base: str,
    fs: "fsspec.AbstractFileSystem",
    refresh: bool = False,
) -> RunMirror:
    """Pull all of a run's data into the GCS cache and return its RunMirror. Blocking worker; the server
    runs it in a background task (never inline in a request) — see MirrorRegistry.

    Idempotent + commit-safe:
      - If manifest.json exists at run_root, its `state` was terminal, and refresh is False: return the
        existing mirror without re-downloading.
      - Otherwise write all payload objects first, then write manifest.json LAST as the atomic commit
        marker. A crash mid-write leaves no manifest -> readers treat the run as not cached.
      - Re-downloads when refresh=True or the cached run was still "running".

    History: calls `run.download_history_exports(dir, require_complete_history=False)`; if it returns
    paths (history_source="exports"). In practice recent runs return empty paths + contains_live_data=True
    (wandb exports lazily), so it falls through to `run.scan_history()` (history_source="scan"), which is
    the common path. Never the sampled `run.history()`. Result -> `normalize_history` -> history.parquet.

    Profiles: scans `run.logged_artifacts()` for type == "jax_profile", downloads each into
    `artifacts/<name>/`, records them in `manifest.profiles` newest-version-first (logdir = artifact root).

    Concurrency: the server holds a per-run_key asyncio.Lock so concurrent mirrors of one run coalesce;
    cross-pod concurrency is excluded by single-replica deployment (replicas:1, Recreate).

    Raises RunNotFoundError if the run does not exist or the key cannot read it.
    """
```

### GCS cache layout (written by `mirror_run`)

```
{cache_base}/{entity}/{project}/{run_id}/
  config.json            # run.config verbatim                (written before manifest)
  summary.json           # run.summary verbatim               (written before manifest)
  metadata.json          # run.metadata (wandb-metadata.json) (written before manifest)
  history.parquet        # normalized: column "_step" + one float64 column per scalar metric
  files/                 # run.files() (uploaded source/code, requirements, diff patch, etc.)
    ...
  artifacts/
    <artifact_name>/      # one dir per mirrored artifact, e.g. "...-profiler:v0"
      plugins/profile/<ts>/<host>.xplane.pb   # for jax_profile artifacts
  manifest.json          # Manifest; WRITTEN LAST = commit marker
```

`manifest.json` example:

```json
{
  "entity": "marin-community",
  "project": "marin_moe",
  "run_id": "GM2560-MAY-...-cw-20260627-021250",
  "display_name": "GM2560-MAY-...-cw-20260627-021250",
  "state": "finished",
  "created_at": "2026-06-27T02:12:50Z",
  "mirrored_at": "2026-06-27T18:40:11Z",
  "url": "https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-...",
  "files": ["config.json", "summary.json", "metadata.json", "history.parquet",
            "artifacts/GM2560-MAY-...-profiler:v0/plugins/profile/.../host.xplane.pb"],
  "metric_keys": ["train/loss", "train/lr", "throughput/tokens_per_sec"],
  "history_source": "exports",
  "artifacts": ["GM2560-MAY-...-profiler:v0"],
  "profiles": [{
    "artifact_name": "GM2560-MAY-...-profiler:v0",
    "version": 0,
    "logdir": "gs://marin-us-central1/tmp/ttl=30d/buoy/marin-community/marin_moe/GM2560-.../artifacts/GM2560-MAY-...-profiler:v0",
    "size_bytes": 288358400
  }]
}
```

## Serve layer — `buoy.server`

```python
class MirrorStatus(StrEnum):
    PENDING = "pending"; RUNNING = "running"; DONE = "done"; FAILED = "failed"

@dataclass
class MirrorState:
    status: MirrorStatus
    manifest: Manifest | None       # set when DONE
    error: str | None               # set when FAILED

class MirrorRegistry:
    """In-memory map run_key -> MirrorState + per-run asyncio.Lock. Runs mirror_run in a background task
    (threadpool) so requests return inside the 30s proxy budget. A duplicate POST for an in-flight run
    returns the existing PENDING/RUNNING state instead of starting a second mirror."""
    def start(self, entity: str, project: str, run_id: str, *, refresh: bool) -> MirrorState: ...
    def get(self, run_key: str) -> MirrorState | None: ...

def create_app(config: BuoyConfig) -> "starlette.applications.Starlette":
    """Build the ASGI app: SPA shell + static assets + JSON API + xprof proxy. Owns one wandb.Api(),
    one MirrorRegistry, one XprofManager, and an fsspec handle to cache_base. Starlette root_path is set
    from X-Forwarded-Prefix so SPA asset links resolve under /proxy/system.buoy."""
```

### HTTP routes

`{entity}/{project}/{run_id}` segments are URL-encoded path params. JSON unless noted. In production these sit under the Iris proxy prefix `/proxy/system.buoy/` (paths below are app-relative; the SPA builds links from `root_path`).

| Method | Path | Params | Response |
|---|---|---|---|
| GET | `/` | — | SPA `index.html` |
| GET | `/assets/{path}` | — | static asset |
| GET | `/api/runs` | `entity`, `project`, `limit?`, `cursor?`, `search?` | `{ "runs": [RunSummary], "next_cursor": str\|null }` |
| POST | `/api/mirror` | body `{entity, project, run_id, refresh?}` | **`202`** `{ "run_key": str, "status": MirrorStatus }` — starts/*joins* a background mirror |
| GET | `/api/runs/{entity}/{project}/{run_id}/mirror_status` | — | `{ "status": MirrorStatus, "manifest"?: Manifest, "error"?: str }` (SPA polls until `done`) |
| GET | `/api/runs/{entity}/{project}/{run_id}/manifest` | — | `Manifest`; 404 if not cached. If cached `state == "running"`, also kicks a background refresh (returns current). |
| GET | `/api/runs/{entity}/{project}/{run_id}/metrics` | `keys` (comma-sep; omit = all) | `{ "step": [int], "metrics": { "<key>": [float\|null] } }` from `history.parquet` |
| GET | `/api/runs/{entity}/{project}/{run_id}/config` | — | run config JSON |
| GET | `/api/runs/{entity}/{project}/{run_id}/summary` | — | run summary JSON |
| GET | `/api/runs/{entity}/{project}/{run_id}/metadata` | — | run metadata JSON |
| GET | `/xprof/{entity}/{project}/{run_id}/{sub_path:path}` | `profile?` (artifact name; default = latest version) | reverse-proxied xprof response (HTML/JS/JSON); 409 if the run has no profile; 202 while the logdir is still downloading |

**View protocol (normative).** Opening a run in the SPA MUST `POST /api/mirror` and poll `mirror_status` before reading; this is what realizes re-fetch-on-view for `running` runs. `GET /manifest`'s server-side refresh is a backstop, not the trigger. Only `POST /api/mirror`, `GET /api/runs`, and the live refresh touch wandb; all other reads hit the GCS cache.

## xprof process manager — `buoy.xprof`

```python
class XprofManager:
    def __init__(self, local_cache_dir: str, max_procs: int, idle_timeout: float): ...

    async def ensure(self, run_key: str, logdir_gcs: str) -> int | None:
        """Ensure an xprof subprocess is serving `run_key`'s logdir; return its localhost port, or None
        while the one-time GCS->local download is still in flight (caller returns 202).

        Downloads logdir_gcs to {local_cache_dir}/{run_key}/ on first use (xprof reads the LOCAL copy,
        never gs://). Launches `xprof --logdir {local} --port {free}` (standalone xprof has no
        --path_prefix and needs none — see proxy()). Reuses a live process. Raises XprofLaunchError if
        the binary fails to bind."""

    async def proxy(self, run_key: str, sub_path: str, request) -> "starlette.responses.Response":
        """Reverse-proxy sub_path (+ query) to this run's xprof subprocess via httpx, streaming back the
        body UNCHANGED (no URL rewriting). xprof emits relative asset URLs + a JS <base href> computed
        from document.location.pathname, so serving it under /proxy/system.buoy/xprof/{run_key}/ self-
        prefixes correctly. The SPA's iframe src MUST be `.../xprof/{run_key}/data/plugin/profile/` so the
        base-href logic triggers. Strips content-encoding/length/transfer-encoding hop headers. Calls
        ensure() first; updates last_request_at (pins the process against eviction)."""

    def evict(self) -> None:
        """Eviction is by last-proxied-request time: terminate processes idle longer than idle_timeout.
        A process with an in-flight request or touched within idle_timeout is never killed. If a new
        ensure() hits max_procs with all processes recently active, it raises (server -> 503) rather than
        killing an active session. Run periodically."""
```

## Errors — `buoy.mirror` / `buoy.server`

```python
class BuoyError(Exception): ...
class RunNotFoundError(BuoyError): ...          # run missing or key lacks access -> HTTP 404
class ProfileNotAvailableError(BuoyError): ...  # run has no jax_profile artifact -> HTTP 409 on /xprof
class XprofLaunchError(BuoyError): ...          # xprof subprocess failed to start -> HTTP 502
class XprofCapacityError(BuoyError): ...        # max_xprof_procs hit, all active -> HTTP 503
```

`wandb`'s `IncompleteRunHistoryError` is caught inside `mirror_run` (triggers the `scan_history` fallback), never surfaced. All other wandb/network errors propagate; the Starlette handler maps `BuoyError` subclasses to the codes above and everything else to 500. Mirror failures surface via `mirror_status` (`FAILED` + `error`), not an HTTP error on the original `202`.

## Iris registration

Primary (finelog model):

- `deploy/k8s/02-service.yaml.tmpl`: a `ClusterIP` Service named `buoy` in namespace `iris`, selecting the buoy Deployment, exposing `config.port`.
- `deploy/k8s/01-deployment.yaml.tmpl`: `replicas: 1`, `strategy: Recreate` (single writer — see mirror concurrency), an `emptyDir` volume (`sizeLimit` ≈ `max_xprof_procs × ~300Mi`) mounted at `local_cache_dir`, and explicit `ephemeral-storage` request/limit so xprof downloads don't trip pod eviction.
- Cluster config gains an `endpoints` entry resolved at controller startup:
  ```yaml
  endpoints:
    /system/buoy:
      uri: k8s://buoy.iris
      metadata:
        port: "<port>"
  ```
- Reached at `https://<iris-host>/proxy/system.buoy/`. No path-prefix config is needed: xprof self-prefixes from the request path (verified), so buoy is agnostic to the exact public prefix.

Entrypoint:

```
buoy serve --host 0.0.0.0 --port <port>      # remaining config from BuoyConfig.from_env()
```

## Out of scope

- Multi-run comparison / overlaying metrics from several runs.
- Background mirroring of an entire entity/project (buoy is on-demand, single-run).
- Auth beyond the Iris controller proxy. (The proxy strips `Authorization` upstream, so per-user authz would require buoy's own session layer — explicitly not built.)
- Writing back to wandb, editing/deleting runs, or mutating artifacts.
- Profile formats other than `jax_profile` xplane (e.g. raw `perfetto`/`trace.json`-only artifacts).
- `used_artifacts()` mirroring (logged artifacts only by default — see design Open Questions).
- Cross-region cache placement / nearest-bucket optimization (single regional bucket).
- Durable archival — the cache is TTL-cleaned; wandb remains source of truth.
- Multi-replica / cross-pod mirror coordination (single replica by design).
```
