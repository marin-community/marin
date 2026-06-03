# Feasibility: embedding the Rust `finelog-server` for Iris's in-process log server

**Status:** feasibility report only — no code changed.
**Date:** 2026-06-03
**Author:** investigation pass (agent)

## TL;DR / Recommendation

The blocker to deleting the Python `finelog.store` / `finelog.server` is that the Iris
controller embeds the Python log+stats server in-process (`_start_local_log_server` in
`lib/iris/src/iris/cluster/controller/controller.py`). The question posed was whether
`finelog` should ship a **PyO3 extension** that starts/stops the Rust axum server inside the
Python process so Iris keeps its current shape.

**Recommendation: do NOT use PyO3. Ship a client-managed _subprocess_ launcher inside
`finelog.client` (an `EmbeddedLogServer` context manager / `LogClient.start_embedded(...)`).**

Reasons, in priority order:

1. **The transport is HTTP regardless.** The controller already talks to its embedded server
   exclusively over localhost HTTP via `LogServiceProxy` / `StatsServiceProxy` and
   `LogClient.connect(...)` — never via in-process Python calls. PyO3-in-process and a
   subprocess present the controller with the *identical* interface: a base URL. PyO3 buys
   **zero** latency or coupling advantage because no call is ever in-process today.
2. **A subprocess is strictly more robust.** Crash isolation, graceful SIGTERM shutdown, and
   memory accounting are all things the standalone Rust binary *already* implements
   (`main.rs::shutdown_signal`, `store.shutdown`). Running tokio + axum + DataFusion inside the
   CPython process means a Rust panic or OOM takes down the controller; a subprocess does not.
3. **The subprocess harness already exists and is proven.** `lib/finelog/tests/parity/conftest.py`
   spawns `finelog-server` on a free port, waits on `/health`, and terminates it on teardown —
   exactly the lifecycle `EmbeddedLogServer` needs. We would lift ~40 lines of battle-tested
   code into `finelog.client`.
4. **Packaging a PyO3 cdylib for the axum server is materially harder** than shipping a binary:
   tokio runtime lifecycle under the GIL, abi3 vs. a heavy native dep tree (DataFusion, arrow,
   parquet, object_store, rusqlite), and a maturin wheel that has to be built per-platform in CI.
   A binary is one `cargo build` artifact already produced by the finelog Docker image.

The one real cost of the subprocess approach: the `finelog` Python wheel must be able to **find a
`finelog-server` binary at runtime**. That is a solved problem (env override + a couple of search
paths, same as the parity harness), and Iris's deploy images already have a Rust toolchain or a
build stage available.

Rough effort: **~1–1.5 days** for the `EmbeddedLogServer` + Iris swap + test migration, versus
**~4–6 days + ongoing CI wheel maintenance** for PyO3. See §5.

---

## 1. Current Iris embedding — precise read

### 1.1 What gets constructed, where, and on what transport

`Controller.__init__` (`controller.py:333–367`) sets up the log service. The decision is binary:

```python
if config.log_service_address:                 # production: external finelog-server
    self._log_service_address = config.log_service_address
else:                                           # tests + local mode: in-process fallback
    self._log_service_address = self._start_local_log_server()
```

`_start_local_log_server` (`controller.py:486–531`) does **all** of the Python-server embedding:

- `find_free_port()`
- builds an **in-memory** `DuckDBLogStore(log_dir=None, …)` (no segmentation, no flush thread, no
  compaction; logs lost on restart — explicitly a fallback, per its own docstring),
- wraps it in `LogServiceImpl` + `StatsServiceImpl`,
- `build_log_server_asgi(...)` → an ASGI app,
- runs it on a **uvicorn thread** via `self._threads.spawn_server(self._log_server, name="log-server")`,
- waits for `self._log_server.started`,
- returns `http://{external_host}:{port}`.

**Crucial:** after this, *every* access is over localhost HTTP. The controller immediately builds:

```python
self._remote_log_service  = LogServiceProxy(self._log_service_address, …)   # HTTP client
self._remote_stats_service = StatsServiceProxy(self._log_service_address, …) # HTTP client
self._log_client           = LogClient.connect(self._log_service_address, …) # HTTP client
```

`LogServiceProxy` / `StatsServiceProxy` (`finelog/client/proxy.py`) wrap the **sync ConnectRPC
clients** (`LogServiceClientSync`, `StatsServiceClientSync`) and dispatch each call via
`asyncio.to_thread`. They are HTTP, not in-process. So *the data plane is already a network
boundary*, whether the server is a Python uvicorn thread or any other process.

### 1.2 What the dashboard mounts

`ControllerDashboard` (`dashboard.py`) receives the two **proxies** (not the impls):

- `log_service=self._remote_log_service` (a `LogServiceProxy`)
- `finelog_stats_service=self._remote_stats_service` (a `StatsServiceProxy`)

In `_create_app` it mounts:

- `LogServiceASGIApplication(service=self._log_service, …)` at `/finelog.logging.LogService` **and**
  at the legacy `/iris.logging.LogService` (`dashboard.py:520–521`). This is a **forwarding proxy**:
  old workers cached `/system/log-server` → controller URL, so the controller must still accept
  their `PushLogs`/`FetchLogs` and forward them over RPC to the real server. The proxy's
  `push_logs`/`fetch_logs` simply re-issue the call to the remote.
- `FinelogStatsServiceASGIApplication(service=self._finelog_stats_service, …)` (when present) at the
  finelog stats path (`dashboard.py:525–531`).

The dashboard's *only* hard dependency on the Python server package is the **type annotation**
`log_service: LogServiceImpl | LogServiceProxy` and the `from finelog.server import LogServiceImpl`
import at `dashboard.py:38`. At runtime it is always handed a `LogServiceProxy`. The `LogServiceImpl`
arm of the union is exercised only by Iris **tests** (see §1.4).

### 1.3 Production already uses an external server

The in-process server is a **fallback**, not the production path:

- `controller/main.py::_resolve_cluster_endpoints` derives `/system/log-server` from
  `cluster_config.log_server_config` (`marin.yaml: log_server_config: marin`,
  `marin-dev.yaml: log_server_config: marin-dev`).
- That address is passed as `ControllerConfig.log_service_address`, so production controllers take
  the `if config.log_service_address:` branch and **never** call `_start_local_log_server`.
- The production finelog server is a separate k8s Deployment
  (`lib/finelog/deploy/k8s/02-deployment.yaml.tmpl`) running the **Rust** `finelog-server`
  Docker image (`lib/finelog/deploy/Dockerfile`).

So the embedded server matters for exactly two audiences: **(a) Iris unit tests** and **(b)
local / single-host `iris` runs** that don't declare an external endpoint.

### 1.4 The harder coupling: 9 test files use `LogServiceImpl` directly (not over RPC)

This is the part that actually constrains the design. The 9 test files do **not** go through
`_start_local_log_server` or any HTTP transport — they construct `LogServiceImpl()` directly and
either pass it straight to `ControllerDashboard(log_service=…)` or wrap it in an in-process fake:

| File | Usage |
| --- | --- |
| `lib/iris/tests/cluster/conftest.py` | `_FakeLogClientFromService(LogServiceImpl())` — `asyncio.run(svc.fetch_logs(...))` directly |
| `lib/iris/tests/cluster/controller/conftest.py` | `log_service` fixture builds `LogServiceImpl(log_dir=…)`, monkeypatches `push_logs`/`fetch_logs` to force-flush |
| `lib/iris/tests/cluster/controller/test_dashboard.py` | `ControllerDashboard(service, log_service=LogServiceImpl())` |
| `lib/iris/tests/cluster/controller/test_auth.py` | same, plus auth variants |
| `lib/iris/tests/cluster/controller/test_api_keys.py` | `fake_log_client_from_service(LogServiceImpl())` |
| `lib/iris/tests/cluster/controller/test_service.py` | `fake_log_client_from_service(LogServiceImpl())` |
| `lib/iris/tests/test_budget.py` | `fake_log_client_from_service(LogServiceImpl())` |
| `lib/iris/tests/cluster/providers/k8s/conftest.py` | `InProcessLogClient(LogServiceImpl())` — calls `push_logs`/`fetch_logs` directly |
| `lib/iris/tests/cluster/providers/k8s/test_provider.py` | `asyncio.run(log_service.fetch_logs(...))` directly |

These tests deliberately avoid the network: they call the Python service's async methods inline,
and the controller-test conftest even monkeypatches `LogServiceImpl.push_logs` to bypass the
flush-interval wait. **No PyO3 or subprocess scheme helps these tests** — they want a synchronous,
in-process, force-flushable Python log service. They are the strongest argument that the Python
`LogServiceImpl` / `DuckDBLogStore` cannot be deleted purely by changing the production embedding;
the test surface must be migrated separately (to the HTTP/RPC-over-subprocess pattern the parity
harness already uses, or to a thin in-process fake of the proxy interface).

### 1.5 Iris touch-points, ranked by disruptiveness

| # | Touch-point | Disruptiveness | Notes |
| --- | --- | --- | --- |
| 1 | `controller.py:486–531` `_start_local_log_server` (build DuckDBLogStore + LogServiceImpl + StatsServiceImpl + ASGI + uvicorn thread) | **Low to replace** | Self-contained method returning a URL. Swap the whole body for `EmbeddedLogServer` start; keep the return-a-URL contract. |
| 2 | `controller.py:22–25` imports of `LogServiceImpl`, `build_log_server_asgi`, `StatsServiceImpl`, `DuckDBLogStore` | **Low** | Deleted once #1 is swapped. `LogServiceProxy`/`StatsServiceProxy`/`LogClient` imports stay. |
| 3 | `controller.py:339,505,654–655` `self._log_service` attribute + `self._log_service.close()` in `stop()` | **Low** | Becomes the `EmbeddedLogServer` handle; `stop()` calls `handle.close()`. |
| 4 | `dashboard.py:38` `from finelog.server import LogServiceImpl` + type union `LogServiceImpl \| LogServiceProxy` | **Low** | Annotation-only; narrow to `LogServiceProxy` once tests stop passing the impl. |
| 5 | 9 Iris **test** files using `LogServiceImpl` directly (§1.4) | **Medium–High** | The real work. Not addressed by changing the production embedding. Migrate to either (a) a spawned `EmbeddedLogServer` + real `LogClient`, or (b) a small in-process fake conforming to the proxy Protocol. |
| 6 | `controller/main.py:105–111` warning text about the in-process MemStore fallback | **Trivial** | Reword if the fallback's persistence story changes (a subprocess can use a temp dir → gains persistence-within-process-life). |

Items 1–4 are a small, contained diff in `iris`. Item 5 is the bulk of the effort and is
**independent of PyO3-vs-subprocess** — it is about whether the tests keep a Python in-process
service at all.

---

## 2. PyO3 packaging feasibility (and why it is over-engineering here)

`dupekit` (`rust/dupekit/`) is the in-repo precedent for a PyO3 extension: `crate-type =
["cdylib"]`, `pyo3 = { features = ["extension-module", "abi3-py311"] }`, built via maturin
(`rust/dupekit/pyproject.toml` `build-backend = "maturin"`), published as `marin-dupekit` wheels by
`.github/workflows/dupekit-release-wheels.yaml` driving `rust/dupekit/build_package.py`. So PyO3 +
maturin is a known, working pattern in this repo. The question is whether it *fits the finelog
server*, and the answer is "technically yes, but it is the wrong tool."

### 2.1 What a minimal PyO3 surface would require

The crate currently declares both a `[[bin]]` (`finelog-server`) and a `[lib]` (`finelog`, the axum
app + store). The lib target already exposes `build_app(store, config)` and `Store::new(...)`. A
minimal extension would:

- add `crate-type = ["cdylib"]` (alongside the existing `rlib`) and a `pyo3` dependency,
- expose `start(port, log_dir, remote_log_dir) -> handle` and `stop(handle)`:
  - `start` creates a **tokio runtime** (`Runtime::new()`), `block_on(Store::new + bootstrap_maintenance)`,
    binds a `TcpListener`, spawns `axum::serve(...).with_graceful_shutdown(rx)` onto the runtime, and
    keeps the runtime + a shutdown `oneshot::Sender` alive in a `#[pyclass]` handle stored on a
    background OS thread (the runtime cannot run on the GIL thread).
  - `stop` fires the shutdown sender, joins the server task with a timeout, calls `store.shutdown`,
    then drops the runtime.

### 2.2 The hard parts (all real, none bought back by avoiding a subprocess)

- **Tokio runtime under the GIL.** The runtime must live on a non-GIL thread; `start` must release
  the GIL (`Python::allow_threads`) while blocking on bootstrap/bind. Mixing CPython's signal
  handling with tokio's — the binary installs SIGTERM/SIGINT handlers (`main.rs:117`) — must be
  *removed* for the in-process variant (CPython owns signals), so the in-process path diverges from
  the binary's shutdown path and needs its own oneshot-based shutdown. More code, separately tested.
- **Graceful shutdown / drain.** Has to be reimplemented against a `oneshot` instead of the signal
  future, and `store.shutdown(Duration)` joined within a bound so `LogClient.close()` on the Python
  side doesn't race a half-drained store. Subprocess gets this for free via SIGTERM → existing path.
- **Crash blast radius.** A `panic!` in a DataFusion query, an arrow OOM, or a rusqlite catalog
  corruption inside the extension can abort the **controller** process. The controller is the
  cluster control plane; this is a serious regression in fault isolation versus today's separate
  uvicorn thread (which at least can't segfault the interpreter) and versus a subprocess.
- **SPA dist discovery.** `spa.rs::vue_dist_dir()` resolves `dist` from `FINELOG_DASHBOARD_DIST`,
  then `CARGO_MANIFEST_DIR/../../lib/finelog/dashboard/dist`, then `/app/dashboard/dist`. Inside a
  pip-installed wheel, `CARGO_MANIFEST_DIR` is meaningless and `/app/...` won't exist, so the
  extension would need the dist bundled into the wheel and a wheel-relative resolution path added to
  `vue_dist_dir`. (A subprocess binary has the same dist-location concern, but the controller's own
  dashboard is the SPA users actually hit in-cluster; the embedded log server's SPA only matters for
  standalone finelog, which already ships the Docker layout.)
- **Wheel build matrix.** finelog's native dep tree is heavy (DataFusion 53, arrow/parquet 58,
  object_store + gcp, rusqlite bundled). A maturin wheel must compile all of it per platform
  (manylinux x86_64/aarch64, macOS) in CI — a new `finelog-release-wheels.yaml` mirroring the
  dupekit workflow, plus `build_package.py`. abi3 helps with Python-version fan-out but not
  platform fan-out. This is real, recurring CI surface.
- **Monorepo packaging.** `marin-finelog` is currently a **pure-Python hatchling** wheel
  (`lib/finelog/pyproject.toml`, `build-backend = "hatchling.build"`). Adding a native extension
  means either (a) converting `marin-finelog` to maturin (which then must still package all the
  pure-Python `finelog.client` / `finelog.store` code — maturin's `python-source` supports this but
  it's a build-system swap for a workspace member that iris/zephyr depend on), or (b) a *second*
  package `marin-finelog-native` that the first depends on. Both are more moving parts than "find a
  binary on PATH."

### 2.3 Verdict on PyO3

PyO3 is feasible but is **pure over-engineering for this problem**: the controller never makes an
in-process call into the log server, so the marquee PyO3 benefit (avoiding serialization / a socket
hop) does not apply. Every downside (runtime lifecycle, crash isolation loss, wheel matrix, build-
system swap) is incurred for no upside over a subprocess that exposes the same `http://127.0.0.1:port`.

---

## 3. PyO3 vs. managed subprocess — head to head

| Dimension | PyO3 in-process | Managed subprocess (recommended) |
| --- | --- | --- |
| **Latency to controller** | localhost HTTP (proxies/`LogClient` are HTTP) | localhost HTTP — **identical** |
| **In-process Python calls** | Not used today; would require a new Python-callable RPC surface to exploit | n/a — same HTTP |
| **Implementation effort** | High: tokio-under-GIL, oneshot shutdown, `#[pyclass]` handle, signal de-conflict, dist-in-wheel | Low: lift `conftest.py` Popen+`/health`+terminate into `finelog.client` |
| **Packaging** | maturin wheel + per-platform CI matrix + build-system swap or 2nd package | locate a `finelog-server` binary (env var + search paths); already-built artifact |
| **Crash isolation** | Rust panic/OOM kills the controller | Server crash is contained; controller observes a dead `/health` and can restart it |
| **Graceful shutdown** | Must reimplement against oneshot | SIGTERM → existing `main.rs` drain path, already correct |
| **Memory accounting** | Shares the controller's RSS; DataFusion arenas inside the control plane | Separate process RSS; easy to cap/observe |
| **Iris diff size** | Small (same as subprocess) — both just return a URL | Small — both just return a URL |
| **Reuses proven code** | No (new extension) | Yes (parity harness lifecycle) |

The Iris-side diff is **the same small size either way** — the disruptiveness difference is entirely
inside `finelog`/packaging, and there the subprocess wins decisively.

A note on the "subprocess needs the binary present" objection: it is the *only* genuine subprocess
downside, and it is mild. The binary is already produced by `lib/finelog/deploy/Dockerfile`
(`cargo build --release -p finelog --bin finelog-server`), the iris image already installs a Rust
toolchain, and the parity harness already locates the binary via `FINELOG_RUST_BIN` + the
`rust/target/{release,debug}/finelog-server` search. A pip-installed `marin-finelog` would need the
binary either bundled in the wheel (a maturin/hatchling `force-include` of a prebuilt binary) or
discoverable on `$PATH`/an env var. For Iris's in-cluster and local-dev use, an env var + a couple
of search paths is sufficient and matches existing practice.

---

## 4. Client-managed lifecycle sketch (`finelog.client`)

Keep the change inside `finelog` (honoring the dependency rule: `iris → finelog`, never the
reverse). Add an `EmbeddedLogServer` to `finelog.client` that owns the subprocess and a convenience
`LogClient.start_embedded(...)`.

```python
# lib/finelog/src/finelog/client/embedded.py  (new)
from __future__ import annotations
import os, shutil, socket, subprocess
from contextlib import closing
from pathlib import Path

import httpx
from rigging.timing import Duration, ExponentialBackoff

_BIN_ENV = "FINELOG_SERVER_BIN"
_HEALTH_TIMEOUT = Duration.from_seconds(20.0)


def _locate_binary() -> str:
    """Resolve the finelog-server binary: env override, PATH, then in-repo target."""
    if (override := os.environ.get(_BIN_ENV)):
        return override
    if (found := shutil.which("finelog-server")):
        return found
    repo_root = Path(__file__).resolve().parents[5]   # lib/finelog/src/finelog/client -> repo
    for profile in ("release", "debug"):
        candidate = repo_root / "rust" / "target" / profile / "finelog-server"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"finelog-server binary not found (set {_BIN_ENV}, put it on PATH, or build it with "
        "`cargo build --release -p finelog --bin finelog-server`)"
    )


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EmbeddedLogServer:
    """Spawn `finelog-server` as a child process and expose its localhost URL.

    The controller (and tests) talk to it over HTTP exactly as they would an
    external server, so the data-plane transport is unchanged. Use as a context
    manager or call `start()` / `close()` explicitly.
    """

    def __init__(self, log_dir: str | None = None, remote_log_dir: str = "") -> None:
        self._log_dir = log_dir
        self._remote_log_dir = remote_log_dir
        self._proc: subprocess.Popen | None = None
        self._url: str | None = None

    @property
    def url(self) -> str:
        assert self._url is not None, "server not started"
        return self._url

    def start(self) -> str:
        port = _free_port()
        argv = [_locate_binary(), "--port", str(port)]
        if self._log_dir:
            argv += ["--log-dir", self._log_dir]
        if self._remote_log_dir:
            argv += ["--remote-log-dir", self._remote_log_dir]
        self._proc = subprocess.Popen(argv)
        url = f"http://127.0.0.1:{port}"
        self._wait_health(url)
        self._url = url
        return url

    def _wait_health(self, url: str) -> None:
        def up() -> bool:
            if self._proc.poll() is not None:
                raise RuntimeError(f"finelog-server exited early ({self._proc.returncode})")
            try:
                return httpx.get(f"{url}/health", timeout=1.0).status_code == 200
            except httpx.HTTPError:
                return False
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(up, timeout=_HEALTH_TIMEOUT)

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()            # SIGTERM -> Rust graceful drain
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5.0)
        self._proc = None
        self._url = None

    def __enter__(self) -> "EmbeddedLogServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

(`finelog.client` already depends on `httpx` in dev and on `rigging.timing`; the controller already
uses `ExponentialBackoff` the same way for its uvicorn startup wait.)

### Iris controller: before / after

**Before** (`controller.py`, the in-process Python server):

```python
from finelog.server import LogServiceImpl
from finelog.server.asgi import build_log_server_asgi
from finelog.server.stats_service import StatsServiceImpl
from finelog.store.duckdb_store import EMBEDDED_DUCKDB_MEMORY_LIMIT, EMBEDDED_DUCKDB_THREADS, DuckDBLogStore
...
def _start_local_log_server(self) -> str:
    log_server_port = find_free_port()
    log_store = DuckDBLogStore(log_dir=None,
                               duckdb_memory_limit=EMBEDDED_DUCKDB_MEMORY_LIMIT,
                               duckdb_threads=EMBEDDED_DUCKDB_THREADS)
    self._log_service = LogServiceImpl(log_store=log_store)
    stats_service = StatsServiceImpl(log_store=log_store)
    interceptors = (NullAuthInterceptor(verifier=self._config.auth_verifier),)
    app = build_log_server_asgi(self._log_service, interceptors=interceptors, stats_service=stats_service)
    log_server_config = uvicorn.Config(app, host=self._config.host, port=log_server_port,
                                        log_level="warning", log_config=None, timeout_keep_alive=120)
    self._log_server = uvicorn.Server(log_server_config)
    self._threads.spawn_server(self._log_server, name="log-server")
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
        lambda: self._log_server is not None and self._log_server.started, timeout=Duration.from_seconds(5.0))
    address = f"http://{self.external_host}:{log_server_port}"
    logger.info("Local log server ready at %s", address)
    return address
```

**After** (`controller.py`, subprocess-backed; ~6 lines + one import):

```python
from finelog.client import EmbeddedLogServer  # plus existing LogClient/proxy imports
# (drop: LogServiceImpl, build_log_server_asgi, StatsServiceImpl, DuckDBLogStore imports)
...
def _start_local_log_server(self) -> str:
    # A child finelog-server process; the controller talks to it over localhost
    # HTTP exactly as it would an external endpoint. Backed by a temp dir so a
    # local run gets real segmentation/persistence instead of the old MemStore.
    self._embedded_log_server = EmbeddedLogServer(
        log_dir=str(self._config.local_state_dir / "finelog"),
    )
    address = self._embedded_log_server.start()
    logger.info("Local log server ready at %s", address)
    return address
```

And in `stop()`:

```python
# was: if self._log_service: self._log_service.close()
if self._embedded_log_server is not None:
    self._embedded_log_server.close()
```

Note the embedded-server auth is no longer wired through `NullAuthInterceptor` — the Rust
`finelog-server` is unauthenticated by design (it's network-restricted), matching the standalone
deploy. The local/test fallback runs on `127.0.0.1`, so this is acceptable; production never hits
this path. (If the embedded server ever needs to verify worker tokens, that becomes a real gap —
see §5 risks.)

### Tests

The 9 test files (§1.4) are migrated independently of this change. Two viable shapes:

- **Spawn + real client** (closest to production behavior): a session/function `EmbeddedLogServer`
  fixture in the iris `conftest.py`, with the controller-test `log_service` fixture's force-flush
  monkeypatch replaced by the `--debug-admin` `POST /debug/maintain` route the parity harness already
  uses (`conftest.py::maintain`). This deletes the in-process Python service from the test surface
  entirely.
- **In-process fake** (keeps tests hermetic and fast, no subprocess per test): a tiny fake that
  satisfies the `LogServiceProxy` / `LogClient` method surface used by these tests
  (`fetch_logs`, `push_logs`, `get_table`, `close`). The existing `_FakeLogClientFromService` and
  `InProcessLogClient` are already 80% of this; they'd just stop importing the real `LogServiceImpl`
  and back onto an in-memory dict or a spawned server. This is the cheaper migration and still lets
  `finelog.server`/`finelog.store` be deleted.

The choice between these is the crux of the remaining work and should be made deliberately (see §5).

---

## 5. Effort, risk, and what still blocks deletion

### Effort estimate

| Work item | Estimate |
| --- | --- |
| `EmbeddedLogServer` + `LogClient.start_embedded` + unit test in `finelog` | 0.5 day |
| Iris controller swap (`_start_local_log_server`, `stop()`, imports) | 0.25 day |
| Iris test migration (9 files → spawned server or in-process fake) | 0.5–1.5 days (depends on shape) |
| Binary distribution wiring (wheel `force-include` of prebuilt binary, or `$PATH`/env in iris image) | 0.25–0.5 day |
| **Total (subprocess path)** | **~1.5–2.5 days** |
| **PyO3 path, for comparison** | **~4–6 days + ongoing CI wheel matrix maintenance** |

### Main risks (subprocess path)

1. **Binary availability at runtime.** A pip-installed `marin-finelog` with no co-located binary
   fails at controller start with `FileNotFoundError`. Mitigation: bundle the prebuilt binary in the
   wheel via hatchling `force-include` (keeps `marin-finelog` pure-Python build but ships a platform
   binary — note this makes the wheel platform-specific, a packaging change), or require the binary
   on `$PATH` and document it. Decide based on how `marin-finelog` is consumed off-monorepo.
2. **Auth on the embedded server.** The old in-process server installed
   `NullAuthInterceptor(verifier=...)`. The Rust `finelog-server` is unauthenticated. For the
   loopback fallback this is fine, but confirm no local/test path depends on the embedded server
   *rejecting* an invalid token. (Quick grep suggests the embedded server's auth is only ever
   `Null`, so this is low-risk.)
3. **Startup latency / flakiness.** Spawning a process + `/health` poll is ~50–300 ms vs. a uvicorn
   thread. Per-test subprocess spawn in the migrated tests could add up; the in-process-fake test
   shape avoids this. The parity harness already runs this way without flakiness at a 20 s timeout.
4. **Process leak on hard crash.** If the controller is `kill -9`'d, the child can orphan. Mitigation:
   the controller already has an `atexit` hook and `stop()`; add the child to those paths (and
   optionally `prctl(PR_SET_PDEATHSIG)` / a process group on Linux).

### What still blocks deleting `finelog.store` / `finelog.server` after this

- **The 9 Iris test files (§1.4).** Until they stop importing `from finelog.server import
  LogServiceImpl` and constructing `DuckDBLogStore`-backed services in-process, the Python server
  cannot be deleted. This migration is the real gate and is **orthogonal to PyO3-vs-subprocess** —
  it's about removing in-process Python-service usage from the test surface.
- **The `finelog.client` → `finelog.store` schema-type coupling.** `LogClient`
  (`finelog/client/log_client.py:43–53`) imports `LOG_REGISTERED_SCHEMA` from
  `finelog.store.log_namespace`, `StoragePolicy` from `finelog.store.policy`, and `Column`, `Schema`,
  `schema_to_arrow`, `schema_to_proto`, `schema_from_proto`, `IMPLICIT_SEQ_COLUMN`, `ColumnTypeValue`
  from `finelog.store.schema`. `finelog.client.__init__` and `finelog.server.__init__` also re-export
  store/client symbols. So even with the server out-of-process, **the client still pulls in
  `finelog.store`** for schema/type definitions. The task description notes this is being decoupled
  separately; that decoupling (moving the shared schema/policy types out of `finelog.store` into a
  neutral module the client can import without the DuckDB store) is a hard prerequisite for deleting
  `finelog.store`. Deleting `finelog.server` is unblocked by the test migration alone; deleting
  `finelog.store` additionally requires that schema-type decoupling.

### Dependency-rule check

The recommended change lives almost entirely in `finelog` (`EmbeddedLogServer`), with a tiny
consuming diff in `iris`. That respects `iris → finelog` and never introduces a reverse edge. Iris's
diff is ~10 lines plus the (separately-scoped) test migration.

---

## Appendix: key file references

- Iris embedding: `lib/iris/src/iris/cluster/controller/controller.py` — `_start_local_log_server`
  (486–531), log-service setup (333–367), `stop()` close path (647–656), imports (20–25).
- Dashboard mounts: `lib/iris/src/iris/cluster/controller/dashboard.py` — `log_service` annotation
  (372), LogService/legacy mounts (520–521), finelog stats mount (525–531), import (38).
- Production endpoint resolution: `lib/iris/src/iris/cluster/controller/main.py:46–111,227–228`;
  configs `lib/iris/config/marin.yaml:4`, `marin-dev.yaml:5`.
- Proxies (HTTP transport): `lib/finelog/src/finelog/client/proxy.py`.
- Client→store schema coupling: `lib/finelog/src/finelog/client/log_client.py:43–53`;
  `lib/finelog/src/finelog/client/__init__.py`; `lib/finelog/src/finelog/server/__init__.py`.
- Subprocess precedent: `lib/finelog/tests/parity/conftest.py` (Popen + `/health` + terminate;
  `_rust_binary` locator; `/debug/maintain` helper).
- Rust server: `rust/finelog/Cargo.toml` (`[[bin]]` + `[lib]`, no PyO3), `rust/finelog/src/main.rs`
  (CLI flags, tokio main, graceful shutdown), `rust/finelog/src/server/mod.rs` & `server/app.rs`
  (`build_app`, `ServerConfig`), `rust/finelog/src/server/spa.rs` (`vue_dist_dir`).
- PyO3 precedent: `rust/dupekit/Cargo.toml` (cdylib + pyo3 abi3), `rust/dupekit/pyproject.toml`
  (maturin), `.github/workflows/dupekit-release-wheels.yaml`, `rust/dupekit/build_package.py`.
- Standalone Rust deploy: `lib/finelog/deploy/Dockerfile` (multi-stage cargo build),
  `lib/finelog/deploy/k8s/02-deployment.yaml.tmpl`.
- Iris test coupling: `lib/iris/tests/cluster/conftest.py`,
  `lib/iris/tests/cluster/controller/conftest.py`, `lib/iris/tests/cluster/providers/k8s/conftest.py`
  (+ 6 more listed in §1.4).
