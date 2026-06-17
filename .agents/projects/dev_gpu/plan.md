# dev_gpu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `dev_tpu.py`-style CLI that reserves a CoreWeave H100 pod through Iris and `kubectl exec -it`s into it for ad-hoc GPU dev. Lean MVP: `allocate` / `connect` / `status` / `release`.

**Architecture:** Iris is used only to submit + hold + resolve a GPU holder job; interactive access shells out to `kubectl` against the backing task pod (resolved by the `iris.task_id` label in namespace `iris`). Pure helpers (platform gate, label selector, kubectl arg-building, pod-JSON parsing, session state) are split from the I/O glue so they can be unit-tested with the kubectl/Iris boundaries faked.

**Tech Stack:** Python 3.11, `click`, Iris client (`iris.client.IrisClient`), `kubectl` (subprocess), pytest.

---

## Background facts (verified against the repo)

- `gpu_device("H100", 8)` exists in `lib/iris/src/iris/cluster/types.py:508`; `.gpu.count == 8`, `.gpu.variant == "H100"`.
- CoreWeave scale group `h100-8x` in `lib/iris/config/coreweave.yaml` matches `device_type: gpu, device_variant: H100, device_count: 8`.
- Platform is a proto oneof: `config.platform.WhichOneof("platform")` returns `"coreweave"` / `"gcp"`. `CoreweavePlatformConfig` has `namespace` (default `"iris"`) and `kubeconfig_path` (`lib/iris/src/iris/rpc/config.proto:38`).
- Task pods are labeled `iris.task_id=<_sanitize_label_value(task_id)>` (`lib/iris/src/iris/cluster/backends/k8s/tasks.py:81,587`), container name `task`. Example: `_sanitize_label_value("/matt/dev-gpu-matt/0") == "matt.dev-gpu-matt.0"`.
- `scripts.iris.*` is importable as a namespace package **only with repo root on `sys.path`**. There is no root `conftest.py` and no pytest `pythonpath`, so tests must be run with `uv run python -m pytest …` (the `-m` form puts CWD on the path); the bare `pytest` console script will NOT resolve `from scripts.iris...`.

## File structure

- **Create** `scripts/iris/dev_gpu.py` — the CLI tool. One file, mirroring `scripts/iris/dev_tpu.py`'s structure (it is the reference for every reused pattern).
- **Create** `scripts/iris/tests/test_dev_gpu.py` — unit tests for the pure helpers.

No `__init__.py` files are added (namespace-package imports already work, per the background facts).

---

### Task 1: Module scaffolding + session-state dataclasses

**Files:**
- Create: `scripts/iris/dev_gpu.py`
- Test: `scripts/iris/tests/test_dev_gpu.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/iris/tests/test_dev_gpu.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.iris.dev_gpu import CoreweaveTarget, DevGpuState, PodRef


def test_state_round_trip():
    state = DevGpuState(
        session_name="matt",
        config_file="/abs/coreweave.yaml",
        job_id="/matt/dev-gpu-matt",
        gpu_count=8,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg"),
        pod=PodRef(namespace="iris", pod_name="dev-gpu-matt-abc", container="task"),
    )
    assert DevGpuState.from_json(state.to_json()) == state
```

- [ ] **Step 2: Run test to verify it fails**

Run (from repo root): `uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.iris.dev_gpu'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/iris/dev_gpu.py`:

```python
#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate and use development CoreWeave H100 pods on Iris-managed clusters."""

from __future__ import annotations

import getpass
import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from iris.client import IrisClient, JobAlreadyExists
from iris.cluster.backends.k8s.tasks import _LABEL_TASK_ID, _sanitize_label_value
from iris.cluster.config import IrisConfig
from iris.cluster.types import Entrypoint, JobName, ResourceSpec, gpu_device
from iris.rpc import config_pb2, job_pb2

logger = logging.getLogger(__name__)

HOLDER_COMMAND = (
    "import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
    "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
    "print('iris dev coreweave holder ready', flush=True); "
    "time.sleep(365 * 24 * 60 * 60)"
)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_gpu_iris"
DEFAULT_GPU_COUNT = 8
TASK_CONTAINER = "task"
GPU_VARIANT = "H100"

TERMINAL_JOB_STATES = {
    job_pb2.JOB_STATE_FAILED,
    job_pb2.JOB_STATE_KILLED,
    job_pb2.JOB_STATE_UNSCHEDULABLE,
    job_pb2.JOB_STATE_WORKER_FAILED,
}
INACTIVE_JOB_STATES = TERMINAL_JOB_STATES | {job_pb2.JOB_STATE_SUCCEEDED}


@dataclass(frozen=True)
class PodRef:
    """The k8s pod backing a dev GPU session."""

    namespace: str
    pod_name: str
    container: str = TASK_CONTAINER


@dataclass(frozen=True)
class CoreweaveTarget:
    """Cluster-level kubectl target. Empty kubeconfig_path => kubectl default resolution."""

    namespace: str
    kubeconfig_path: str


@dataclass(frozen=True)
class DevGpuState:
    """Persisted local state for an active dev GPU session."""

    session_name: str
    config_file: str
    job_id: str
    gpu_count: int
    target: CoreweaveTarget
    pod: PodRef

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> DevGpuState:
        data = json.loads(raw)
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            gpu_count=data["gpu_count"],
            target=CoreweaveTarget(**data["target"]),
            pod=PodRef(**data["pod"]),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/iris/dev_gpu.py scripts/iris/tests/test_dev_gpu.py
git commit -m "[dev-coreweave] Session-state dataclasses + round-trip test"
```

---

### Task 2: Platform gate (`require_coreweave_platform`)

**Files:**
- Modify: `scripts/iris/dev_gpu.py`
- Test: `scripts/iris/tests/test_dev_gpu.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/iris/tests/test_dev_gpu.py`:

```python
import click
import pytest
from iris.rpc import config_pb2

from scripts.iris.dev_gpu import require_coreweave_platform


def _coreweave_config(namespace: str = "iris", kubeconfig: str = "") -> config_pb2.IrisClusterConfig:
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.SetInParent()
    if namespace:
        c.platform.coreweave.namespace = namespace
    if kubeconfig:
        c.platform.coreweave.kubeconfig_path = kubeconfig
    return c


def test_require_coreweave_accepts_and_expands_kubeconfig():
    target = require_coreweave_platform(_coreweave_config(kubeconfig="~/.kube/coreweave-iris"))
    assert target.namespace == "iris"
    assert target.kubeconfig_path.endswith("/.kube/coreweave-iris")
    assert "~" not in target.kubeconfig_path


def test_require_coreweave_default_namespace_when_unset():
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.SetInParent()
    assert require_coreweave_platform(c).namespace == "iris"


def test_require_coreweave_rejects_gcp_with_pointer_to_dev_tpu():
    g = config_pb2.IrisClusterConfig()
    g.platform.gcp.SetInParent()
    with pytest.raises(click.ClickException, match="dev_tpu.py"):
        require_coreweave_platform(g)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v`
Expected: FAIL — `ImportError: cannot import name 'require_coreweave_platform'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/iris/dev_gpu.py` (after the dataclasses):

```python
def require_coreweave_platform(config: config_pb2.IrisClusterConfig) -> CoreweaveTarget:
    """Resolve the kubectl target for a CoreWeave cluster, or fail fast.

    Inverts dev_tpu.py's GCP gate: this tool only works against
    CoreWeave/Kubernetes-backed clusters.
    """
    if config.platform.WhichOneof("platform") != "coreweave":
        raise click.ClickException(
            "dev_gpu requires a CoreWeave/Kubernetes-backed cluster. "
            "For GCP TPU clusters use scripts/iris/dev_tpu.py."
        )
    cw = config.platform.coreweave
    namespace = cw.namespace or "iris"
    kubeconfig_path = os.path.expanduser(cw.kubeconfig_path) if cw.kubeconfig_path else ""
    return CoreweaveTarget(namespace=namespace, kubeconfig_path=kubeconfig_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/iris/dev_gpu.py scripts/iris/tests/test_dev_gpu.py
git commit -m "[dev-coreweave] CoreWeave platform gate"
```

---

### Task 3: Pure kubectl helpers + pod-JSON parsing

**Files:**
- Modify: `scripts/iris/dev_gpu.py`
- Test: `scripts/iris/tests/test_dev_gpu.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/iris/tests/test_dev_gpu.py`:

```python
from iris.cluster.backends.k8s.tasks import _LABEL_TASK_ID, _sanitize_label_value

from scripts.iris.dev_gpu import (
    kubectl_base,
    kubectl_connect_cmd,
    kubectl_get_pods_cmd,
    parse_running_pod,
    pod_label_selector,
)


def test_pod_label_selector_matches_iris_label():
    sel = pod_label_selector("/matt/dev-gpu-matt/0")
    assert sel == f"{_LABEL_TASK_ID}={_sanitize_label_value('/matt/dev-gpu-matt/0')}"
    assert sel == "iris.task_id=matt.dev-gpu-matt.0"


def test_kubectl_base_with_kubeconfig():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg")
    assert kubectl_base(t) == ["kubectl", "--kubeconfig=/k/cfg", "--namespace=iris"]


def test_kubectl_base_without_kubeconfig():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="")
    assert kubectl_base(t) == ["kubectl", "--namespace=iris"]


def test_kubectl_get_pods_cmd():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="")
    assert kubectl_get_pods_cmd(t, "iris.task_id=x") == [
        "kubectl", "--namespace=iris", "get", "pods", "-l", "iris.task_id=x", "-o", "json",
    ]


def test_kubectl_connect_cmd():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg")
    pod = PodRef(namespace="iris", pod_name="dev-gpu-matt-abc", container="task")
    assert kubectl_connect_cmd(t, pod) == [
        "kubectl", "--kubeconfig=/k/cfg", "--namespace=iris",
        "exec", "-it", "dev-gpu-matt-abc", "-c", "task", "--", "bash", "-l",
    ]


def test_parse_running_pod_picks_running():
    pods = {"items": [
        {"metadata": {"name": "b"}, "status": {"phase": "Pending"}},
        {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
    ]}
    assert parse_running_pod(pods) == "a"


def test_parse_running_pod_is_deterministic_by_name():
    pods = {"items": [
        {"metadata": {"name": "z"}, "status": {"phase": "Running"}},
        {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
    ]}
    assert parse_running_pod(pods) == "a"


def test_parse_running_pod_none_when_no_running():
    pods = {"items": [{"metadata": {"name": "a"}, "status": {"phase": "Pending"}}]}
    assert parse_running_pod(pods) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v`
Expected: FAIL — `ImportError: cannot import name 'kubectl_base'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/iris/dev_gpu.py` (after `require_coreweave_platform`):

```python
def pod_label_selector(task_id: str) -> str:
    """k8s label selector matching the task pod Iris created for ``task_id``."""
    return f"{_LABEL_TASK_ID}={_sanitize_label_value(task_id)}"


def kubectl_base(target: CoreweaveTarget) -> list[str]:
    cmd = ["kubectl"]
    if target.kubeconfig_path:
        cmd.append(f"--kubeconfig={target.kubeconfig_path}")
    cmd.append(f"--namespace={target.namespace}")
    return cmd


def kubectl_get_pods_cmd(target: CoreweaveTarget, selector: str) -> list[str]:
    return kubectl_base(target) + ["get", "pods", "-l", selector, "-o", "json"]


def kubectl_connect_cmd(target: CoreweaveTarget, pod: PodRef) -> list[str]:
    return kubectl_base(target) + ["exec", "-it", pod.pod_name, "-c", pod.container, "--", "bash", "-l"]


def parse_running_pod(pods_json: dict) -> str | None:
    """Return the lexicographically-first Running pod name, or None."""
    items = pods_json.get("items", [])
    running = [p for p in items if p.get("status", {}).get("phase") == "Running"]
    if not running:
        return None
    running.sort(key=lambda p: p.get("metadata", {}).get("name", ""))
    return running[0]["metadata"]["name"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/iris/dev_gpu.py scripts/iris/tests/test_dev_gpu.py
git commit -m "[dev-coreweave] Pure kubectl helpers + pod-JSON parsing"
```

---

### Task 4: I/O glue + CLI commands

This task wires the I/O layer (Iris client, kubectl subprocess, session files) and the click CLI. It mirrors `scripts/iris/dev_tpu.py` and is validated end-to-end manually in Task 5 (it talks to a live cluster, so it is not unit-tested).

**Files:**
- Modify: `scripts/iris/dev_gpu.py`

- [ ] **Step 1: Add the I/O helpers**

Add to `scripts/iris/dev_gpu.py` (after `parse_running_pod`):

```python
def run_logged(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("Running command: %s", shlex.join(cmd))
    return subprocess.run(cmd, **kwargs)


def find_workspace_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


@contextmanager
def controller_client(config_file: str) -> Iterable[IrisClient]:
    iris_config = IrisConfig.load(config_file)
    controller_address = iris_config.controller_address()
    providers = iris_config.provider_bundle()
    controller = providers.controller
    workspace = find_workspace_root(Path.cwd())
    if not controller_address:
        controller_address = controller.discover_controller(iris_config.proto.controller)
    with controller.tunnel(address=controller_address) as tunneled:
        client = IrisClient.remote(tunneled, workspace=workspace)
        try:
            yield client
        finally:
            client.shutdown()


def state_path(state_dir: Path, session_name: str) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{session_name}.json"


def load_state(path: Path) -> DevGpuState:
    if not path.exists():
        raise click.ClickException(f"No active dev GPU session at {path}")
    return DevGpuState.from_json(path.read_text())


def save_state(path: Path, state: DevGpuState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json())


def is_job_active(client: IrisClient, job_id: str) -> bool:
    return client.status(JobName.from_wire(job_id)).state not in INACTIVE_JOB_STATES


def wait_for_running_task(job, *, timeout: float) -> str:
    """Block until the holder job's single task is RUNNING; return its task_id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = job.status()
        if status.state in TERMINAL_JOB_STATES:
            error = status.error or job_pb2.JobState.Name(status.state)
            raise click.ClickException(f"Dev GPU allocation failed: {error}")
        tasks = job.tasks()
        if tasks:
            task = tasks[0]
            if task.status().state == job_pb2.TASK_STATE_RUNNING:
                return str(task.task_id)
        time.sleep(5)
    raise click.ClickException(f"Timed out waiting for dev GPU task after {int(timeout)}s")


def wait_for_running_pod(target: CoreweaveTarget, task_id: str, *, timeout: float) -> PodRef:
    """Poll kubectl until the task's backing pod is Running."""
    selector = pod_label_selector(task_id)
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        result = run_logged(kubectl_get_pods_cmd(target, selector), capture_output=True, text=True)
        if result.returncode == 0:
            pod_name = parse_running_pod(json.loads(result.stdout or "{}"))
            if pod_name:
                return PodRef(namespace=target.namespace, pod_name=pod_name, container=TASK_CONTAINER)
        else:
            last_err = result.stderr.strip()
        time.sleep(3)
    raise click.ClickException(
        f"Timed out resolving a Running pod for selector {selector!r} after {int(timeout)}s. {last_err}"
    )
```

- [ ] **Step 2: Add the click CLI**

Add to `scripts/iris/dev_gpu.py` (after the I/O helpers):

```python
@dataclass
class Context:
    config_file: str | None = None
    session_name: str | None = None
    state_dir: Path = STATE_DIR


@click.group()
@click.option("--config", help="Path to an Iris cluster config file.")
@click.option("--name", "session_name", help="Local dev GPU session name.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx, config: str | None, session_name: str | None, verbose: bool) -> None:
    """Development CoreWeave H100 pod management for Iris clusters."""
    ctx.ensure_object(Context)
    ctx.obj.config_file = str(Path(config).resolve()) if config else None
    ctx.obj.session_name = session_name or getpass.getuser()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@cli.command("allocate")
@click.option("--gpu-count", default=DEFAULT_GPU_COUNT, show_default=True, help="H100 GPUs to reserve.")
@click.option("--timeout", default=900, show_default=True, help="Seconds to wait for the task to run.")
@click.option("--pod-timeout", default=120, show_default=True, help="Seconds to wait for the pod to run.")
@click.pass_context
def allocate(ctx, gpu_count: int, timeout: int, pod_timeout: int) -> None:
    """Allocate a dev GPU H100 pod and hold it until Ctrl-C."""
    if not ctx.obj.config_file:
        raise click.ClickException("--config is required")

    session_name = ctx.obj.session_name
    state_file = state_path(ctx.obj.state_dir, session_name)
    if state_file.exists():
        raise click.ClickException(
            f"Dev GPU session '{session_name}' already exists. Use release first or choose a new --name."
        )

    target = require_coreweave_platform(IrisConfig.load(ctx.obj.config_file).proto)

    state: DevGpuState | None = None
    with controller_client(ctx.obj.config_file) as client:
        resources = ResourceSpec(
            cpu=0.5, memory="1GB", disk="5GB", device=gpu_device(GPU_VARIANT, gpu_count)
        )
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", "-c", HOLDER_COMMAND),
                name=f"dev-gpu-{session_name}",
                resources=resources,
            )
        except JobAlreadyExists as exc:
            raise click.ClickException(f"Job already exists for session '{session_name}': {exc}") from exc

        try:
            task_id = wait_for_running_task(job, timeout=timeout)
            pod = wait_for_running_pod(target, task_id, timeout=pod_timeout)
            state = DevGpuState(
                session_name=session_name,
                config_file=ctx.obj.config_file,
                job_id=str(job.job_id),
                gpu_count=gpu_count,
                target=target,
                pod=pod,
            )
            save_state(state_file, state)

            print(f"Session: {session_name}")
            print(f"Job: {job.job_id}")
            print(f"GPUs: {gpu_count} x {GPU_VARIANT}")
            print(f"Pod: {pod.pod_name} (namespace={pod.namespace})")
            print("\nAllocation is active. Press Ctrl-C to release.")

            while True:
                time.sleep(30)
                if not is_job_active(client, str(job.job_id)):
                    raise click.ClickException("The dev GPU holder job terminated unexpectedly.")
        except KeyboardInterrupt:
            print("\nReleasing dev GPU session...")
        finally:
            try:
                client.terminate(JobName.from_wire(str(job.job_id)))
            except Exception:
                logger.warning("Failed to terminate holder job %s", job.job_id, exc_info=True)
            if state is not None:
                state_file.unlink(missing_ok=True)


@cli.command("connect")
@click.pass_context
def connect(ctx) -> None:
    """Open an interactive shell into the reserved pod."""
    state = load_state(state_path(ctx.obj.state_dir, ctx.obj.session_name))
    with controller_client(state.config_file) as client:
        if not is_job_active(client, state.job_id):
            raise click.ClickException(
                f"Dev GPU session '{state.session_name}' is no longer active. Use release to clean up."
            )
    run_logged(kubectl_connect_cmd(state.target, state.pod), check=True)


@cli.command("status")
@click.pass_context
def status(ctx) -> None:
    """Show the current session state."""
    state = load_state(state_path(ctx.obj.state_dir, ctx.obj.session_name))
    print(f"Session: {state.session_name}")
    print(f"Job: {state.job_id}")
    print(f"Config: {state.config_file}")
    print(f"GPUs: {state.gpu_count} x {GPU_VARIANT}")
    print(f"Pod: {state.pod.pod_name} (namespace={state.pod.namespace})")


@cli.command("release")
@click.pass_context
def release(ctx) -> None:
    """Terminate the holder job and clear the local session file."""
    state_file = state_path(ctx.obj.state_dir, ctx.obj.session_name)
    state = load_state(state_file)
    try:
        with controller_client(state.config_file) as client:
            client.terminate(JobName.from_wire(state.job_id))
    finally:
        state_file.unlink(missing_ok=True)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the module imports and the CLI is wired**

Run (from repo root):
```bash
uv run python -m pytest scripts/iris/tests/test_dev_gpu.py -v
uv run python scripts/iris/dev_gpu.py --help
```
Expected: 12 passed; `--help` lists `allocate`, `connect`, `status`, `release`.

- [ ] **Step 4: Lint**

Run: `./infra/pre-commit.py --files scripts/iris/dev_gpu.py scripts/iris/tests/test_dev_gpu.py --fix`
Expected: clean (or auto-fixed). Re-run the pytest command above if the formatter changed anything.

- [ ] **Step 5: Commit**

```bash
git add scripts/iris/dev_gpu.py
git commit -m "[dev-coreweave] I/O glue + allocate/connect/status/release CLI"
```

---

### Task 5: Manual end-to-end validation (live cluster)

No automated test covers the live path (it costs an H100 node and needs cluster creds). This task records the manual checks. Do NOT run it speculatively — only against a CoreWeave cluster you are authorized to use, and release promptly.

**Files:** none (validation only).

- [ ] **Step 1: Allocate against a CoreWeave config**

Run (replace the config with the real CoreWeave cluster config; `--gpu-count 8` holds a whole `h100-8x` node):
```bash
uv run python scripts/iris/dev_gpu.py \
  --config lib/iris/config/coreweave.yaml --name "$USER-cw" \
  allocate --gpu-count 8
```
Expected: prints `Pod: …`, then "Allocation is active. Press Ctrl-C to release." Leave it running in this terminal.

- [ ] **Step 2: Connect from a second terminal**

```bash
uv run python scripts/iris/dev_gpu.py --config lib/iris/config/coreweave.yaml --name "$USER-cw" connect
```
Expected: an interactive shell inside the pod. Inside it, confirm GPUs are visible:
```bash
nvidia-smi -L   # expect 8x H100
```
**This is the key open validation point** — that `kubectl exec -it` against an Iris task pod yields a working TTY shell.

- [ ] **Step 3: Status**

```bash
uv run python scripts/iris/dev_gpu.py --config lib/iris/config/coreweave.yaml --name "$USER-cw" status
```
Expected: prints the session, job id, pod, and GPU count.

- [ ] **Step 4: Release**

Either Ctrl-C the `allocate` terminal, or from another shell:
```bash
uv run python scripts/iris/dev_gpu.py --config lib/iris/config/coreweave.yaml --name "$USER-cw" release
```
Expected: holder job terminated, session file gone. Confirm the node is released:
```bash
uv run iris --config lib/iris/config/coreweave.yaml job list --prefix /$USER/dev-gpu
```

- [ ] **Step 5: Record results**

Append findings (did interactive exec work? did `--gpu-count 8` schedule? any pod-resolution flakiness?) to `.agents/projects/dev_gpu/research.md` under a new "Live validation" heading, resolving the design's open questions. Commit:
```bash
git add .agents/projects/dev_gpu/research.md
git commit -m "[dev-coreweave] Record live validation findings"
```

---

## Notes for the implementer

- **`python -m pytest`, not bare `pytest`.** `from scripts.iris...` needs repo root on `sys.path`; the `-m` form provides it. The bare console script will `ModuleNotFoundError`.
- **Private imports are intentional.** `_LABEL_TASK_ID` and `_sanitize_label_value` come from `iris.cluster.backends.k8s.tasks` so the pod selector is built from the same source of truth Iris uses to label the pod. Do not re-implement them — drift would silently break pod resolution.
- **No file sync / execute / watch.** Deferred per the spec. The task image is self-contained; the loop is "reserve a node, shell in." Add sync only after the live path is proven.
- **Reference implementation:** `scripts/iris/dev_tpu.py`. Every reused pattern (holder command, controller_client, session state, wait loop) is modeled on it.
