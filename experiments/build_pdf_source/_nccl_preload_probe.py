# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- test whether a given NCCL build is what breaks ``import torch``.

DELETE once the answer is recorded in ``.agents/ops/``. Nothing in the pipeline imports this.

The datakit closure installs ``nvidia-nccl-cu12`` (via xgboost) and ``nvidia-nccl-cu13`` (via
torch); both wheels own ``nvidia/nccl/lib/libnccl.so.2``, so whichever unpacked last is the one
``import torch`` loads into global scope. This forces the question rather than waiting for the
race: for every distinct ``libnccl.so.2`` in the node's shared uv cache, run ``import torch`` in a
subprocess with that library in ``LD_PRELOAD`` and record what happens.

``LD_PRELOAD`` puts the chosen library ahead of everything in the global symbol scope, which is
exactly the position the winner of the shared-path race occupies. If one build reproduces::

    ImportError: .../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommResume

then the race is the mechanism and that build is the loser. If none does, the mechanism is
somewhere else. Nothing here mutates the venv, so concurrent tasks on the same pod are unaffected.

Run it on the cluster the failure happened on::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name nccl-preload-probe \\
        -- python -m experiments.build_pdf_source._nccl_preload_probe
"""

import logging
import os
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import prefix_join, url_to_fs
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

logger = logging.getLogger(__name__)

# Enough tasks to reach several nodes without re-walking each shared cache many times.
PROBE_TASKS = 8
_SYMBOL = b"ncclCommResume"
_SCAN_CHUNK = 8 * 1024 * 1024
_NCCL_RELATIVE_PATH = "nvidia/nccl/lib/libnccl.so.2"
_IMPORT_TIMEOUT = 600

_PROBE_SCHEMA = pa.schema(
    [
        pa.field("index", pa.int64(), nullable=False),
        pa.field("host", pa.string(), nullable=False),
        pa.field("candidate", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
    ]
)

_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="16g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
_MAP_TASK_RESOURCES = ResourceConfig(cpu=2, ram="14g", disk="4g")
_MAX_WORKERS = 8
_HEARTBEAT_TIMEOUT = 30 * 60


class PreloadReport(BaseModel):
    """What ``import torch`` does when each available NCCL build is forced into global scope."""

    version: str = "v1"
    rows: int
    outcomes: dict[str, int]


def _exports_symbol(path: str) -> bool:
    """Whether ``path``'s ELF string tables contain ``ncclCommResume``."""
    overlap = len(_SYMBOL) - 1
    tail = b""
    with open(path, "rb") as handle:
        while chunk := handle.read(_SCAN_CHUNK):
            if _SYMBOL in tail + chunk:
                return True
            tail = chunk[-overlap:]
    return False


def _distribution_of(path: str) -> str:
    """The wheel an unpacked uv cache archive came from, read off its ``.dist-info``."""
    archive = os.path.realpath(path)
    while archive != "/" and os.path.basename(os.path.dirname(archive)) != "archive-v0":
        archive = os.path.dirname(archive)
    if archive == "/":
        return "unpacked"
    for entry in sorted(os.listdir(archive)):
        if entry.endswith(".dist-info"):
            return entry.removesuffix(".dist-info")
    return "unknown"


def _candidates(site_packages: str, cache_dir: str) -> dict[str, str]:
    """One ``libnccl.so.2`` path per distinct distribution, plus whatever the venv itself has."""
    found: dict[str, str] = {}
    venv_library = os.path.join(site_packages, _NCCL_RELATIVE_PATH)
    if os.path.exists(venv_library):
        found[f"venv({_distribution_of(venv_library)})"] = venv_library
    for root, _, files in os.walk(cache_dir):
        if "libnccl.so.2" not in files or not root.endswith(os.path.dirname(_NCCL_RELATIVE_PATH)):
            continue
        path = os.path.join(root, "libnccl.so.2")
        found.setdefault(_distribution_of(path), path)
    return found


def _import_outcome(preload: str | None) -> str:
    """Run ``import torch`` in a subprocess and report how it ended."""
    env = dict(os.environ)
    if preload:
        env["LD_PRELOAD"] = preload
    else:
        env.pop("LD_PRELOAD", None)
    completed = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_IMPORT_TIMEOUT,
        check=False,
    )
    if completed.returncode == 0:
        return f"ok {completed.stdout.strip()}"
    tail = [line for line in completed.stderr.strip().splitlines() if line.strip()]
    return f"fail {tail[-1][:300] if tail else 'no stderr'}"


def probe(work: dict) -> Iterator[dict]:
    """Import torch once per available NCCL build, with that build forced into global scope."""
    site_packages = os.path.join(
        sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
    )
    cache_dir = os.environ.get("UV_CACHE_DIR", "/uv/cache")
    host = socket.gethostname()

    yield {
        "index": work["index"],
        "host": host,
        "candidate": "no-preload",
        "symbol": "n/a",
        "path": "",
        "outcome": _import_outcome(None),
    }
    for candidate, path in sorted(_candidates(site_packages, cache_dir).items()):
        yield {
            "index": work["index"],
            "host": host,
            "candidate": candidate,
            "symbol": "has" if _exports_symbol(path) else "missing",
            "path": path,
            "outcome": _import_outcome(path),
        }


def run_probe(output_path: str) -> PreloadReport:
    """Fan out the preload matrix and tally the outcomes."""
    work = [{"index": index} for index in range(PROBE_TASKS)]
    rows_dir = prefix_join(output_path, "outputs/preload")
    pipeline = (
        Dataset.from_list(work)
        .flat_map(probe)
        .write_parquet(
            prefix_join(rows_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_PROBE_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="nccl-preload-probe",
        resources=_WORKER_RESOURCES,
        max_workers=min(_MAX_WORKERS, len(work)),
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)

    rows: list[dict] = []
    filesystem, path = url_to_fs(rows_dir)
    for written in sorted(filesystem.glob(f"{path}/*.parquet")):
        with filesystem.open(written, "rb") as stream:
            rows.extend(pq.read_table(stream).to_pylist())

    outcomes: Counter = Counter()
    for row in rows:
        outcomes[f"{row['candidate']} symbol={row['symbol']} -> {row['outcome'][:120]}"] += 1

    logger.info("=== NCCL PRELOAD PROBE ===")
    for key, count in sorted(outcomes.items()):
        logger.info("  %4d  %s", count, key)
    return PreloadReport(rows=len(rows), outcomes=dict(outcomes))


def probe_step() -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/nccl_preload_probe",
        deps=[],
        hash_attrs={"tasks": PROBE_TASKS, "attempt": 1},
        fn=remote(
            partial(run_probe),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    StepRunner().run([probe_step()])


if __name__ == "__main__":
    main()
