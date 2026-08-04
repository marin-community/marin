# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- find out why ``import torch`` intermittently dies with an NCCL undefined symbol.

DELETE once the answer is recorded in ``.agents/ops/``. Nothing in the pipeline imports this.

``/muchanem/control-layout-variance-2`` lost 600 of 1200 extractions to::

    ImportError: /app/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so:
    undefined symbol: ncclCommResume

while ``/muchanem/compare-layout-backends-5``, launched minutes later off the same lock onto the
same cluster, lost none. Both runs installed byte-identical package sets, so the variable is not
resolution. The datakit closure installs ``nvidia-nccl-cu12`` and ``nvidia-nccl-cu13`` at the same
version and both wheels own ``nvidia/nccl/lib/libnccl.so.2``; ``uv sync --link-mode symlink``
points that one path at whichever wheel's unpacked copy in the node's shared cache landed last.

This fans out over the control run's task shape and inventories, per node, every unpacked
``libnccl.so.2`` and ``libtorch_cuda.so`` in the shared uv cache: which distribution each came
from, and whether it carries ``ncclCommResume``. That is what says whether the two NCCL wheels
differ in the symbol torch needs, and whether the venv's own copies agree.

Run it on the cluster the failure happened on::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name nccl-import-probe \\
        -- python -m experiments.build_pdf_source._nccl_import_probe
"""

import logging
import os
import socket
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

# Enough tasks to land on every node the control run used without re-scanning each cache 48 times.
PROBE_TASKS = 16
_SYMBOL = b"ncclCommResume"
_SCAN_CHUNK = 8 * 1024 * 1024
# Wheels whose presence or version decides which NCCL wins the shared path.
_TRACKED_DISTRIBUTIONS = (
    "torch",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
    "nvidia-cudnn-cu13",
)
_NCCL_RELATIVE_PATH = "nvidia/nccl/lib/libnccl.so.2"
_TORCH_RELATIVE_PATH = "torch/lib/libtorch_cuda.so"

_PROBE_SCHEMA = pa.schema(
    [
        pa.field("index", pa.int64(), nullable=False),
        pa.field("host", pa.string(), nullable=False),
        pa.field("node", pa.string(), nullable=True),
        pa.field("versions", pa.string(), nullable=False),
        pa.field("venv_nccl", pa.string(), nullable=False),
        pa.field("venv_torch_cuda", pa.string(), nullable=False),
        pa.field("cache_nccl", pa.string(), nullable=False),
        pa.field("cache_torch_cuda", pa.string(), nullable=False),
        pa.field("torch_version", pa.string(), nullable=False),
        pa.field("import_error", pa.string(), nullable=True),
    ]
)

_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="16g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="7g", disk="4g")
_MAX_WORKERS = 28
_HEARTBEAT_TIMEOUT = 30 * 60


class ProbeReport(BaseModel):
    """Which NCCL and torch builds the shared node caches hold, and what each venv picked."""

    version: str = "v2"
    tasks: int
    import_failures: int
    by_versions: dict[str, int]
    venv_nccl: dict[str, int]
    venv_torch_cuda: dict[str, int]
    cache_nccl_by_node: dict[str, str]
    cache_torch_cuda_by_node: dict[str, str]
    import_errors: dict[str, int]


def _exports_symbol(path: str) -> bool:
    """Whether ``path``'s ELF string tables contain ``ncclCommResume``.

    A chunked byte scan rather than ``nm``: the task image carries no binutils, these libraries run
    to a gigabyte, and a dynamic symbol name is stored verbatim in ``.dynstr`` -- so a hit is
    exactly what the loader resolves against and a miss is exactly what makes it raise.
    """
    overlap = len(_SYMBOL) - 1
    tail = b""
    with open(path, "rb") as handle:
        while chunk := handle.read(_SCAN_CHUNK):
            if _SYMBOL in tail + chunk:
                return True
            tail = chunk[-overlap:]
    return False


def _distribution_of(archive_dir: str) -> str:
    """The wheel an unpacked uv cache archive came from, read off its ``.dist-info``."""
    for entry in sorted(os.listdir(archive_dir)):
        if entry.endswith(".dist-info"):
            return entry.removesuffix(".dist-info")
    return f"unknown({os.path.basename(archive_dir)})"


def _describe(path: str) -> str:
    """``<distribution>:<has|missing>`` for one unpacked library, or why it could not be read."""
    if not os.path.exists(path):
        return "absent"
    real = os.path.realpath(path)
    # The archive root is the ancestor holding the wheel's top-level entries; walk up from the
    # library to the directory whose parent is the cache's archive store.
    archive = real
    while archive != "/" and os.path.basename(os.path.dirname(archive)) != "archive-v0":
        archive = os.path.dirname(archive)
    distribution = _distribution_of(archive) if archive != "/" else f"unpacked({os.path.dirname(real)})"
    return f"{distribution}:{'has' if _exports_symbol(real) else 'missing'}"


def _cache_inventory(cache_dir: str, library: str) -> list[str]:
    """Every unpacked copy of ``library`` in the node's shared uv cache, by distribution.

    The cache is a node-local hostPath every pod on the node shares and nothing ever prunes, so
    this is the full menu a venv's symlinks can be drawn from.
    """
    name = os.path.basename(library)
    found: set[str] = set()
    for root, _, files in os.walk(cache_dir):
        if name in files and root.endswith(os.path.dirname(library)):
            found.add(_describe(os.path.join(root, name)))
    return sorted(found)


def probe(work: dict) -> Iterator[dict]:
    """Report one task's view of the venv and its node's cache, then try the production import."""
    site_packages = os.path.join(
        sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
    )
    cache_dir = os.environ.get("UV_CACHE_DIR", "/uv/cache")

    import importlib.metadata as metadata  # noqa: PLC0415

    versions = []
    for name in _TRACKED_DISTRIBUTIONS:
        try:
            versions.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{name}==absent")

    import_error = None
    torch_version = "unimported"
    try:
        import torch  # noqa: PLC0415

        torch_version = f"{torch.__version__} cuda={torch.version.cuda}"
    except Exception as error:
        import_error = f"{type(error).__name__}: {error}"

    yield {
        "index": work["index"],
        "host": socket.gethostname(),
        "node": os.environ.get("IRIS_NODE_NAME") or os.environ.get("NODE_NAME"),
        "versions": " ".join(versions),
        "venv_nccl": _describe(os.path.join(site_packages, _NCCL_RELATIVE_PATH)),
        "venv_torch_cuda": _describe(os.path.join(site_packages, _TORCH_RELATIVE_PATH)),
        "cache_nccl": " | ".join(_cache_inventory(cache_dir, _NCCL_RELATIVE_PATH)) or "none",
        "cache_torch_cuda": " | ".join(_cache_inventory(cache_dir, _TORCH_RELATIVE_PATH)) or "none",
        "torch_version": torch_version,
        "import_error": import_error,
    }


def _summarize(rows: list[dict]) -> ProbeReport:
    """Tally what each venv picked, and inventory each node's cache once."""
    by_versions: Counter = Counter()
    venv_nccl: Counter = Counter()
    venv_torch_cuda: Counter = Counter()
    import_errors: Counter = Counter()
    cache_nccl_by_node: dict[str, str] = {}
    cache_torch_cuda_by_node: dict[str, str] = {}

    for row in rows:
        by_versions[row["versions"]] += 1
        venv_nccl[row["venv_nccl"]] += 1
        venv_torch_cuda[row["venv_torch_cuda"]] += 1
        node = row["node"] or row["host"]
        cache_nccl_by_node[node] = row["cache_nccl"]
        cache_torch_cuda_by_node[node] = row["cache_torch_cuda"]
        if row["import_error"]:
            import_errors[row["import_error"][:200]] += 1

    return ProbeReport(
        tasks=len(rows),
        import_failures=sum(import_errors.values()),
        by_versions=dict(by_versions),
        venv_nccl=dict(venv_nccl),
        venv_torch_cuda=dict(venv_torch_cuda),
        cache_nccl_by_node=cache_nccl_by_node,
        cache_torch_cuda_by_node=cache_torch_cuda_by_node,
        import_errors=dict(import_errors),
    )


def run_probe(output_path: str) -> ProbeReport:
    """Fan out over the control run's task shape and tally what each venv resolved."""
    work = [{"index": index} for index in range(PROBE_TASKS)]
    rows_dir = prefix_join(output_path, "outputs/probe")
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
        name="nccl-import-probe",
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

    report = _summarize(rows)
    logger.info("=== NCCL IMPORT PROBE ===")
    logger.info("  tasks: %s  import_failures: %s", report.tasks, report.import_failures)
    logger.info("  by_versions: %s", report.by_versions)
    logger.info("  venv_nccl: %s", report.venv_nccl)
    logger.info("  venv_torch_cuda: %s", report.venv_torch_cuda)
    logger.info("  import_errors: %s", report.import_errors)
    for node, inventory in sorted(report.cache_nccl_by_node.items()):
        logger.info("  cache nccl      %s: %s", node, inventory)
    for node, inventory in sorted(report.cache_torch_cuda_by_node.items()):
        logger.info("  cache torch     %s: %s", node, inventory)
    return report


def probe_step() -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/nccl_import_probe",
        deps=[],
        hash_attrs={"tasks": PROBE_TASKS, "attempt": 2},
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
