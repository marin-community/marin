# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- can an OCR sender run on a GB200 node's Grace (aarch64) cores?

DELETE once the answer is recorded. Nothing in the pipeline imports this.

The full-sample OCR run wants more sender CPU than the x86 pool holds, and the obvious spare
capacity is the GB200 nodes' own Grace cores -- 16 per node beside a full vLLM fleet, 144 on every
idle node. Nothing has ever run the datakit stack there, and Iris has no architecture constraint to
steer with (``is_cpu_device_type_constraint`` strips ``device_type=cpu`` before routing, so a CPU
task lands wherever it fits). This probe answers, cheaply and separately from any benchmark:

1. Does ``uv sync`` resolve and install the ``datakit`` group on aarch64 at all?
2. Do the sender's imports (pymupdf, pyarrow, the OpenAI client) load?
3. What does a Grace core render, in pages per core-second, against the 13.2 measured on x86?

Placement is forced the only way it can be: by asking for ``cpu=80``, which no cpu-erapids node
(64 vCPU) can satisfy, so the task can only fit a gb200-4x node (144 vCPU). If this job sits
PENDING forever, that itself is an answer -- CPU tasks cannot land on the GPU pool.

Submit::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name arm-render-probe \\
        -- python -m experiments.build_pdf_source._probe_arm_render
"""

import importlib
import logging
import platform
import time
from functools import partial

from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.common import PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# cpu=80 is the placement mechanism, not a need: it exceeds a cpu-erapids node's 64 vCPU, so the
# scheduler's only fit is a gb200-4x node. The probe itself uses a handful of cores.
_PROBE_RESOURCES = ResourceConfig(cpu=80, ram="64g", disk="32g")

_RENDER_PAGES = 120
_THREAD_RENDER_PAGES = 240
_RENDER_THREADS = 8


class ArmProbeReport(BaseModel):
    version: str = "v1"
    machine: str
    processor: str
    node: str
    imports: dict[str, str]
    render_pages_per_core_second: float
    threaded_pages_per_second: float
    threaded_threads: int
    pages_rendered: int


def _import_versions() -> dict[str, str]:
    """Import everything a sender touches, recording versions or the failure."""
    versions: dict[str, str] = {}
    for module_name in ("pymupdf", "pyarrow", "openai", "PIL", "fsspec", "zephyr"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "ok")
        except Exception as error:
            versions[module_name] = f"FAILED: {type(error).__name__}: {error}"
    return versions


def probe(output_path: str, source_output_path: str) -> ArmProbeReport:
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import pyarrow.parquet as pq  # noqa: PLC0415
    from rigging.filesystem import StoragePath, prefix_join, url_to_fs  # noqa: PLC0415

    from experiments.build_pdf_source.ocr_extract.render import (  # noqa: PLC0415
        RenderOptions,
        iter_rendered_pages,
        open_pdf,
    )

    machine = platform.machine()
    logger.info("Probe running on machine=%s node=%s", machine, platform.node())
    versions = _import_versions()
    logger.info("Imports: %s", versions)

    source = read_artifact(source_output_path, PdfSourceData)
    filesystem, path = url_to_fs(source.main_output_dir)
    shard = sorted(filesystem.glob(f"{path}/*.parquet"))[0]
    with filesystem.open(shard, "rb") as stream:
        table = pq.read_table(stream, columns=["pdf", "url"])
    rows = table.to_pylist()
    logger.info("Read %d rows from %s", len(rows), shard)

    options = RenderOptions()

    def render_rows(page_budget: int) -> tuple[int, float]:
        rendered = 0
        start = time.monotonic()
        for row in rows:
            if rendered >= page_budget:
                break
            try:
                with open_pdf(row["pdf"]) as document:
                    for _page in iter_rendered_pages(document, options):
                        rendered += 1
                        if rendered >= page_budget:
                            break
            except Exception:
                logger.warning("Could not render %s", row["url"], exc_info=True)
        return rendered, time.monotonic() - start

    # Single core, comparable to the preflight's 13.2 pages/core-s on x86.
    rendered, elapsed = render_rows(_RENDER_PAGES)
    single = rendered / elapsed if elapsed else 0.0
    logger.info("Single-thread render: %d pages in %.1fs (%.2f pages/s)", rendered, elapsed, single)

    # Threaded, because the senders render from a thread pool and MuPDF holds the GIL differently
    # across builds; per-thread scaling on Grace is worth one number.
    chunk = len(rows) // _RENDER_THREADS or 1
    slices = [rows[i * chunk : (i + 1) * chunk] for i in range(_RENDER_THREADS)]
    per_thread_budget = _THREAD_RENDER_PAGES // _RENDER_THREADS

    def render_slice(slice_rows: list[dict]) -> int:
        done = 0
        for row in slice_rows:
            if done >= per_thread_budget:
                break
            try:
                with open_pdf(row["pdf"]) as document:
                    for _page in iter_rendered_pages(document, options):
                        done += 1
                        if done >= per_thread_budget:
                            break
            except Exception:
                logger.debug("Could not render %s", row["url"], exc_info=True)
        return done

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=_RENDER_THREADS) as pool:
        threaded_total = sum(pool.map(render_slice, slices))
    threaded_elapsed = time.monotonic() - start
    threaded = threaded_total / threaded_elapsed if threaded_elapsed else 0.0
    logger.info(
        "%d-thread render: %d pages in %.1fs (%.2f pages/s)",
        _RENDER_THREADS,
        threaded_total,
        threaded_elapsed,
        threaded,
    )

    report = ArmProbeReport(
        machine=machine,
        processor=platform.processor() or "unknown",
        node=platform.node(),
        imports=versions,
        render_pages_per_core_second=round(single, 2),
        threaded_pages_per_second=round(threaded, 2),
        threaded_threads=_RENDER_THREADS,
        pages_rendered=rendered + threaded_total,
    )
    print("ARM_PROBE_RESULT " + report.model_dump_json(), flush=True)
    StoragePath(prefix_join(output_path, "arm-probe-report.json")).write_bytes(
        report.model_dump_json(indent=2).encode("utf-8")
    )
    return report


def main() -> None:
    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    # classify/model are built only so the DAG's cached deps resolve identically to other probes.
    classify_step(fetch, model_step())
    step = StepSpec(
        name="data/datakit/validate/arm_render_probe",
        deps=[fetch],
        hash_attrs={"attempt": 1},
        fn=remote(
            partial(probe, source_output_path=fetch.output_path),
            resources=_PROBE_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    StepRunner().run([step])


if __name__ == "__main__":
    main()
