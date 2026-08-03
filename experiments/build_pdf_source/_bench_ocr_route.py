# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- run the OCR extraction route over a small sample on one GB200 node.

DELETE once the numbers are recorded. Nothing in the pipeline imports this.

Drives the real sender (:func:`experiments.build_pdf_source.extract_ocr.ocr_batch`) against a real
four-instance fleet, over a bounded number of source shards, and records what a full run would need
to be planned from: end-to-end pages/s and pages/s/GPU, the per-page latency distribution, the
effective-DPI distribution, completion-token cost, and every way a page or a document was lost.

Two modes:

``--preflight`` is CPU-only. It reports how many OCR-route documents and pages a shard holds (so the
benchmark can be sized), measures single-core render throughput for the CPU:GPU ratio, and checks
that the samples can be handed back. Run it first; it costs a couple of minutes and no GPU.

``--bench`` starts the fleet and runs the pipeline, then writes a handful of PDF/Markdown pairs so
the output can be read rather than only counted.

Submit::

    iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name ocr-route-preflight \\
        -- python -m experiments.build_pdf_source._bench_ocr_route --preflight
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source import extract_ocr
from experiments.build_pdf_source.classify import classify_step, model_step, routing_keys
from experiments.build_pdf_source.common import PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.ocr_extract import fleet
from experiments.build_pdf_source.ocr_extract.client import OcrEndpoint, PageOcr
from experiments.build_pdf_source.ocr_extract.render import RenderOptions
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Samples and reports go next to the run's own output, in CoreWeave object storage. That is
# readable from a dev box with CW_KEY_ID / CW_KEY_SECRET out of GCP Secret Manager
# (``cw-object-storage-key-{id,secret}`` in ``hai-gcp-models``), so nothing has to cross clouds.
SAMPLE_COUNT = 10
# Small enough to move off-cluster cheaply. The OCR route is scans, so the median document is far
# bigger than this; the samples are deliberately from the small end.
SAMPLE_MAX_PDF_BYTES = 600_000

# Sized from the preflight: enough shards for roughly two thousand OCR-route documents.
BENCH_SHARDS = 40

_PREFLIGHT_SHARDS = 3
_PREFLIGHT_RENDER_PAGES = 60
# Source shards scanned for sample candidates. Each is hundreds of MB, and a few hundred candidates
# is far more than the ten smallest are drawn from.
_SAMPLE_SCAN_SHARDS = 4

# Per-page latency histogram. Zephyr counters only sum, so the distribution comes from cumulative
# bucket counts; these bounds bracket the measured p50 of ~21s at the fleet's operating point.
_LATENCY_BUCKETS = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0)

_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]

_PREFLIGHT_RESOURCES = ResourceConfig(cpu=4, ram="16g", disk="16g")
_BENCH_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="16g")


class PreflightReport(BaseModel):
    version: str = "v1"
    source_dir: str
    classification_dir: str
    source_shards: int
    ocr_route_documents: int
    ocr_route_pages: int
    mean_pages_per_document: float
    ocr_documents_per_shard: float
    shards_for_two_thousand: int
    render_pages_per_core_second: float
    render_dpi_percentiles: dict[str, float]
    pages_below_legibility_floor: int
    sample_channel: str


class BenchReport(BaseModel):
    version: str = "v1"
    output_dir: str
    sample_dir: str
    gpus: int
    shards: int
    documents: int
    pages_ocred: int
    fleet_startup_seconds: float
    pipeline_wall_seconds: float
    pages_per_second: float
    pages_per_second_per_gpu: float
    gpu_hours_per_million_pages: float
    documents_per_second: float
    completion_tokens: int
    completion_tokens_per_second: float
    page_latency: dict[str, float]
    counters: dict[str, int | float]


def _publish(report: BaseModel, name: str, output_path: str) -> None:
    """Write a report next to the run's output, as plain JSON.

    The step already returns it as an artifact, but an artifact has to be read back through
    ``read_artifact``; a JSON file at a predictable path can be fetched with one command. Job logs
    are not a substitute -- log retrieval for a federated job has proven unreliable.
    """
    print(f"{name.upper()}_RESULT " + report.model_dump_json(), flush=True)
    destination = prefix_join(output_path, f"{name}-report.json")
    StoragePath(destination).write_bytes(report.model_dump_json(indent=2).encode("utf-8"))
    logger.info("Wrote %s report to %s", name, destination)


def _source_shards(source_dir: str) -> list[str]:
    filesystem, path = url_to_fs(source_dir)
    return sorted(f"s3://{shard}" for shard in filesystem.glob(f"{path}/*.parquet"))


def _read_parquet(url: str, columns: list[str] | None = None) -> pa.Table:
    """Read a Parquet file through fsspec.

    Not ``pq.read_table(url)``: the CoreWeave endpoint reaches this process as an ``FSSPEC_S3``
    block, which configures fsspec and not PyArrow's own S3 filesystem, so handing PyArrow an
    ``s3://`` URL sends it at AWS proper and fails in HeadObject with a bare 400.
    """
    filesystem, path = url_to_fs(url)
    with filesystem.open(path, "rb") as stream:
        return pq.read_table(stream, columns=columns)


def _percentiles(values: list[float], points: tuple[int, ...] = (5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{point}": round(ordered[min(len(ordered) - 1, max(0, round(point / 100 * (len(ordered) - 1))))], 2)
        for point in points
    }


def _latency_percentiles(tallies: dict[str, int | float]) -> dict[str, float]:
    """Recover an approximate latency distribution from the cumulative bucket counters."""
    total = int(tallies.get("ocr_bench/page_latency_count", 0))
    if not total:
        return {}
    seen = 0
    result: dict[str, float] = {}
    targets = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    for bound in _LATENCY_BUCKETS:
        seen += int(tallies.get(f"ocr_bench/page_latency_le_{bound:g}s", 0))
        for name, fraction in list(targets.items()):
            if seen >= fraction * total:
                result[name] = bound
                del targets[name]
    for name in targets:
        result[name] = float("inf")
    result["mean"] = round(float(tallies.get("ocr_bench/page_latency_seconds", 0.0)) / total, 2)
    return result


def _timed_ocr_page(endpoint: OcrEndpoint, connections: int, page) -> PageOcr:
    """The real page request, with its latency recorded into counters."""
    from experiments.build_pdf_source.ocr_extract.client import ocr_page  # noqa: PLC0415

    start = time.monotonic()
    try:
        return ocr_page(endpoint, connections, page)
    finally:
        elapsed = time.monotonic() - start
        counters.pipeline.update_counter("ocr_bench/page_latency_seconds", elapsed)
        counters.pipeline.update_counter("ocr_bench/page_latency_count", 1)
        for bound in _LATENCY_BUCKETS:
            if elapsed <= bound:
                counters.pipeline.update_counter(f"ocr_bench/page_latency_le_{bound:g}s", 1)
                break
        else:
            counters.pipeline.update_counter("ocr_bench/page_latency_over", 1)


def timed_ocr_batch(batch: pa.RecordBatch, **kwargs) -> Iterator[dict]:
    """The real sender, with the page request wrapped so latencies are recorded.

    Patching the module attribute is what ``ocr_batch`` resolves at submit time, so this measures
    the production path rather than a copy of it.
    """
    extract_ocr.ocr_page = _timed_ocr_page
    yield from extract_ocr.ocr_batch(batch, **kwargs)


def preflight(output_path: str, source_output_path: str, classification_output_path: str) -> PreflightReport:
    """Size the benchmark and confirm the samples can be handed back, without touching a GPU."""

    from experiments.build_pdf_source.ocr_extract.render import (  # noqa: PLC0415
        iter_rendered_pages,
        open_pdf,
    )

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)

    filesystem, path = url_to_fs(classification.main_output_dir)
    pages = []
    ocr_documents = 0
    for shard in sorted(filesystem.glob(f"{path}/*.parquet")):
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=["needs_ocr", "num_pages"])
        needs_ocr = table.column("needs_ocr").to_pylist()
        page_counts = table.column("num_pages").to_pylist()
        for flag, count in zip(needs_ocr, page_counts, strict=True):
            if flag is True:
                ocr_documents += 1
                pages.append(count or 0)
    total_pages = int(sum(pages))

    shards = _source_shards(source.main_output_dir)
    keys = routing_keys(classification.main_output_dir, needs_ocr=True)

    # How much of a source shard is on the OCR route, and what a core renders per second.
    options = RenderOptions()
    sampled_documents = 0
    rendered = 0
    dpis: list[float] = []
    render_seconds = 0.0
    sample_rows: list[dict] = []
    for shard in shards[:_PREFLIGHT_SHARDS]:
        table = _read_parquet(shard, _SOURCE_COLUMNS)
        for row in table.to_pylist():
            if (row["warc_filename"], row["warc_record_offset"]) not in keys:
                continue
            sampled_documents += 1
            if len(row["pdf"]) <= SAMPLE_MAX_PDF_BYTES and len(sample_rows) < SAMPLE_COUNT:
                sample_rows.append(row)
            if rendered >= _PREFLIGHT_RENDER_PAGES:
                continue
            start = time.monotonic()
            try:
                with open_pdf(row["pdf"]) as document:
                    for page in iter_rendered_pages(document, options):
                        dpis.append(page.dpi)
                        rendered += 1
                        if rendered >= _PREFLIGHT_RENDER_PAGES:
                            break
            except Exception:
                logger.warning("Could not render %s", row["url"], exc_info=True)
            render_seconds += time.monotonic() - start

    per_shard = sampled_documents / _PREFLIGHT_SHARDS

    # Confirm the samples can be written before spending a GPU-hour finding out they cannot.
    channel = prefix_join(output_path, "samples")
    StoragePath(prefix_join(channel, "_probe.txt")).write_bytes(b"ocr-route-preflight\n")

    report = PreflightReport(
        source_dir=source.main_output_dir,
        classification_dir=classification.main_output_dir,
        source_shards=len(shards),
        ocr_route_documents=ocr_documents,
        ocr_route_pages=total_pages,
        mean_pages_per_document=round(total_pages / ocr_documents, 2) if ocr_documents else 0.0,
        ocr_documents_per_shard=round(per_shard, 1),
        shards_for_two_thousand=max(1, round(2000 / per_shard)) if per_shard else 0,
        render_pages_per_core_second=round(rendered / render_seconds, 2) if render_seconds else 0.0,
        render_dpi_percentiles=_percentiles(dpis),
        pages_below_legibility_floor=sum(1 for dpi in dpis if dpi < options.legibility_floor_dpi),
        sample_channel=channel,
    )
    logger.info("PREFLIGHT %s", report.model_dump_json(indent=2))
    _publish(report, "preflight", output_path)
    return report


def _write_samples(output_dir: str, source_shards: list[str], sample_dir: str) -> int:
    """Write a few PDF/Markdown pairs so the output can be read, not only counted."""
    filesystem, path = url_to_fs(output_dir)
    # Every output row, because the grouping stage sorts by ``id``: an output shard's documents come
    # from arbitrary source shards, so a partial read would leave most source rows unmatched.
    wanted: dict[tuple[str, int], dict] = {}
    for shard in sorted(filesystem.glob(f"{path}/*.parquet")):
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream)
        for row in table.to_pylist():
            wanted[(row["warc_filename"], row["warc_record_offset"])] = row
    logger.info("Matching samples against %d extracted documents", len(wanted))

    # Take the smallest PDFs rather than the first ones that fit: these have to move off-cluster,
    # and the OCR route is scans, so an arbitrary document is tens of megabytes.
    candidates: list[tuple[int, bytes, dict]] = []
    for shard in source_shards[:_SAMPLE_SCAN_SHARDS]:
        table = _read_parquet(shard, ["pdf", "warc_filename", "warc_record_offset"])
        for row in table.to_pylist():
            record = wanted.get((row["warc_filename"], row["warc_record_offset"]))
            if record is not None:
                candidates.append((len(row["pdf"]), row["pdf"], record))
    candidates.sort(key=lambda candidate: candidate[0])
    logger.info(
        "%d sample candidates; smallest %s bytes",
        len(candidates),
        [size for size, _, _ in candidates[:SAMPLE_COUNT]],
    )

    _METADATA_FIELDS = (
        "id",
        "url",
        "num_pages",
        "pages_ocred",
        "pages_failed",
        "pages_unrendered",
        "mean_render_dpi",
        "pages_below_legibility_floor",
        "completion_tokens",
        "extraction_status",
        "extraction_error",
        "boilerplate_lines_removed",
    )
    written = 0
    for size, pdf, record in candidates:
        if written >= SAMPLE_COUNT or size > SAMPLE_MAX_PDF_BYTES:
            break
        name = record["id"][:16]
        StoragePath(prefix_join(sample_dir, f"{name}.pdf")).write_bytes(pdf)
        StoragePath(prefix_join(sample_dir, f"{name}.md")).write_bytes(record["text"].encode("utf-8"))
        metadata = {field: record[field] for field in _METADATA_FIELDS}
        StoragePath(prefix_join(sample_dir, f"{name}.json")).write_bytes(
            json.dumps(metadata, indent=2, default=str).encode("utf-8")
        )
        written += 1
    logger.info("Wrote %d PDF/Markdown pairs to %s", written, sample_dir)
    return written


def bench(output_path: str, source_output_path: str, classification_output_path: str) -> BenchReport:
    """Run the real OCR route over a bounded slice of the corpus against a real fleet."""
    from marin.datakit.normalize import make_split_writer  # noqa: PLC0415
    from marin.inference.iris import remote_inference  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=True)
    shards = _source_shards(source.main_output_dir)[:BENCH_SHARDS]
    logger.info("Benchmarking over %d source shards", len(shards))

    startup_start = time.monotonic()
    with remote_inference(fleet.build_inference_config()) as session:
        startup_seconds = time.monotonic() - startup_start
        endpoint = OcrEndpoint(
            base_url=session.model.endpoint.base_url,
            model=session.model.endpoint.model,
            max_visual_tokens=extract_ocr.RENDER_OPTIONS.max_visual_tokens,
        )
        logger.info("Fleet ready in %.0fs at %s (%s)", startup_seconds, endpoint.base_url, session.backend_name)

        pipeline = (
            Dataset.from_list(shards)
            .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
            .flat_map(
                partial(
                    timed_ocr_batch,
                    keys=keys,
                    endpoint=endpoint,
                    render_options=extract_ocr.RENDER_OPTIONS,
                    boilerplate=extract_ocr.BOILERPLATE_OPTIONS,
                )
            )
            .group_by(
                key=lambda record: record["id"],
                reducer=extract_ocr._keep_all,
                sort_by=lambda record: record["id"],
                num_output_shards=len(shards),
            )
            .map_shard(make_split_writer(output_path, output_schema=extract_ocr._OUTPUT_SCHEMA))
        )
        pipeline_start = time.monotonic()
        outcome = ZephyrContext(
            name="ocr-route-bench",
            resources=extract_ocr._WORKER_RESOURCES,
            max_workers=extract_ocr.sender_fleet_size(fleet.INSTANCES)[1],
            stage_runner_factory=SubprocessRunner,
            map_task_resources=extract_ocr._MAP_TASK_RESOURCES,
            heartbeat_timeout=extract_ocr._HEARTBEAT_TIMEOUT,
        ).execute(pipeline)
        wall = time.monotonic() - pipeline_start
        session.check_alive()

    tallies = dict(outcome.counters)
    documents = int(tallies.get("focus_crawl_pdf_ocr/extracted", 0))
    pages = int(tallies.get("focus_crawl_pdf_ocr/extracted_pages", 0))
    tokens = int(tallies.get("focus_crawl_pdf_ocr/completion_tokens", 0))
    gpus = fleet.INSTANCES

    main_dir = prefix_join(output_path, "outputs/main")
    sample_dir = prefix_join(output_path, "samples")
    try:
        _write_samples(main_dir, shards, sample_dir)
    except Exception:
        logger.warning("Could not write samples to %s", sample_dir, exc_info=True)
        sample_dir = "unavailable"

    report = BenchReport(
        output_dir=main_dir,
        sample_dir=sample_dir,
        gpus=gpus,
        shards=len(shards),
        documents=documents,
        pages_ocred=pages,
        fleet_startup_seconds=round(startup_seconds, 1),
        pipeline_wall_seconds=round(wall, 1),
        pages_per_second=round(pages / wall, 2) if wall else 0.0,
        pages_per_second_per_gpu=round(pages / wall / gpus, 2) if wall else 0.0,
        gpu_hours_per_million_pages=round(gpus * wall / pages * 1e6 / 3600, 1) if pages else 0.0,
        documents_per_second=round(documents / wall, 3) if wall else 0.0,
        completion_tokens=tokens,
        completion_tokens_per_second=round(tokens / wall, 1) if wall else 0.0,
        page_latency=_latency_percentiles(tallies),
        counters=tallies,
    )
    logger.info("BENCH %s", report.model_dump_json(indent=2))
    _publish(report, "bench", output_path)
    return report


def _step(name: str, function, source: StepSpec, classification: StepSpec, resources: ResourceConfig) -> StepSpec:
    return StepSpec(
        name=f"data/datakit/validate/{name}",
        deps=[source, classification],
        hash_attrs={
            "shards": BENCH_SHARDS,
            "max_visual_tokens": extract_ocr.RENDER_OPTIONS.max_visual_tokens,
            "attempt": 6,
        },
        fn=remote(
            partial(
                function,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
            ),
            resources=resources,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="CPU-only sizing pass; no GPUs")
    parser.add_argument("--bench", action="store_true", help="start the fleet and run the route")
    args = parser.parse_args()
    if not (args.preflight or args.bench):
        raise SystemExit("pass --preflight or --bench")

    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    classify = classify_step(fetch, model_step())

    steps = []
    if args.preflight:
        steps.append(_step("ocr_route_preflight", preflight, fetch, classify, _PREFLIGHT_RESOURCES))
    if args.bench:
        steps.append(_step("ocr_route_bench", bench, fetch, classify, _BENCH_DRIVER_RESOURCES))
    StepRunner().run(steps)


if __name__ == "__main__":
    main()
