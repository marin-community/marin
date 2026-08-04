# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- how many GB200 instances can one broker + proxy actually feed?

DELETE once the number is recorded. Nothing in the pipeline imports this.

The full-sample OCR run wants ~172 GPUs. The measured operating point (4 instances, one broker,
47.6 pages/s) says nothing about where the *broker* saturates: every request body -- ~1.9 MB of
base64 PNG -- passes through one Python proxy process and one Python broker process, and the plan
is to shard the fleet into however many independent broker+fleet sessions that ceiling demands.
This bench runs the production sender against **32 instances behind a single broker** and reports
end-to-end pages/s. Two outcomes:

* ~568 pages/s (32 x 17.75): the broker is not the limit at this size; the full run shards as
  ~6 fleets of ~29 and nothing more needs measuring.
* materially less: halve the per-broker shard and rerun.

Differences from ``_bench_ocr_route`` (which this reuses), each with a reason:

* ``instances=32`` and everything derived from it (client concurrency 16,384; proxy
  ``max_pending_requests`` 32,768).
* The sender fleet is sized by offered rate -- 384 tasks x ~1.5 pages/s -- because attempt 1
  proved the task, not the broker, was the limit (see the sizing comment below). The workers
  spill onto the GB200 nodes' Grace cores, which ``_probe_arm_render`` validated.
* ``broker_resources`` raised to 8 CPU / 64 GB: at 16,384 in flight the broker holds ~31 GB of
  leased request payloads, and the stock 2 CPU / 8 GB would OOM before measuring anything.
* The driver gets 12 CPU / 96 GB for the same reason -- the proxy parks one thread per in-flight
  request, each holding its body until the response lands.
* The proxy's two INFO lines per request would be ~1,100 log lines/s at the target rate; the
  driver quiets that logger to WARNING before the fleet starts.

Submit::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name broker-ceiling-bench \\
        -- python -m experiments.build_pdf_source._bench_broker_ceiling
"""

import dataclasses
import logging
import time
from collections.abc import Iterator
from functools import partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source import extract_ocr
from experiments.build_pdf_source._bench_ocr_route import (
    BenchReport,
    _latency_percentiles,
    _publish,
    _source_shards,
    timed_ocr_batch,
)
from experiments.build_pdf_source.classify import classify_step, model_step, routing_keys
from experiments.build_pdf_source.common import PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.ocr_extract import fleet
from experiments.build_pdf_source.ocr_extract.client import OcrEndpoint
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# The point under test: one broker, one proxy, this many one-GPU instances.
INSTANCES = 32
CLIENT_CONCURRENCY = fleet.MAX_IN_FLIGHT * INSTANCES  # 16,384

# Sender fleet sized by OFFERED RATE, not by in-flight capacity. Attempt 1 sized it to hold the
# in-flight budget from 128 tasks x 128 threads and measured why that is wrong twice over: a task
# delivers ~1.5 pages/s at 64 threads (bench4: 47.6 p/s / 32 tasks) and ~0.7 at 128 (attempt 1:
# 4.6 docs/s from 128 tasks) -- the render loop, the 2 MB JSON/base64 encodes, and 100+ waking
# threads all contend for one cgroup-throttled CPU, so more threads make a task slower. The
# preflight's 13.2 pages/core-s was measured unthrottled and single-threaded and does not describe
# a task. So: the proven 64-thread shape, and enough tasks that 1.5 p/s each offers ~576 p/s --
# right at 32 instances' saturation -- pushing ~1.1 GB/s through the single broker under test.
# 384 tasks no longer fit the x86 pool; most workers land on the GB200 nodes' Grace cores, which
# _probe_arm_render validated (aarch64 installs, imports, renders at x86 speed).
REQUEST_THREADS = 64
SENDER_TASKS = 384
TASKS_PER_WORKER = 8
MAX_WORKERS = SENDER_TASKS // TASKS_PER_WORKER  # 48

# Two waves over the 384 task slots. ~40 OCR docs / ~800 OCR pages per shard puts this near 600k
# pages: ~18 minutes at the offered rate, long enough that startup and drain are noise.
BENCH_SHARDS = 768

_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]

_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_WORKER_RESOURCES = ResourceConfig(cpu=TASKS_PER_WORKER, ram="40g", disk="32g")
_HEARTBEAT_TIMEOUT = 30 * 60

# The proxy parks one thread per in-flight request, each holding its ~1.9 MB body until the
# response lands: ~31 GB of payload at full depth, before Python overhead.
_DRIVER_RESOURCES = ResourceConfig(cpu=12, ram="96g", disk="32g")
_BROKER_RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="64g", disk="20g", preemptible=False)


class CeilingReport(BenchReport):
    instances: int
    client_concurrency: int
    sender_tasks: int
    request_threads: int


def ceiling_ocr_batch(batch: pa.RecordBatch, **kwargs) -> Iterator[dict]:
    """The production sender pinned to the 64-thread task shape under test.

    The constants are module globals that ``ocr_batch`` reads at call time, and this function runs
    in the worker process, so patching here reshapes every task in that process. ``_request_pool``
    is keyed by its thread count, so the pool matches the patched width.
    """
    extract_ocr._REQUEST_THREADS = REQUEST_THREADS
    extract_ocr._PAGES_IN_FLIGHT = 2 * REQUEST_THREADS
    yield from timed_ocr_batch(batch, **kwargs)


def build_ceiling_config():
    """The production fleet config, scaled to the point under test."""
    config = fleet.build_inference_config()
    return dataclasses.replace(
        config,
        instances=INSTANCES,
        broker=dataclasses.replace(
            config.broker,
            broker_resources=_BROKER_RESOURCES,
            proxy=dataclasses.replace(
                config.broker.proxy,
                max_pending_requests=CLIENT_CONCURRENCY * 2,
            ),
        ),
    )


def bench(output_path: str, source_output_path: str, classification_output_path: str) -> CeilingReport:
    from marin.datakit.normalize import make_split_writer  # noqa: PLC0415
    from marin.inference.iris import remote_inference  # noqa: PLC0415

    # The proxy logs two INFO lines per request from this process; at the target rate that is
    # ~1,100 lines/s of pure overhead in the middle of the thing being measured.
    logging.getLogger("marin.inference.proxy").setLevel(logging.WARNING)

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=True)
    shards = _source_shards(source.main_output_dir)[:BENCH_SHARDS]
    logger.info("Ceiling bench: %d instances, one broker, %d source shards", INSTANCES, len(shards))

    startup_start = time.monotonic()
    with remote_inference(build_ceiling_config()) as session:
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
                    ceiling_ocr_batch,
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
            name="broker-ceiling",
            resources=_WORKER_RESOURCES,
            max_workers=MAX_WORKERS,
            stage_runner_factory=SubprocessRunner,
            heartbeat_timeout=_HEARTBEAT_TIMEOUT,
        ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
        wall = time.monotonic() - pipeline_start
        lost_pages = int(outcome.counters.get("focus_crawl_pdf_ocr/page_request_failed", 0))
        try:
            session.check_alive()
        except Exception as error:
            if lost_pages:
                raise
            logger.warning("An inference job ended before the fleet was released: %s", error)

    tallies = dict(outcome.counters)
    documents = int(tallies.get("focus_crawl_pdf_ocr/extracted", 0))
    pages = int(tallies.get("focus_crawl_pdf_ocr/extracted_pages", 0))
    tokens = int(tallies.get("focus_crawl_pdf_ocr/completion_tokens", 0))

    report = CeilingReport(
        instances=INSTANCES,
        client_concurrency=CLIENT_CONCURRENCY,
        sender_tasks=SENDER_TASKS,
        request_threads=REQUEST_THREADS,
        output_dir=output_path,
        sample_dir="not-written",
        gpus=INSTANCES,
        shards=len(shards),
        documents=documents,
        pages_ocred=pages,
        fleet_startup_seconds=round(startup_seconds, 1),
        pipeline_wall_seconds=round(wall, 1),
        pages_per_second=round(pages / wall, 2) if wall else 0.0,
        pages_per_second_per_gpu=round(pages / wall / INSTANCES, 2) if wall else 0.0,
        gpu_hours_per_million_pages=round(INSTANCES * wall / pages * 1e6 / 3600, 1) if pages else 0.0,
        documents_per_second=round(documents / wall, 3) if wall else 0.0,
        completion_tokens=tokens,
        completion_tokens_per_second=round(tokens / wall, 1) if wall else 0.0,
        page_latency=_latency_percentiles(tallies),
        counters=tallies,
    )
    logger.info("CEILING %s", report.model_dump_json(indent=2))
    _publish(report, "ceiling", output_path)
    return report


def main() -> None:
    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    classify = classify_step(fetch, model_step())
    step = StepSpec(
        name="data/datakit/validate/broker_ceiling",
        deps=[fetch, classify],
        hash_attrs={
            "instances": INSTANCES,
            "shards": BENCH_SHARDS,
            "request_threads": REQUEST_THREADS,
            "sender_tasks": SENDER_TASKS,
            "attempt": 3,
        },
        fn=remote(
            partial(
                bench,
                source_output_path=fetch.output_path,
                classification_output_path=classify.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    StepRunner().run([step])


if __name__ == "__main__":
    main()
