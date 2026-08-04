# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- smoke the brokered converter fleet end to end on the cluster.

DELETE once the fleet has run at production scale. Nothing in the pipeline imports this.

The unit tests cover the pool's contract with an in-process broker; what they cannot cover is the
Iris plumbing this module exists to prove: the broker actor handle resolving from a *spawned child
process* on a real pod, converter children inheriting the job environment, readiness gating on the
first fully built docling converter, and a whole PDF round-tripping through proxy -> broker ->
converter as raw bytes. It runs a deliberately tiny fleet (2 pods x 2 converters) over ~24 real
documents from the fetch output plus one poison payload, and fails loudly unless every document
comes back and the poison comes back as ``status: "failure"``.

Run on the same x86 cluster as the extraction comparisons::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name smoke-converter-pool \\
        -- python -m experiments.build_pdf_source._smoke_converter_pool
"""

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pyarrow.parquet as pq
from fray.types import ResourceConfig, create_environment
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.inference.config import BrokerConfig, InferenceProxyConfig, InferenceWorkerConfig
from marin.inference.converter_pool import ConverterPoolConfig, remote_converter_pool
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step, routing_keys
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.service import build_arch_adaptive_handler
from experiments.build_pdf_source.extract_fleet import (
    _LEASE_TIMEOUT,
    _PROXY_READINESS_TIMEOUT,
    _PROXY_REQUEST_TIMEOUT,
    _WORKER_REQUEST_TIMEOUT,
    ARM_LAYOUT_BACKEND,
    MODEL_ID,
    TABLE_BACKEND,
    X86_LAYOUT_BACKEND,
    convert_document,
)
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

_DOCUMENTS = 24
_THREADS = 8
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]

_POISON = {"pdf": b"this is not a pdf", "url": "poison://not-a-pdf"}

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")
_POOL_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="16g")


def _pool_config(layout_model: LayoutModelData) -> ConverterPoolConfig:
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415

    x86_options = ExtractionOptions(
        table_backend=TABLE_BACKEND,
        layout_backend=X86_LAYOUT_BACKEND,
        layout_model_path=layout_model.model_path,
        layout_label_map=layout_model.label_map,
    )
    arm_options = ExtractionOptions(table_backend=TABLE_BACKEND, layout_backend=ARM_LAYOUT_BACKEND)
    return ConverterPoolConfig(
        handler_factory=partial(build_arch_adaptive_handler, x86_options, arm_options),
        model_id=MODEL_ID,
        instances=2,
        processes_per_instance=2,
        worker_resources=_POOL_RESOURCES,
        worker_environment=create_environment(extras=["datakit"]),
        broker=BrokerConfig(
            worker=InferenceWorkerConfig(max_in_flight=1, request_timeout_seconds=_WORKER_REQUEST_TIMEOUT),
            request_lease_timeout_seconds=_LEASE_TIMEOUT,
            proxy=InferenceProxyConfig(
                request_timeout_seconds=_PROXY_REQUEST_TIMEOUT,
                readiness_timeout_seconds=_PROXY_READINESS_TIMEOUT,
            ),
        ),
    )


class SmokeReport(BaseModel):
    documents: int
    successes: int
    failures: int
    total_characters: int
    p50_seconds: float
    max_seconds: float
    poison_status: str
    # How placement split the fleet between layout backends -- the observable proof that the
    # arch-adaptive factory picked per node.
    backends: dict[str, int]


def smoke(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> SmokeReport:
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=False)

    filesystem, path = url_to_fs(source.main_output_dir)
    shard = sorted(filesystem.glob(f"{path}/*.parquet"))[0]
    with filesystem.open(shard, "rb") as stream:
        table = pq.read_table(stream, columns=_SOURCE_COLUMNS)
    rows = [row for row in table.to_pylist() if (row["warc_filename"], row["warc_record_offset"]) in keys][:_DOCUMENTS]
    if len(rows) < _DOCUMENTS:
        raise RuntimeError(f"Only {len(rows)} text-extractable documents in {shard}")

    with remote_converter_pool(_pool_config(layout_model)) as session:
        base_url = session.endpoint.base_url
        logger.info("Pool ready at %s; converting %d documents", base_url, len(rows))
        with ThreadPoolExecutor(max_workers=_THREADS) as pool:
            futures = [pool.submit(convert_document, base_url, _THREADS, row) for row in rows]
            poison_future = pool.submit(convert_document, base_url, _THREADS, _POISON)
            documents = [future.result() for future in futures]
            poison = poison_future.result()
        session.check_alive()

    successes = [document for document in documents if document.status != "failure"]
    failures = [document for document in documents if document.status == "failure"]
    seconds = sorted(document.seconds for document in documents)
    logger.info(
        "Converted %d/%d documents (%d failures); chars total=%d; seconds p50=%.1f max=%.1f",
        len(successes),
        len(documents),
        len(failures),
        sum(len(document.text) for document in successes),
        seconds[len(seconds) // 2],
        seconds[-1],
    )
    for document in failures:
        logger.info("Failure: %s", document.error)
    logger.info("Poison payload came back status=%s error=%s", poison.status, poison.error)
    backends = dict(Counter(document.backend for document in documents))
    logger.info("Backend split: %s", backends)

    if not successes:
        raise RuntimeError("No document converted successfully")
    if poison.status != "failure":
        raise RuntimeError(f"Poison payload should fail as data, got status {poison.status!r}")
    logger.info("SMOKE PASS")
    return SmokeReport(
        documents=len(documents),
        successes=len(successes),
        failures=len(failures),
        total_characters=sum(len(document.text) for document in successes),
        p50_seconds=seconds[len(seconds) // 2],
        max_seconds=seconds[-1],
        poison_status=poison.status,
        backends=backends,
    )


def smoke_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/converter_pool_smoke",
        deps=[source, classification, layout_model],
        # 2: the fleet became arch-adaptive (INT8 on x86, FP32 torch elsewhere, TableFormer on).
        hash_attrs={"documents": _DOCUMENTS, "attempt": 2},
        fn=remote(
            partial(
                smoke,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
                layout_model_output_path=layout_model.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    plan = plan_step()
    fetch = fetch_step(plan)
    classify = classify_step(fetch, model_step())
    layout_model = layout_model_step(fetch)
    StepRunner().run([layout_model, smoke_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
