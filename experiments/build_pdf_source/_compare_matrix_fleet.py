# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- the full 2x2 backend matrix, run through the converter fleet.

DELETE once the result is recorded in ``.agents/ops/2026-07-30-layout-quantization-recipe.md``.
Nothing in the pipeline imports this.

The matrix crosses layout backend (FP32 TORCH_HERON vs OpenVINO INT8) with table reader (PyMuPDF
ruling-line vs TableFormer) over the same 600-document corpus every earlier comparison used. The
missing cell is INT8 + TableFormer: whether TableFormer's clip-insensitive grid inference recovers
the tables that INT8's box jitter empties through the clip-and-decline PyMuPDF reader, which would
buy INT8's speed without its table loss.

Arms run sequentially, each behind its own converter fleet at the production operating point
(64 pods x 4 converters). The driver is the sender: it holds the corpus in memory and posts every
document itself, so there is no zephyr stage, no output-prefix ``skip_existing`` hazard, and no
per-task converter builds -- the failure modes that sank the previous one-document-per-task
harness. Per-document conversion ``seconds`` comes from inside the handler, so arm timing excludes
queueing and fleet startup; compare arms on it, never on wall clock.

Run on the same x86 cluster as the other comparisons::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name compare-matrix-fleet \\
        -- python -m experiments.build_pdf_source._compare_matrix_fleet
"""

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.inference.converter_pool import remote_converter_pool
from pydantic import BaseModel
from rigging.filesystem import prefix_join, url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.classify import routing_keys
from experiments.build_pdf_source.extract_fleet import MODEL_ID, build_pool_config, convert_document
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Matched to the earlier comparisons so the runs describe the same corpus.
SOURCE_SHARDS = 24
DOCUMENTS_PER_SHARD = 25
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]

# The baseline arm is listed first; every pairwise comparison is against it.
_ARMS: dict[str, tuple[LayoutBackend, TableBackend]] = {
    "fp32_pymupdf": (LayoutBackend.TORCH_HERON, TableBackend.PYMUPDF),
    "fp32_tableformer": (LayoutBackend.TORCH_HERON, TableBackend.DOCLING),
    "int8_pymupdf": (LayoutBackend.INT8, TableBackend.PYMUPDF),
    "int8_tableformer": (LayoutBackend.INT8, TableBackend.DOCLING),
}
_BASELINE = "fp32_pymupdf"

# The production fleet shape. 600 documents across 256 converters clears in ~3 waves, so the run
# per arm is dominated by fleet startup, not conversion.
_POOL_INSTANCES = 64
_PROCESSES_PER_INSTANCE = 4
# Sender in-flight sits at ~1.5x the converter count: enough that no converter idles between
# documents, low enough that a document never waits in the broker queue long enough to threaten
# the proxy timeout behind the 600-second tail.
_IN_FLIGHT = 384

_TABLE_BLOCK = re.compile(r"<docling_table>(.*?)</docling_table>", re.DOTALL)

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("arm", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error", pa.string(), nullable=True),
        pa.field("seconds", pa.float64(), nullable=False),
        pa.field("text", pa.string(), nullable=True),
    ]
)

_DRIVER_RESOURCES = ResourceConfig(cpu=8, ram="32g", disk="16g")


class ArmStats(BaseModel):
    documents: int
    converted: int
    failed: int
    lost: int
    total_characters: int
    tables: int
    table_characters: int
    empty_tables: int
    mean_seconds: float
    convert_wall_seconds: float


class PairwiseStats(BaseModel):
    """Against the baseline arm, over documents both arms converted."""

    documents_compared: int
    mean_char_ratio: float
    mean_token_f1: float
    shorter_than_baseline: int
    longer_than_baseline: int


class MatrixReport(BaseModel):
    baseline: str
    arms: dict[str, ArmStats]
    versus_baseline: dict[str, PairwiseStats]


def _table_stats(text: str) -> tuple[int, int, int]:
    """Return (table count, characters inside tables, tables that came back empty)."""
    blocks = _TABLE_BLOCK.findall(text)
    return len(blocks), sum(len(b) for b in blocks), sum(1 for b in blocks if not b.strip())


def _token_f1(reference: str, candidate: str) -> float:
    reference_tokens = Counter(reference.split())
    candidate_tokens = Counter(candidate.split())
    overlap = sum((reference_tokens & candidate_tokens).values())
    total = sum(reference_tokens.values()) + sum(candidate_tokens.values())
    if total == 0:
        return 1.0
    return 2 * overlap / total


def _load_corpus(source: PdfSourceData, classification: PdfClassificationData) -> list[dict]:
    keys = routing_keys(classification.main_output_dir, needs_ocr=False)
    filesystem, path = url_to_fs(source.main_output_dir)
    protocol = source.main_output_dir.split("://")[0] if "://" in source.main_output_dir else ""
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:SOURCE_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")
    rows: list[dict] = []
    for shard in shards:
        shard_url = f"{protocol}://{shard}" if protocol else shard
        with filesystem.open(shard_url, "rb") as stream:
            table = pq.read_table(stream, columns=_SOURCE_COLUMNS)
        taken = 0
        for row in table.to_pylist():
            if taken >= DOCUMENTS_PER_SHARD:
                break
            if (row["warc_filename"], row["warc_record_offset"]) not in keys:
                continue
            rows.append(row)
            taken += 1
    logger.info("Corpus: %d documents from %d shards", len(rows), len(shards))
    return rows


def _arm_options(arm: str, layout_model: LayoutModelData):
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415

    layout, table = _ARMS[arm]
    return ExtractionOptions(
        table_backend=table,
        layout_backend=layout,
        layout_model_path=layout_model.model_path if layout is LayoutBackend.INT8 else None,
        layout_label_map=layout_model.label_map if layout is LayoutBackend.INT8 else {},
    )


def _run_arm(arm: str, rows: list[dict], layout_model: LayoutModelData, output_path: str) -> ArmStats:
    """Convert the whole corpus through one arm's fleet and persist its texts."""
    pool_config = replace(
        build_pool_config(_arm_options(arm, layout_model)),
        model_id=f"{MODEL_ID}-{arm.replace('_', '-')}",
        instances=_POOL_INSTANCES,
        processes_per_instance=_PROCESSES_PER_INSTANCE,
    )
    results: list[dict] = []
    with remote_converter_pool(pool_config) as session:
        base_url = session.endpoint.base_url
        logger.info("[%s] fleet ready at %s", arm, base_url)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=_IN_FLIGHT) as pool:
            futures = {pool.submit(convert_document, base_url, _IN_FLIGHT, row): row for row in rows}
            for future, row in futures.items():
                source_id = f"{row['warc_filename']}:{row['warc_record_offset']}"
                try:
                    document = future.result()
                except Exception as error:
                    logger.warning("[%s] request lost for %s: %s", arm, row["url"], error)
                    results.append(
                        {
                            "source_id": source_id,
                            "url": row["url"],
                            "arm": arm,
                            "status": "lost",
                            "error": f"{type(error).__name__}: {error}",
                            "seconds": 0.0,
                            "text": None,
                        }
                    )
                    continue
                results.append(
                    {
                        "source_id": source_id,
                        "url": row["url"],
                        "arm": arm,
                        "status": document.status,
                        "error": document.error,
                        "seconds": document.seconds,
                        "text": document.text if document.status != "failure" else None,
                    }
                )
        wall = time.monotonic() - started
        session.check_alive()

    destination = prefix_join(output_path, f"texts/{arm}.parquet")
    filesystem, destination_path = url_to_fs(destination)
    with filesystem.open(destination_path, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(results, schema=_TEXT_SCHEMA), stream)

    converted = [result for result in results if result["text"] is not None]
    table_counts = [_table_stats(result["text"]) for result in converted]
    seconds = [result["seconds"] for result in results if result["status"] != "lost"]
    stats = ArmStats(
        documents=len(results),
        converted=len(converted),
        failed=sum(1 for result in results if result["status"] == "failure"),
        lost=sum(1 for result in results if result["status"] == "lost"),
        total_characters=sum(len(result["text"]) for result in converted),
        tables=sum(count for count, _, _ in table_counts),
        table_characters=sum(chars for _, chars, _ in table_counts),
        empty_tables=sum(empty for _, _, empty in table_counts),
        mean_seconds=sum(seconds) / len(seconds) if seconds else 0.0,
        convert_wall_seconds=wall,
    )
    logger.info("[%s] %s", arm, stats.model_dump())
    return stats


def _pairwise(baseline_texts: dict[str, str], arm_texts: dict[str, str]) -> PairwiseStats:
    shared = sorted(set(baseline_texts) & set(arm_texts))
    ratios = []
    f1s = []
    shorter = 0
    longer = 0
    for source_id in shared:
        reference = baseline_texts[source_id]
        candidate = arm_texts[source_id]
        if len(reference) > 0:
            ratios.append(len(candidate) / len(reference))
        f1s.append(_token_f1(reference, candidate))
        if len(candidate) < len(reference):
            shorter += 1
        elif len(candidate) > len(reference):
            longer += 1
    return PairwiseStats(
        documents_compared=len(shared),
        mean_char_ratio=sum(ratios) / len(ratios) if ratios else 0.0,
        mean_token_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        shorter_than_baseline=shorter,
        longer_than_baseline=longer,
    )


def matrix(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> MatrixReport:
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    rows = _load_corpus(source, classification)
    arms: dict[str, ArmStats] = {}
    texts: dict[str, dict[str, str]] = {}
    for arm in _ARMS:
        arms[arm] = _run_arm(arm, rows, layout_model, output_path)
        destination = prefix_join(output_path, f"texts/{arm}.parquet")
        filesystem, destination_path = url_to_fs(destination)
        with filesystem.open(destination_path, "rb") as stream:
            table = pq.read_table(stream, columns=["source_id", "text"])
        texts[arm] = {
            source_id: text
            for source_id, text in zip(table["source_id"].to_pylist(), table["text"].to_pylist(), strict=True)
            if text is not None
        }

    report = MatrixReport(
        baseline=_BASELINE,
        arms=arms,
        versus_baseline={arm: _pairwise(texts[_BASELINE], texts[arm]) for arm in _ARMS if arm != _BASELINE},
    )
    for arm, stats in report.arms.items():
        logger.info("ARM %s: %s", arm, stats.model_dump())
    for arm, pairwise in report.versus_baseline.items():
        logger.info("VS BASELINE %s: %s", arm, pairwise.model_dump())
    return report


def matrix_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/backend_matrix_fleet",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "source_shards": SOURCE_SHARDS,
            "documents_per_shard": DOCUMENTS_PER_SHARD,
            "arms": sorted(_ARMS),
            "attempt": 1,
        },
        fn=remote(
            partial(
                matrix,
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
    StepRunner().run([layout_model, matrix_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
