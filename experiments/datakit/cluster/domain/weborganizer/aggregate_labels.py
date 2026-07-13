# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source label histograms + corpus rollup for the weborganizer fan-out.

Reads each source's classify output (column-projected to ``attributes``
only -- skips the ``id`` column entirely), aggregates ``top_label`` into a
24-class histogram per source, and emits a corpus rollup. Threaded across
sources on the coordinator; no Zephyr fan-out since the per-source IO is
tens of MiB after column projection.

Outputs at ``<output_path>/``:

    per_source/<safe_source_name>.json     # SourceLabelCounts per source
    top_labels.tsv                          # corpus rollup, score-sorted
    artifact.json                           # WeborgLabelRollup index

Submit::

    uv run iris --cluster=marin job run --no-wait --cpu=8 --memory=16G \\
        --extra=cpu --region europe-west4 \\
        --job-name "weborg-topic-aggregate-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.domain.weborganizer.aggregate_labels
"""

import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pyarrow.parquet as pq
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.weborganizer.all_sources_topic import WeborgTopicOutput, build_classify_steps

logger = logging.getLogger(__name__)

_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit/weborganizer"
_AGGREGATE_PARALLELISM = 16


class SourceLabelCounts(BaseModel):
    """Per-source ``top_label`` histogram. Written as ``per_source/<name>.json``."""

    source_name: str
    n_docs: int
    counts: dict[str, int]


class WeborgLabelRollup(BaseModel):
    """Step artifact: corpus rollup index. Written as ``artifact.json``.

    Attributes:
        output_dir: Step root.
        n_sources: Number of sources aggregated.
        n_docs_total: Sum of ``n_docs`` across sources.
        corpus_counts: ``top_label -> total docs across all sources``,
            score-sorted on write.
        per_source_paths: ``source_name -> GCS path to per-source JSON``.
    """

    version: str = "v1"
    output_dir: str
    n_sources: int
    n_docs_total: int
    corpus_counts: dict[str, int]
    per_source_paths: dict[str, str]


def _count_labels_one_source(*, source_name: str, attrs: WeborgTopicOutput) -> SourceLabelCounts:
    """Scan one source's classify parquets, return ``{label: count}``."""
    counts: Counter[str] = Counter()
    n_docs = 0
    paths = sorted((p for p in StoragePath(attrs.output_dir).ls() if str(p).endswith(".parquet")), key=str)
    for path in paths:
        with path.open("rb") as f:
            # Column projection: skip ``id`` (largest col); only ``topic.top_label``
            # feeds the histogram.
            table = pq.read_table(f, columns=["topic"])
        top_labels = table["topic"].combine_chunks().field("top_label").to_pylist()
        n_docs += len(top_labels)
        counts.update(top_labels)
    logger.info("aggregated %s: %d docs across %d shard(s)", source_name, n_docs, len(paths))
    return SourceLabelCounts(source_name=source_name, n_docs=n_docs, counts=dict(counts))


def _safe_filename(source_name: str) -> str:
    """Map ``cp/peps`` / ``finepdfs/fra_Latn`` to a flat filename."""
    return source_name.replace("/", "__")


def aggregate_label_counts(
    *,
    output_path: str,
    sources: dict[str, WeborgTopicOutput],
) -> WeborgLabelRollup:
    """Aggregate per-source label counts and write the rollup artifacts."""
    # Per-source counts, threaded
    per_source: dict[str, SourceLabelCounts] = {}
    with ThreadPoolExecutor(max_workers=_AGGREGATE_PARALLELISM) as ex:
        futures = {ex.submit(_count_labels_one_source, source_name=n, attrs=a): n for n, a in sources.items()}
        for fut in as_completed(futures):
            r = fut.result()
            per_source[r.source_name] = r

    # Per-source JSONs
    per_source_dir = f"{output_path.rstrip('/')}/per_source"
    per_source_paths: dict[str, str] = {}
    for name, counts_obj in per_source.items():
        json_path = f"{per_source_dir}/{_safe_filename(name)}.json"
        StoragePath(json_path).write_text(counts_obj.model_dump_json(indent=2))
        per_source_paths[name] = json_path

    # Corpus rollup (sum across sources)
    corpus: Counter[str] = Counter()
    n_docs_total = 0
    for c in per_source.values():
        corpus.update(c.counts)
        n_docs_total += c.n_docs

    # TSV rollup, score-sorted
    tsv_path = f"{output_path.rstrip('/')}/top_labels.tsv"
    with StoragePath(tsv_path).open("w") as f:
        f.write("label\tdocs\tfraction\n")
        for label, n in corpus.most_common():
            frac = n / n_docs_total if n_docs_total else 0.0
            f.write(f"{label}\t{n}\t{frac:.6f}\n")

    logger.info("rollup: %d sources, %d docs total, %d distinct labels", len(per_source), n_docs_total, len(corpus))
    return WeborgLabelRollup(
        output_dir=output_path,
        n_sources=len(per_source),
        n_docs_total=n_docs_total,
        corpus_counts=dict(corpus.most_common()),
        per_source_paths=per_source_paths,
    )


def aggregate_label_counts_step(*, classify_steps: list[StepSpec]) -> StepSpec:
    """StepSpec factory: depends on the per-source classify steps."""
    source_steps = [s for s in classify_steps if s.name.startswith("topic/")]
    source_paths = {s.name.removeprefix("topic/"): s.output_path for s in source_steps}
    hash_attrs: dict[str, Any] = {"sources": sorted(source_paths.keys()), "v": 1}
    return StepSpec(
        name="aggregate_label_counts",
        output_path_prefix=_OUTPUT_PREFIX,
        deps=source_steps,
        hash_attrs=hash_attrs,
        fn=lambda output_path: aggregate_label_counts(
            output_path=output_path,
            sources={n: read_artifact(p, WeborgTopicOutput) for n, p in source_paths.items()},
        ),
    )


def _build_steps() -> list[StepSpec]:
    classify_steps = build_classify_steps()
    rollup_step = aggregate_label_counts_step(classify_steps=classify_steps)
    return [*classify_steps, rollup_step]


if __name__ == "__main__":
    os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")
    configure_logging(logging.INFO)
    StepRunner().run(_build_steps())
