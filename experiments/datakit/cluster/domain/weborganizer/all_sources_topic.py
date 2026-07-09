# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Topic classification across every Datakit source.

Applies AllenAI's Dolma3 fasttext WebOrganizer topic classifier
(``allenai/dolma3-fasttext-weborganizer-topic-classifier``) to every normalized
source in :func:`marin.datakit.sources.all_sources`, producing one
co-partitioned Parquet attributes dataset per source. Output records are
``{id, top_label, top_score, labels, scores}`` (full label distribution, since
downstream consolidate applies its own topic thresholds).

DAG shape::

    HF: allenai/dolma3-fasttext-weborganizer-topic-classifier@<revision>
        │
        ▼ prepare_fasttext_model_step  (_model/dolma3-weborg-topic_<hash>/)
        │     stages model.bin (~4 GiB) to GCS once
        │
        ▼ one classify_weborg_topic_step per Datakit source
              (topic/<source>_<hash>/)
              workers read the .bin from in-region GCS, scan the source's
              normalized parquet, and emit co-partitioned attributes.

The model prep step downloads ~4 GiB once. The fan-out to ~100 sources is what
consumes the bulk of the wall-clock time; each source's classify step is
shard-parallel by ``NormalizedData.num_partitions``.

Output rooted at ``gs://marin-eu-west4/datakit/weborganizer/`` -- permanent
(no TTL), alongside the v0 clustering artifacts under ``gs://marin-eu-west4/datakit/``.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``):

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu --priority production --region europe-west4 \\
        --job-name "weborg-topic-all-sources-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.domain.weborganizer.all_sources_topic
"""

import logging
from typing import Any

from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file
from zephyr.runners import InlineRunner

from experiments.datakit.fasttext import (
    DEFAULT_BATCH_SIZE,
    FastTextModel,
    get_fasttext_batch_predict,
    prepare_fasttext_model_step,
)

logger = logging.getLogger(__name__)


# Model identity. The revision is pinned to a commit hash so the prep step's
# cache key is tied to bytes — HuggingFace allows silent re-uploads on a
# moving tag like ``main``, which would invisibly change downstream results.
# To bump the model, fetch a new hash via
# ``curl -s https://huggingface.co/api/models/<repo>`` and replace below.
MODEL_HF_REPO = "allenai/dolma3-fasttext-weborganizer-topic-classifier"
MODEL_HF_FILENAME = "model.bin"
MODEL_REVISION = "005a0da7d35651eb6f54553171f146bd62c5cdd2"

# Classifier inference knobs. ``K = -1`` keeps the full label distribution so
# downstream consolidate can apply different topic thresholds without
# re-classifying. ``MAX_TEXT_CHARS = 100 KiB`` caps the predict-call cost on
# pathologically long documents (matches the Dolma3 upstream pipeline). All
# three feed ``hash_attrs`` so changing them invalidates the cache.
K = -1
THRESHOLD = 0.0
MAX_TEXT_CHARS = 100_000

# Resourcing. fasttext is CPU-only and single-threaded per predict call.
# The Dolma3 weborganizer model.bin is ~4 GiB on disk and ~6-8 GiB resident
# after fasttext.load_model, so 8 GiB workers OOM (observed end-to-end in the
# weborg-topic-smoke-20260517-184843 run). 16 GiB gives enough headroom for
# the model plus the per-shard parquet writer buffer on the larger sources
# (cp/stackv2_code, finepdfs, nemotron medium_high_quality_synthetic).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Zephyr's default max_workers (None) keeps us at the framework default.
PER_SOURCE_MAX_WORKERS: int | None = None

# Permanent output prefix -- sibling of the v0 clustering artifacts under
# ``gs://marin-eu-west4/datakit/``. No TTL, no churn between runs.
_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit/weborganizer"


class WeborgTopicOutput(BaseModel):
    """Step artifact for one source's weborganizer topic classification.

    Output parquets live at ``<output_dir>/data-NNNNN-of-MMMMM.parquet``, each
    row ``{id, top_label, top_score, labels, scores}`` -- the full per-label
    distribution; downstream aggregate_labels picks ``top_label``.
    """

    version: str = "v1"
    output_dir: str
    model_path: str
    counters: dict[str, int | float]


def _run_one_source(
    *,
    normalized: NormalizedData,
    model_path: str,
    output_path: str,
    source_name: str,
    worker_resources: ResourceConfig,
    max_workers: int | None,
) -> WeborgTopicOutput:
    files = sorted(str(m) for m in StoragePath(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet").glob())
    if not files:
        raise FileNotFoundError(f"{source_name}: no .parquet files under {normalized.main_output_dir}")
    output_pattern = f"{output_path.rstrip('/')}/data-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_file)
        .window(DEFAULT_BATCH_SIZE)
        .flat_map(
            get_fasttext_batch_predict(
                model_path=model_path,
                max_text_chars=MAX_TEXT_CHARS,
                k=K,
                threshold=THRESHOLD,
                output_field_name="topic",
            )
        )
        .select("id", "topic")
        .write_parquet(output_pattern, skip_existing=True)
    )
    # InlineRunner: required so the per-process @cache on _load_fasttext_model
    # survives across shards in the same worker.
    ctx_kwargs: dict[str, Any] = {
        "name": f"weborg-topic-{source_name}",
        "resources": worker_resources,
        "stage_runner_factory": InlineRunner,
    }
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(pipeline)
    return WeborgTopicOutput(
        output_dir=output_path,
        model_path=model_path,
        counters=dict(outcome.counters),
    )


def classify_weborg_topic_step(
    *,
    name: str,
    normalized: StepSpec,
    model_step: StepSpec,
    output_path_prefix: str,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> StepSpec:
    """StepSpec factory for one source's weborganizer-topic classification.

    ``worker_resources`` and ``max_workers`` are execution policy -- not in
    ``hash_attrs`` -- so changing them does not invalidate already-classified
    sources.
    """
    hash_attrs: dict[str, Any] = {
        "max_text_chars": MAX_TEXT_CHARS,
        "k": K,
        "threshold": THRESHOLD,
    }
    resources = worker_resources or WORKER_RESOURCES
    workers = max_workers if max_workers is not None else PER_SOURCE_MAX_WORKERS
    source_name = name.removeprefix("topic/")
    return StepSpec(
        name=name,
        fn=lambda output_path: _run_one_source(
            normalized=read_artifact(normalized.output_path, NormalizedData),
            model_path=read_artifact(model_step.output_path, FastTextModel).model_path,
            output_path=output_path,
            source_name=source_name,
            worker_resources=resources,
            max_workers=workers,
        ),
        deps=[normalized, model_step],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
    )


def build_classify_steps() -> list[StepSpec]:
    # Model prep: stage the .bin to GCS exactly once. The artifact's path is
    # then a dep on every per-source classify step, so all 100+ workers read
    # the same in-region GCS object.
    model_step = prepare_fasttext_model_step(
        name="_model/dolma3-weborg-topic",
        hf_repo_id=MODEL_HF_REPO,
        hf_filename=MODEL_HF_FILENAME,
        revision=MODEL_REVISION,
        output_path_prefix=_OUTPUT_PREFIX,
    )

    classify_steps: list[StepSpec] = []
    for name, src in all_sources().items():
        classify_steps.append(
            classify_weborg_topic_step(
                name=f"topic/{name}",
                normalized=src.normalized,
                model_step=model_step,
                output_path_prefix=_OUTPUT_PREFIX,
            )
        )
    return [model_step, *classify_steps]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_classify_steps())
