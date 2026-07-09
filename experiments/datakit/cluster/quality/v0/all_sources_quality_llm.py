# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the LLM-quality classifier across every Datakit source.

Mirrors :mod:`experiments.datakit.cluster.quality.dolma3_quality.all_sources_quality`
but points the classify fan-out at our locally-trained ``model.bin``
(produced by :mod:`experiments.datakit.cluster.quality.v0.train`) instead
of the AllenAI HuggingFace model. Output records are ``{id, score}``
where ``score = P(label == "1") = P(quality >= threshold)``,
co-partitioned with each source's normalized parquet.

DAG shape::

    GCS .bin (trained by llm_quality/train.py)
        │
        ▼ _register_model_step  (no-op, just emits the FastTextModel artifact)
        │
        ▼ one classify_llm_quality_step per Datakit source
              (quality-llm/<source>_<hash>/)
              workers stream the .bin from in-region GCS, scan the
              normalized parquet, emit co-partitioned attributes.

Same exclude prefixes as the dolma3-quality fan-out (``safety_pt/`` and
``climblab-ja``). Output is permanent (no TTL) under
``gs://marin-eu-west4/datakit/llm-quality-classifier/inference/<name>/``,
where ``<name>`` distinguishes models trained from different rubrics or
thresholds (defaults to ``sonnet46-thr05``).

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu --priority production --region europe-west4 \\
        --job-name "llm-quality-allsources-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.quality.v0.all_sources_quality_llm \\
              --model-bin gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin \\
              --inference-name sonnet46-thr05
"""

import argparse
import logging
import os
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
)

logger = logging.getLogger(__name__)


# Permanent output prefix for the LLM-quality classifier family. Sibling
# of dolma3-quality, weborganizer, etc. under gs://marin-eu-west4/datakit/.
_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit/llm-quality-classifier/inference"

# Inference knobs mirror dolma3-quality (binary classifier, P(label="1")
# collapsed to ``score``) so consumers can swap classifiers without
# changing downstream consolidate code.
K = -1
THRESHOLD = 0.0
MAX_TEXT_CHARS = 100_000
SCORE_TARGET_LABEL = "1"

# Same worker shape as dolma3-quality. Our model.bin is tiny relative to
# dolma3's 4 GiB classifier (typical fasttext-from-scratch with dim=100
# and ~6K vocab is ~25 MB), so 8 GiB workers would be plenty -- but 16
# GiB keeps the same parquet-writer buffer headroom on large sources
# (cp/stackv2_code, finepdfs, nemotron medium_high_quality_synthetic).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Zephyr default for workers (None = framework default).
PER_SOURCE_MAX_WORKERS: int | None = None


class LlmQualityOutput(BaseModel):
    """Step artifact for one source's LLM-quality classification.

    Output parquets live at ``<output_dir>/data-NNNNN-of-MMMMM.parquet``, each
    row ``{id, score: float64}`` where ``score = P(label == "1")``.
    """

    version: str = "v1"
    output_dir: str
    model_path: str
    counters: dict[str, int | float]


def _register_model_step(name: str, model_bin_path: str, output_path_prefix: str | None = None) -> StepSpec:
    """Tiny StepSpec that just emits a FastTextModel artifact pointing at *model_bin_path*.

    :func:`classify_llm_quality_step` consumes its ``model_step`` via
    ``read_artifact(model_step.output_path, FastTextModel)``, so we need a step
    whose ``.artifact`` is a :class:`FastTextModel`. The fn here doesn't
    stage anything -- the bin is already on GCS at the path produced by
    :mod:`llm_quality.train`. We just record provenance.
    """

    def _fn(_output_path: str) -> FastTextModel:
        size = StoragePath(model_bin_path).size()
        return FastTextModel(
            model_dir=os.path.dirname(model_bin_path),
            model_path=model_bin_path,
            hf_repo_id="local/llm-quality-classifier",
            hf_filename=os.path.basename(model_bin_path),
            revision="local",
            size_bytes=size,
        )

    hash_attrs: dict[str, Any] = {"model_path": model_bin_path}
    return StepSpec(
        name=name,
        fn=_fn,
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
    )


def _run_one_source(
    *,
    normalized: NormalizedData,
    model_path: str,
    output_path: str,
    source_name: str,
    worker_resources: ResourceConfig,
    max_workers: int | None,
) -> LlmQualityOutput:
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
                score_target_label=SCORE_TARGET_LABEL,
                output_field_name="score",
            )
        )
        .select("id", "score")
        .write_parquet(output_pattern, skip_existing=True)
    )
    # InlineRunner: required so the per-process @cache on _load_fasttext_model
    # survives across shards in the same worker.
    ctx_kwargs: dict[str, Any] = {
        # Iris rejects job names containing '/' (validator added in #5736), so flatten
        # registry names like ``cp/biodiversity`` to ``cp__biodiversity``.
        "name": f"llm-quality-{source_name.replace('/', '__')}",
        "resources": worker_resources,
        "stage_runner_factory": InlineRunner,
    }
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(pipeline)
    return LlmQualityOutput(
        output_dir=output_path,
        model_path=model_path,
        counters=dict(outcome.counters),
    )


def classify_llm_quality_step(
    *,
    name: str,
    normalized: StepSpec,
    model_step: StepSpec,
    output_path_prefix: str | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> StepSpec:
    """StepSpec factory for one source's LLM-quality classification.

    ``worker_resources`` and ``max_workers`` are execution policy -- not in
    ``hash_attrs`` -- so changing them does not invalidate already-classified
    sources.
    """
    hash_attrs: dict[str, Any] = {
        "max_text_chars": MAX_TEXT_CHARS,
        "k": K,
        "threshold": THRESHOLD,
        "score_target_label": SCORE_TARGET_LABEL,
        # Bumped after fixing the fasttext-wheel 0.9.2 batch-predict quirk
        # (commit 9beb82582). Pre-fix cache slots produced ``score >= 0.5``
        # for every record because ``model.predict(list, k=-1)`` duplicated
        # the top-label prob across both labels; the per-text predict path
        # now yields proper softmax. Force a fresh cache slot so re-runs
        # don't pick up the broken outputs.
        "predict_mode": "per_text",
    }
    resources = worker_resources or WORKER_RESOURCES
    workers = max_workers if max_workers is not None else PER_SOURCE_MAX_WORKERS
    source_name = name.removeprefix("quality-llm/")
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


def build_classify_steps(*, model_bin_path: str, inference_name: str) -> list[StepSpec]:
    output_prefix = f"{_OUTPUT_PREFIX}/{inference_name}"
    model_step = _register_model_step(
        name=f"_model/{inference_name}",
        model_bin_path=model_bin_path,
        output_path_prefix=output_prefix,
    )

    steps: list[StepSpec] = [model_step]
    for name, src in all_sources().items():
        steps.append(
            classify_llm_quality_step(
                name=f"quality-llm/{name}",
                normalized=src.normalized,
                model_step=model_step,
                output_path_prefix=output_prefix,
            )
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", required=True, help="GCS path to trained model.bin")
    parser.add_argument(
        "--inference-name",
        required=True,
        help="Subdir name under the permanent prefix (e.g. 'sonnet46-thr05')",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    StepRunner().run(build_classify_steps(model_bin_path=args.model_bin, inference_name=args.inference_name))


if __name__ == "__main__":
    main()
