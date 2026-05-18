# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the LLM-quality classifier across every Datakit source.

Mirrors :mod:`experiments.datakit.cluster.dolma3_quality.all_sources_quality`
but points the classify fan-out at our locally-trained ``model.bin``
(produced by :mod:`experiments.datakit.cluster.llm_quality.train`) instead
of the AllenAI HuggingFace model. Output schema is the binary
``{high_score: float64}`` attribute -- ``P(label == "1") = P(quality >=
threshold)`` -- co-partitioned with each source's normalized parquet.

DAG shape::

    GCS .bin (trained by llm_quality/train.py)
        │
        ▼ _register_model_step  (no-op, just emits the FastTextModel artifact)
        │
        ▼ one classify_fasttext_step per Datakit source
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
        -- python -m experiments.datakit.cluster.llm_quality.all_sources_quality_llm \\
              --model-bin gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin \\
              --inference-name sonnet46-thr05
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from fray import ResourceConfig
from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.fasttext import FastTextModel, classify_fasttext_step

logger = logging.getLogger(__name__)


# Permanent output prefix for the LLM-quality classifier family. Sibling
# of dolma3-quality, weborganizer, etc. under gs://marin-eu-west4/datakit/.
_OUTPUT_PREFIX = "gs://marin-eu-west4/datakit/llm-quality-classifier/inference"

# Inference knobs mirror dolma3-quality (binary classifier, P(label="1")
# collapsed to attributes.high_score) so consumers can swap classifiers
# without changing downstream consolidate code.
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

# Same set excluded by the dolma3-quality and weborganizer fan-outs.
_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "safety_pt/",
    "climblab-ja",
)


def _register_model_step(name: str, model_bin_path: str, output_path_prefix: str) -> StepSpec:
    """Tiny StepSpec that just emits a FastTextModel artifact pointing at *model_bin_path*.

    The existing :func:`classify_fasttext_step` consumes its ``model_step``
    via ``Artifact.from_path(model_step, FastTextModel)``, so we need a
    step whose ``.artifact`` is a :class:`FastTextModel`. The fn here
    doesn't stage anything -- the bin is already on GCS at the path
    produced by :mod:`llm_quality.train`. We just record provenance.
    """

    def _fn(_output_path: str) -> FastTextModel:
        fs, resolved = url_to_fs(model_bin_path)
        size = int(fs.info(resolved).get("size", 0) or 0)
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


def build_classify_steps(*, model_bin_path: str, inference_name: str) -> list[StepSpec]:
    output_prefix = f"{_OUTPUT_PREFIX}/{inference_name}"
    model_step = _register_model_step(
        name=f"_model/{inference_name}",
        model_bin_path=model_bin_path,
        output_path_prefix=output_prefix,
    )

    steps: list[StepSpec] = [model_step]
    for name, src in all_sources().items():
        if any(name == p or name.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        steps.append(
            classify_fasttext_step(
                name=f"quality-llm/{name}",
                normalized=src.normalized,
                model_step=model_step,
                max_text_chars=MAX_TEXT_CHARS,
                k=K,
                threshold=THRESHOLD,
                score_target_label=SCORE_TARGET_LABEL,
                worker_resources=WORKER_RESOURCES,
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
