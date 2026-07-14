# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the Dolma3 fasttext quality classifier on one source.

Same code paths as :mod:`all_sources_quality` (prep model -> classify per
source), but pinned to a single, small source so the whole thing finishes in
well under an hour. Verifies:

- HF model fetch + GCS staging via :func:`prepare_fasttext_model_step`
- Co-partitioned attributes parquet writes against datakit-normalized input
- The flat ``{id, score}`` output schema downstream consolidate consumes
- That the staged ``.bin`` is reachable from a Zephyr classify worker in eu-west4

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu \\
        --job-name "dolma3-quality-smoke-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.quality.dolma3_quality.ops.exp_smoke
"""

import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray.types import ResourceConfig  # noqa: E402
from marin.datakit.sources import all_sources  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402
from rigging.filesystem import marin_temp_bucket  # noqa: E402

from experiments.datakit.cluster.quality.dolma3_quality.all_sources_quality import (  # noqa: E402
    MODEL_HF_FILENAME,
    MODEL_HF_REPO,
    MODEL_REVISION,
    classify_dolma3_quality_step,
)
from experiments.datakit.fasttext import prepare_fasttext_model_step  # noqa: E402

logger = logging.getLogger(__name__)

# nsf_awards (~170M tokens) — small enough that fasttext classify finishes in
# minutes, but big enough that the resulting quality-score distribution is
# informative.
SOURCE_NAME = "nsf_awards"

# Pin worker to eu-west4 for the smoke run; production
# (all_sources_quality.py) leaves region open since it's already running with
# MARIN_PREFIX pointed at eu-west4. 16 GiB matches the production cap (8 GiB
# OOMs on the sibling weborganizer-topic smoke run 20260517-184843).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", regions=[DATA_REGION])

# Pin to eu-west4 explicitly via source_prefix so the output doesn't drift to a
# different region's temp bucket if the driver's MARIN_PREFIX is set elsewhere.
# Matches the convention in experiments.datakit.cluster.domain.v0.ops.exp_smoke.
_OUTPUT_PREFIX = marin_temp_bucket(
    ttl_days=7,
    prefix="rav/dolma3-quality-smoke",
    source_prefix="gs://marin-eu-west4",
)


def _build_steps() -> list[StepSpec]:
    sources = all_sources()
    source = sources[SOURCE_NAME]

    model_step = prepare_fasttext_model_step(
        name="datakit/classify/_model/dolma3-quality",
        hf_repo_id=MODEL_HF_REPO,
        hf_filename=MODEL_HF_FILENAME,
        revision=MODEL_REVISION,
        output_path_prefix=_OUTPUT_PREFIX,
    )

    classify_step = classify_dolma3_quality_step(
        name=f"datakit/classify/quality/{SOURCE_NAME}",
        normalized=source.normalized,
        model_step=model_step,
        output_path_prefix=_OUTPUT_PREFIX,
        worker_resources=WORKER_RESOURCES,
    )

    return [*source.normalize_steps, model_step, classify_step]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    StepRunner().run(_build_steps())
