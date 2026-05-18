# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the Dolma3 fasttext quality classifier on one source.

Same code paths as :mod:`all_sources_quality` (prep model -> classify per
source), but pinned to a single, small source so the whole thing finishes in
well under an hour. Verifies:

- HF model fetch + GCS staging via :func:`prepare_fasttext_model_step`
- Co-partitioned attributes parquet writes against datakit-normalized input
- The datakit ``{id, partition_id, attributes}`` schema downstream consolidate consumes
- That the staged ``.bin`` is reachable from a Zephyr classify worker in eu-west4

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G \\
        --extra=cpu \\
        --job-name "dolma3-quality-smoke-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.cluster.dolma3_quality.exp_smoke
"""

import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray import ResourceConfig  # noqa: E402
from marin.datakit.sources import all_sources  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402
from rigging.filesystem import marin_temp_bucket  # noqa: E402

from experiments.datakit.fasttext import classify_fasttext_step, prepare_fasttext_model_step  # noqa: E402

logger = logging.getLogger(__name__)

# nsf_awards (~170M tokens) — small enough that fasttext classify finishes in
# minutes, but big enough that the resulting quality-score distribution is
# informative.
SOURCE_NAME = "nsf_awards"

# Same model + revision as all_sources_quality.py. Smoke and prod share a
# cache slot, so once the .bin is staged by either path the other reuses it.
MODEL_HF_REPO = "allenai/dolma3-fasttext-quality-classifier"
MODEL_HF_FILENAME = "model.bin"
MODEL_REVISION = "bb89085994fef638ca8dc2ca25169db328e314bb"

K = -1
THRESHOLD = 0.0
MAX_TEXT_CHARS = 100_000
# Binary classifier: collapse output to a single ``attributes.high_score``
# = ``P(label == "1")``. See all_sources_quality.py for rationale.
SCORE_TARGET_LABEL = "1"

# Dolma3 quality model.bin is ~4 GiB on disk and ~6-8 GiB resident after
# fasttext.load_model. 16 GiB worker RAM gives enough headroom for the model
# plus per-shard parquet I/O buffers; 8 GiB OOMs (observed on the sibling
# weborganizer-topic smoke run 20260517-184843).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", regions=[DATA_REGION])

# Pin to eu-west4 explicitly via source_prefix so the output doesn't drift to a
# different region's temp bucket if the driver's MARIN_PREFIX is set elsewhere.
# Matches the convention in experiments.datakit.cluster.v0.exp_smoke.
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

    classify_step = classify_fasttext_step(
        name=f"datakit/classify/quality/{SOURCE_NAME}",
        normalized=source.normalized,
        model_step=model_step,
        max_text_chars=MAX_TEXT_CHARS,
        k=K,
        threshold=THRESHOLD,
        score_target_label=SCORE_TARGET_LABEL,
        worker_resources=WORKER_RESOURCES,
        output_path_prefix=_OUTPUT_PREFIX,
    )

    return [*source.normalize_steps, model_step, classify_step]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    StepRunner().run(_build_steps())
