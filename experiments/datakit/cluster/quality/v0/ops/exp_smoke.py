# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke for the LLM-quality classifier.

Exercises the full sample → score → train chain with a budget small
enough to run on a laptop: ~200 docs (≈$1 with Sonnet 4.6), a fast
training pass, and TTL'd output paths so we don't litter permanent
storage. Use this before launching the $50 production run to verify
the rubric parses, the API key works, and the model converges to a
sensible label balance.

Submit (eu-west4 worker so we read marin-eu-west4 in-region):

    uv run iris --cluster=marin job run --no-wait --cpu=2 --memory=4G \\
        --extra=cpu --region europe-west4 \\
        --job-name "llm-quality-smoke-$(date +%Y%m%d-%H%M%S)" \\
        --env-file .marin.yaml -- \\
        python -m experiments.datakit.cluster.quality.v0.ops.exp_smoke
"""

from __future__ import annotations

import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from rigging.filesystem import marin_temp_bucket  # noqa: E402
from rigging.log_setup import configure_logging  # noqa: E402

from experiments.datakit.cluster.quality.v0.sample import sample  # noqa: E402
from experiments.datakit.cluster.quality.v0.score import score  # noqa: E402
from experiments.datakit.cluster.quality.v0.train import train  # noqa: E402

logger = logging.getLogger(__name__)


# Tiny budget that still covers floor*n_active_sources + a few extras.
# floor=2 across 104 active sources -> 208 floor docs; +~92 extra ~= 300
# scored at ~$0.005/doc (Sonnet 4.6 with cached system prompt) ~= $1.50.
TOTAL_SIZE = 300
FLOOR_PER_SOURCE = 2
BUDGET_USD = 3.0  # 2x headroom over expected $1.50 so transient cost spikes don't truncate

# Sonnet, matching production. Smoke is small enough that the cost
# difference (vs Haiku) is irrelevant and we exercise the same prompt path.
ORACLE_MODEL = "claude-sonnet-4-6"

SEED = 42
# Per-stage concurrency. Sampling is sequential -- big sources have
# multi-hundred-MB row groups and concurrent buffering OOMs the 2 GB
# driver. Scoring is API-bound; 4 concurrent Anthropic calls is gentle
# on rate limits.
SAMPLE_NUM_WORKERS = 1
SCORE_NUM_WORKERS = 4
VAL_FRAC = 0.1


def main() -> None:
    configure_logging(logging.INFO)

    base = marin_temp_bucket(
        ttl_days=7,
        prefix="rav/llm-quality-smoke",
        source_prefix="gs://marin-eu-west4",
    )
    samples_path = f"{base}/samples.parquet"
    scored_path = f"{base}/scored.parquet"
    model_dir = f"{base}/model"

    logger.info("=== smoke output base: %s ===", base)

    logger.info("=== STAGE 1/3: sample ===")
    sample(
        output_path=samples_path,
        total_size=TOTAL_SIZE,
        floor_per_source=FLOOR_PER_SOURCE,
        seed=SEED,
        num_workers=SAMPLE_NUM_WORKERS,
    )

    logger.info("=== STAGE 2/3: score ===")
    score(
        input_path=samples_path,
        output_path=scored_path,
        oracle_model=ORACLE_MODEL,
        budget_usd=BUDGET_USD,
        max_workers=SCORE_NUM_WORKERS,
        max_output_tokens=200,
    )

    logger.info("=== STAGE 3/3: train ===")
    meta = train(
        input_path=scored_path,
        output_dir=model_dir,
        threshold=None,  # median split
        val_frac=VAL_FRAC,
        seed=SEED,
        max_text_chars=4_000,
        hp_overrides={"epoch": 5},
    )

    logger.info("=== SMOKE COMPLETE ===")
    logger.info("threshold=%.4f train=%d val=%d", meta.threshold, meta.n_train, meta.n_val)
    logger.info("val precision=%.4f recall=%.4f f1=%.4f", meta.val_precision, meta.val_recall, meta.val_f1)
    logger.info("model at: %s/model.bin", model_dir)


if __name__ == "__main__":
    main()
