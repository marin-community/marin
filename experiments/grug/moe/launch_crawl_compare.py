# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused-crawl vs main-crawl data comparison on the grug-MoE compute-optimal ladder.

Trains the four compute-optimal baseline rungs from ``README.md`` / ``agent.md``
(d512, d768, d1024, d1280) on two token-matched raw-WET corpora:

- ``focus`` — ``CC-SUPPLEMENTAL-2026-22``, Common Crawl's quality-steered
  "top-10k science domains" focus crawl.
- ``main``  — ``CC-MAIN-2024-18``, the general main crawl used to build the
  focus crawl's seed list (and so the natural baseline).

Both corpora are a seeded random sample of WET files large enough to clear the
largest rung's token budget (d1280 = 1.24e10), so every rung sees the same
number of fresh tokens from either corpus. Ingestion reuses the Common Crawl
WET path (:mod:`marin.datakit.download.common_crawl_wet`).

Two stages, selected by the ``STAGE`` env var, so the shared data prep runs once
before the eight training jobs fan out (one v5p-8 per job):

    STAGE=data                      # ingest + tokenize both corpora
    STAGE=train CRAWL=focus SCALE=d512   # one rung on one corpus

Submit each as its own Iris job (see ``agent.md`` for the command).
"""

import os

from fray.cluster import ResourceConfig
from levanter.data.text import BlockShuffleConfig, LmDataConfig, TextLmDatasetFormat
from levanter.tracker import TrackerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.datakit.download.common_crawl_wet import (
    download_cc_wet_step,
    focus_crawl_wet_paths,
    main_crawl_wet_paths,
)
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer
from experiments.tokenization import default_tokenize

# --- corpora -----------------------------------------------------------------

# One focus-crawl index parquet part. The index is row-partitioned, so a single
# part enumerates the crawl's whole WET file universe (4,573 files).
_FOCUS_INDEX_PARQUET = (
    "https://data.commoncrawl.org/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22"
    "/index/table/cc-supplemental/warc/crawl=CC-SUPPLEMENTAL-2026-22/subset=warc"
    "/part-00000-8637f21e-a055-46d1-8233-990f59974248.c000.gz.parquet"
)
_MAIN_CRAWL = "CC-MAIN-2024-18"

# Sample enough WET files to clear the largest rung (d1280 = 1.24e10 tokens) with
# margin. Measured yield: ~13.8M llama3 tokens per focus WET file (~19.8 MB gz)
# and ~80M per main WET file (~94 MB gz).
_SAMPLE_SEED = 0
_FOCUS_NUM_FILES = 1200  # ~16.5B tokens
_MAIN_NUM_FILES = 230  # ~18.4B tokens


def _ingest_focus() -> ExecutorStep:
    return download_cc_wet_step(
        name="raw/cc_wet/focus_supplemental_2026_22",
        crawl="CC-SUPPLEMENTAL-2026-22",
        wet_paths_fn=lambda: focus_crawl_wet_paths(_FOCUS_INDEX_PARQUET),
        seed=_SAMPLE_SEED,
        num_files=_FOCUS_NUM_FILES,
    ).as_executor_step()


def _ingest_main() -> ExecutorStep:
    return download_cc_wet_step(
        name="raw/cc_wet/main_2024_18",
        crawl=_MAIN_CRAWL,
        wet_paths_fn=lambda: main_crawl_wet_paths(_MAIN_CRAWL),
        seed=_SAMPLE_SEED,
        num_files=_MAIN_NUM_FILES,
    ).as_executor_step()


def _tokenize(name: str, ingest: ExecutorStep) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=ingest,
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
    )


TOKENIZE_STEPS: dict[str, ExecutorStep] = {
    "focus": _tokenize("cc_wet/focus_supplemental_2026_22", _ingest_focus()),
    "main": _tokenize("cc_wet/main_2024_18", _ingest_main()),
}


def _crawl_data(crawl: str) -> LmDataConfig:
    """Single-source training config with paloma/uncheatable validation sets."""
    return lm_data_config(
        training_set=TOKENIZE_STEPS[crawl],
        validation_sets=default_validation_sets(tokenizer=llama3_tokenizer),
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel"),
    )


# --- ladder ------------------------------------------------------------------

# Compute-optimal baseline rungs (budget per hidden dim) from README.md.
LADDER: dict[str, tuple[int, float]] = {
    "d512": (512, 2.19e17),
    "d768": (768, 1.70e18),
    "d1024": (1024, 9.00e18),
    "d1280": (1280, 2.83e19),
}


def _tracker(crawl: str, scale: str) -> TrackerConfig:
    """W&B when ``WANDB_API_KEY`` is present, else a JSON logger fallback.

    The fallback keeps the run unblocked when no W&B key is plumbed through; the
    same ``eval/paloma/macro_loss`` scalars are written to the run's logs either
    way.
    """
    if os.environ.get("WANDB_API_KEY"):
        return WandbConfig(
            project="marin_moe",
            tags=["moe", "crawl-compare", crawl, scale],
            group="crawl-compare-focus-vs-main",
            name=None,
        )
    return JsonLoggerConfig(logger_name="grug_moe_crawl_compare.metrics")


def build_train_step(crawl: str, scale: str) -> ExecutorStep:
    if crawl not in TOKENIZE_STEPS:
        raise ValueError(f"CRAWL={crawl!r} must be one of {sorted(TOKENIZE_STEPS)}")
    if scale not in LADDER:
        raise ValueError(f"SCALE={scale!r} must be one of {sorted(LADDER)}")
    hidden_dim, budget = LADDER[scale]
    model, optimizer, batch_size, steps = build_from_heuristic(budget=budget, hidden_dim=hidden_dim)
    run_id = f"crawlcmp-{crawl}-{scale}-{budget:.2e}"

    return ExecutorStep(
        name=f"experiments/grug-moe-crawl-compare/{crawl}/{scale}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=_crawl_data(crawl),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=_tracker(crawl, scale),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


def main() -> None:
    stage = os.environ.get("STAGE", "train")
    if stage == "data":
        executor_main(
            steps=list(TOKENIZE_STEPS.values()),
            description="Ingest + tokenize focus (CC-SUPPLEMENTAL-2026-22) and main (CC-MAIN-2024-18) WET.",
        )
        return
    if stage != "train":
        raise ValueError(f"STAGE={stage!r} must be 'data' or 'train'")

    crawl = os.environ["CRAWL"]
    scale = os.environ["SCALE"]
    executor_main(
        steps=[build_train_step(crawl, scale)],
        description=f"grug-MoE crawl-compare: {crawl} / {scale}.",
    )


if __name__ == "__main__":
    main()
