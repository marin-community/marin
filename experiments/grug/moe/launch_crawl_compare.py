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

A separate larger rung, d1536 @ 1e20, runs on v5p-16 with bigger ~30B-token pools
(the compute-optimal 1e20 point needs more tokens than the d512-d1280 pools hold).

Selected by the ``STAGE`` env var:

    STAGE=all                       # ingest + tokenize + all 8 baseline rungs, one DAG
    STAGE=data                      # ingest + tokenize both corpora only
    STAGE=data_big                  # ingest + tokenize the larger pools for the 1e20 rung
    STAGE=train CRAWL=focus SCALE=d512    # one baseline rung on one corpus
    STAGE=train CRAWL=focus SCALE=d1536   # the 1e20 rung (v5p-16)

Submit with the TPU type and let the region be inferred — do NOT pass
``-e MARIN_PREFIX`` or ``--region``. Pinning the prefix to a non-TPU region puts
the data/checkpoints across regions from the v5p, which trips rigging's
cross-region transfer-budget guard mid-run. ``STAGE=all`` runs the whole graph in
one executor invocation so the inference co-locates ingest, tokenize, and the
eight trainings in the same TPU-capable region.
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


def _tokenize(name: str, ingest: ExecutorStep, *, allow_test_in_train: bool = False) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=ingest,
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
        allow_test_in_train=allow_test_in_train,
    )


TOKENIZE_STEPS: dict[str, ExecutorStep] = {
    # The focus source bucket is named ``cc-open-athena-test``; the ``test`` substring
    # is in the source path, not a held-out split, so the tokenize guard is waived.
    "focus": _tokenize("cc_wet/focus_supplemental_2026_22", _ingest_focus(), allow_test_in_train=True),
    "main": _tokenize("cc_wet/main_2024_18", _ingest_main()),
}

# Bigger pools for the 1e20 rung. d1536 @ 1e20 is compute-optimal at ~26B tokens,
# which exceeds the d512-d1280 pools (focus ~23B, main ~16B), so it gets its own
# freshly-sampled larger corpora (~30B tokens each).
_FOCUS_BIG_FILES = 2200  # ~30B tokens (~48% of the 4,573-file focus crawl)
_MAIN_BIG_FILES = 460  # ~32B tokens


def _ingest_focus_big() -> ExecutorStep:
    return download_cc_wet_step(
        name="raw/cc_wet/focus_supplemental_2026_22_big",
        crawl="CC-SUPPLEMENTAL-2026-22",
        wet_paths_fn=lambda: focus_crawl_wet_paths(_FOCUS_INDEX_PARQUET),
        seed=_SAMPLE_SEED,
        num_files=_FOCUS_BIG_FILES,
    ).as_executor_step()


def _ingest_main_big() -> ExecutorStep:
    return download_cc_wet_step(
        name="raw/cc_wet/main_2024_18_big",
        crawl=_MAIN_CRAWL,
        wet_paths_fn=lambda: main_crawl_wet_paths(_MAIN_CRAWL),
        seed=_SAMPLE_SEED,
        num_files=_MAIN_BIG_FILES,
    ).as_executor_step()


TOKENIZE_BIG: dict[str, ExecutorStep] = {
    "focus": _tokenize("cc_wet/focus_supplemental_2026_22_big", _ingest_focus_big(), allow_test_in_train=True),
    "main": _tokenize("cc_wet/main_2024_18_big", _ingest_main_big()),
}


def _data_for(crawl: str, big: bool) -> LmDataConfig:
    """Single-source training config (+ paloma/uncheatable validation sets)."""
    return lm_data_config(
        training_set=(TOKENIZE_BIG if big else TOKENIZE_STEPS)[crawl],
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

# Larger 1e20 rung — compute-optimal d1536. Runs on v5p-16 (8 chips, FSDP) with the
# bigger pools, and is launched separately from the 8-run ladder above.
BIG_LADDER: dict[str, tuple[int, float]] = {"d1536": (1536, 1e20)}


def _rung(scale: str) -> tuple[tuple[int, float], bool]:
    if scale in LADDER:
        return LADDER[scale], False
    if scale in BIG_LADDER:
        return BIG_LADDER[scale], True
    raise ValueError(f"SCALE={scale!r} must be one of {sorted(LADDER) + sorted(BIG_LADDER)}")


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
    (hidden_dim, budget), big = _rung(scale)
    model, optimizer, batch_size, steps = build_from_heuristic(budget=budget, hidden_dim=hidden_dim)
    run_id = f"crawlcmp-{crawl}-{scale}-{budget:.2e}"
    # Big rung: v5p-16 (8 chips) with FSDP (replica_axis_size=1) for the larger model.
    tpu = "v5p-16" if big else "v5p-8"
    trainer = GrugTrainerConfig(
        z_loss_weight=1e-4, ema_beta=None, log_every=1, **({"replica_axis_size": 1} if big else {})
    )

    return ExecutorStep(
        name=f"experiments/grug-moe-crawl-compare/{crawl}/{scale}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=_data_for(crawl, big),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(tpu)),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=_tracker(crawl, scale),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(trainer),
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


def all_steps() -> list[ExecutorStep]:
    """Both tokenize steps plus all 8 ladder rungs, as one DAG.

    Running the whole graph in a single executor invocation lets the region
    inference co-locate ingest, tokenize, and training in the same TPU-capable
    region — so nothing crosses regions. Do not pin MARIN_PREFIX or TPU regions.
    """
    return [
        *TOKENIZE_STEPS.values(),
        *(build_train_step(crawl, scale) for crawl in TOKENIZE_STEPS for scale in LADDER),
    ]


def big_steps() -> list[ExecutorStep]:
    """Big-pool tokenize + both d1536 @ 1e20 rungs, as one co-located DAG."""
    return [
        *TOKENIZE_BIG.values(),
        *(build_train_step(crawl, scale) for crawl in TOKENIZE_BIG for scale in BIG_LADDER),
    ]


def main() -> None:
    stage = os.environ.get("STAGE", "train")
    if stage == "all":
        executor_main(
            steps=all_steps(),
            description="grug-MoE crawl-compare: ingest+tokenize+train full focus-vs-main ladder.",
        )
        return
    if stage == "data":
        executor_main(
            steps=list(TOKENIZE_STEPS.values()),
            description="Ingest + tokenize focus (CC-SUPPLEMENTAL-2026-22) and main (CC-MAIN-2024-18) WET.",
        )
        return
    if stage == "data_big":
        executor_main(
            steps=list(TOKENIZE_BIG.values()),
            description="Ingest + tokenize the larger ~30B-token focus/main pools for the 1e20 (d1536) rung.",
        )
        return
    if stage == "big":
        executor_main(
            steps=big_steps(),
            description="grug-MoE crawl-compare: big-pool ingest+tokenize + d1536 @ 1e20 on v5p-16, both corpora.",
        )
        return
    if stage != "train":
        raise ValueError(f"STAGE={stage!r} must be 'all', 'data', 'data_big', 'big', or 'train'")

    crawl = os.environ["CRAWL"]
    scale = os.environ["SCALE"]
    executor_main(
        steps=[build_train_step(crawl, scale)],
        description=f"grug-MoE crawl-compare: {crawl} / {scale}.",
    )


if __name__ == "__main__":
    main()
