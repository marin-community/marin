# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scaling Ladder Analysis for Deduplicated of fineweb-edu.

This experiment runs scaling ladder analysis on the isoflop training sweeps
on both vanilla fineweb-edu and deduplicated fineweb-edu mixtures.
"""

import logging
import humanfriendly
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.dedup_commons import DedupMode, DedupConfig, deduplicate
from marin.processing.tokenize import tokenize
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig
from levanter.data.text import LMMixtureDatasetConfig
from experiments.pretraining_datasets.simple import downloads
from experiments.isoflop_sweep import (
    IsoFlopAnalysisConfig,
    MARIN_2025_RECIPE,
    create_isoflop_sweep_steps,
    run_isoflop_analysis_step,
)
import os

from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner

logger = logging.getLogger(__name__)


# --- Configuration ---
BUDGETS: list[float] = [1e18, 3e18, 6e18]
EXPERIMENT_NAME_PREFIX = "fineweb-edu-scaling-laws"
TOKENIZER = "stanford-crfm/marin-tokenizer"


def _get_vanilla_data_mixture(*, variant: str) -> LMMixtureDatasetConfig:
    """Vanilla fineweb-edu mixture"""
    tokenize_step = StepSpec(
        name=f"tokenized/{variant}",
        hash_attrs={
            "train_paths": [downloads[variant]],
            "validation_paths": [],
            "tokenizer": TOKENIZER,
        },
        fn=lambda output_path, _v=variant: tokenize(
            TokenizeConfig(
                train_paths=[downloads[_v]],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=TOKENIZER,
                window_size_bytes=humanfriendly.parse_size("512MB", binary=True),
            )
        ),
    )

    vanilla_fineweb_edu_mixture_config = lm_mixture_data_config(
        components={variant: tokenize_step},
        weights={variant: 1.0},
    )
    return vanilla_fineweb_edu_mixture_config


def _get_deduped_data_mixture(*, variant: str, mode: DedupMode, max_parallelism: int = 1024) -> LMMixtureDatasetConfig:
    """Dedup fineweb-edu mixture"""
    dedup_step = StepSpec(
        name=f"dedup/{variant}_{mode.lower()}",
        hash_attrs={
            "input_paths": downloads[variant],
            "mode": mode,
            "processes": max_parallelism,
        },
        fn=lambda output_path, _v=variant, _m=mode: deduplicate(
            DedupConfig(
                input_paths=downloads[_v],
                mode=_m,
                processes=max_parallelism,
            )
        ),
    )

    dedup_mode_to_filter_type = {
        DedupMode.EXACT_PARAGRAPH: FilterType.REMOVE_SPANS,
        DedupMode.EXACT_DOCUMENT: FilterType.REMOVE_DOC,
        DedupMode.FUZZY_DOCUMENT: FilterType.REMOVE_DOC,
    }

    consolidate_step = StepSpec(
        name=f"clean/{variant}_{mode.lower()}",
        hash_attrs={
            "input_path": downloads[variant],
            "filetype": "parquet",
            "mode": str(mode),
        },
        deps=[dedup_step],
        fn=lambda output_path, _v=variant, _m=mode, _ds=dedup_step: consolidate(
            ConsolidateConfig(
                input_path=downloads[_v],
                output_path=output_path,
                filetype="parquet",
                filters=[
                    FilterConfig(
                        type=dedup_mode_to_filter_type[_m],
                        attribute_path=os.path.join(_ds.output_path, "data"),
                        name=str(_m),
                    ),
                ],
            )
        ),
    )

    tokenize_step = StepSpec(
        name=f"tokenized/dedup_{variant}_{mode.lower()}",
        hash_attrs={
            "train_paths": [consolidate_step.output_path],
            "validation_paths": [],
            "tokenizer": TOKENIZER,
        },
        deps=[consolidate_step],
        fn=lambda output_path, _cs=consolidate_step: tokenize(
            TokenizeConfig(
                train_paths=[_cs.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=TOKENIZER,
                window_size_bytes=humanfriendly.parse_size("512MB", binary=True),
            )
        ),
    )

    dedup_name = f"dedup_{variant}_{mode.lower()}"
    deduped_fineweb_edu_mixture_config = lm_mixture_data_config(
        components={dedup_name: tokenize_step},
        weights={dedup_name: 1.0},
    )
    return deduped_fineweb_edu_mixture_config


fineweb_edu_variant = "fineweb_edu_sample_10bt"
training_steps, _ = create_isoflop_sweep_steps(
    tokenized=_get_vanilla_data_mixture(variant=fineweb_edu_variant),
    experiment_name=f"{EXPERIMENT_NAME_PREFIX}-vanilla",
    recipe=MARIN_2025_RECIPE,
    budgets=BUDGETS,
)
for mode in DedupMode:
    training_steps.extend(
        create_isoflop_sweep_steps(
            tokenized=_get_deduped_data_mixture(variant=fineweb_edu_variant, mode=mode),
            experiment_name=f"{EXPERIMENT_NAME_PREFIX}-dedup-{mode.lower()}",
            recipe=MARIN_2025_RECIPE,
            budgets=BUDGETS,
        )[0]
    )


analysis_step = StepSpec(
    name=f"{EXPERIMENT_NAME_PREFIX}-analysis",
    hash_attrs={
        "training_runs": [r.output_path for r in training_steps],
    },
    deps=training_steps,
    fn=lambda output_path: run_isoflop_analysis_step(
        IsoFlopAnalysisConfig(
            training_runs=[r.output_path for r in training_steps],
            output_path=output_path,
            recipe=MARIN_2025_RECIPE,
        )
    ),
)

all_steps = [analysis_step]


if __name__ == "__main__":
    StepRunner().run(all_steps)
