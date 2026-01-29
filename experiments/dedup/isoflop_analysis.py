# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scaling Ladder Analysis for Deduplicated of fineweb-edu.

This experiment runs scaling ladder analysis on the isoflop training sweeps
on both vanilla fineweb-edu and deduplicated fineweb-edu mixtures.
"""

import logging
import humanfriendly
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.dedup_commons import DedupMode, DedupConfig, deduplicate
from marin.processing.tokenize import tokenize
from marin.processing.tokenize.data_configs import lm_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig
from levanter.data.text import LMMixtureDatasetConfig
from experiments.pretraining_datasets.simple import downloads
from experiments.isoflop_sweep import (
    IsoFlopAnalysisConfig,
    MARIN_2025_RECIPE,
    create_isoflop_sweep_steps,
    run_isoflop_analysis_step,
)
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

logger = logging.getLogger(__name__)


# --- Configuration ---
BUDGETS: list[float] = [1e18, 3e18, 6e18]
EXPERIMENT_NAME_PREFIX = "fineweb-edu-scaling-laws"
TOKENIZER = "stanford-crfm/marin-tokenizer"


def _get_vanilla_data_mixture(*, variant: str) -> LMMixtureDatasetConfig:
    """Vanilla fineweb-edu mixture"""
    tokenize_config = TokenizeConfig(
        train_paths=[downloads[variant]],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(TOKENIZER),
        window_size_bytes=humanfriendly.parse_size("512MB", binary=True),
    )

    tokenize_step = ExecutorStep(
        name=f"tokenized/{variant}",
        fn=tokenize,
        config=tokenize_config,
    )

    vanilla_fineweb_edu_mixture_config = lm_data_config(tokenize_step)
    return vanilla_fineweb_edu_mixture_config


def _get_deduped_data_mixture(*, variant: str, mode: DedupMode, max_parallelism: int = 1024) -> LMMixtureDatasetConfig:
    """Dedup fineweb-edu mixture"""
    dedup_config = DedupConfig(
        input_paths=downloads[variant],
        mode=mode,
        processes=max_parallelism,
    )

    dedup_step = ExecutorStep(
        name=f"dedup/{variant}_{mode.lower()}",
        fn=deduplicate,
        config=dedup_config,
    )

    dedup_mode_to_filter_type = {
        DedupMode.EXACT_PARAGRAPH: FilterType.REMOVE_SPANS,
        DedupMode.EXACT_DOCUMENT: FilterType.REMOVE_DOC,
        DedupMode.FUZZY_DOCUMENT: FilterType.REMOVE_DOC,
    }

    consolidate_step = ExecutorStep(
        name=f"clean/{variant}_{mode.lower()}",
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=downloads[variant],
            output_path=this_output_path(),
            # TODO (rav): can we infer this?
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=dedup_mode_to_filter_type[mode],
                    # TODO (rav): it's not cool that we need to cd to data!
                    attribute_path=dedup_step.cd("data"),
                    name=str(mode),
                ),
            ],
        ),
    )

    tokenize_config = TokenizeConfig(
        train_paths=[consolidate_step],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(TOKENIZER),
        window_size_bytes=humanfriendly.parse_size("512MB", binary=True),
    )

    tokenize_step = ExecutorStep(
        name=f"tokenized/dedup_{variant}_{mode.lower()}",
        fn=tokenize,
        config=tokenize_config,
    )

    deduped_fineweb_edu_mixture_config = lm_data_config(tokenize_step)
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


analysis_step = ExecutorStep(
    name=f"{EXPERIMENT_NAME_PREFIX}-analysis",
    fn=run_isoflop_analysis_step,
    config=IsoFlopAnalysisConfig(
        training_runs=[r.as_input_name() for r in training_steps],
        output_path=this_output_path(),
        recipe=MARIN_2025_RECIPE,
    ),
)

all_steps = [analysis_step]


if __name__ == "__main__":
    executor_main(steps=all_steps)
