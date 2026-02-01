# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry of ISOFlop sweep runs.

This module defines the sweep configurations for different datasets and budgets.
Recipe implementations live in experiments/scaling_law_sweeps/.
"""

from experiments.common_pile.tokenize_common_pile import comma_main_mixture
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple import downloads
from experiments.scaling_law_sweeps import adamh as adamh_recipe
from experiments.scaling_law_sweeps import c_adamc as c_adamc_recipe
from experiments.scaling_law_sweeps import muonh as muonh_recipe
from experiments.tootsie.exp1295_32b import nemotron_mix
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# --- Budget configurations ---
LEGACY_BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)
MUONH_NEMOTRON_BUDGETS: tuple[float, ...] = (1e18, 3e18, 6e18, 9e18, 1e19, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)
ADAMH_NEMOTRON_BUDGETS: tuple[float, ...] = (1e18, 3e18, 6e18, 9e18, 1e19, 1.8e19, 3e19)

# --- Tokenized Datasets ---
dclm_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=downloads["dclm_baseline"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dclm_baseline-0206f1/")

dclm_mix = lm_mixture_data_config(
    components={"dclm": dclm_tokenized},
    weights={"dclm": 1.0},
    num_validation_sequences={"dclm": 1024},
)

dolma3_mix_tokenized = default_tokenize(
    name="dolma3_mix-150B-1025",
    dataset=downloads["dolma3_mix_150b_1025"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dolma3_mix-150B-1025-15d04ee/")

dolma3_mix = lm_mixture_data_config(
    components={"dolma3_mix-150B-1025": dolma3_mix_tokenized},
    weights={"dolma3_mix-150B-1025": 1.0},
    num_validation_sequences={"dolma3_mix-150B-1025": 1024},
)

# --- Original C-AdamC sweeps (from main) ---
SCALING_SUITES = {
    "nemotron": c_adamc_recipe.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="nemo-wider-depth-adapt",
        budgets=LEGACY_BUDGETS,
    ),
    "common_pile": c_adamc_recipe.create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(permutation_type="linear"),
        experiment_name="comma-mix",
        budgets=LEGACY_BUDGETS,
    ),
    "common_pile_feistel": c_adamc_recipe.create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(permutation_type="feistel"),
        experiment_name="comma-mix-feistel",
        budgets=LEGACY_BUDGETS,
    ),
    "dclm-default": c_adamc_recipe.create_isoflop_sweep_steps(
        tokenized=dclm_mix,
        experiment_name="dclm-default",
        budgets=LEGACY_BUDGETS,
    ),
    "dolma3_mix_150b": c_adamc_recipe.create_isoflop_sweep_steps(
        tokenized=dolma3_mix,
        experiment_name="dolma3-mix-150b-1025",
        budgets=LEGACY_BUDGETS,
    ),
    # --- New MuonH sweeps ---
    "nemotron-muonh": muonh_recipe.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="muonh-scale-lr",
        budgets=MUONH_NEMOTRON_BUDGETS,
    ),
    # --- New AdamH sweeps ---
    "nemotron-adamh": adamh_recipe.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="scale-lr-adamh",
        budgets=ADAMH_NEMOTRON_BUDGETS,
    ),
}

if __name__ == "__main__":
    steps, _ = SCALING_SUITES["nemotron-muonh"]
    executor_main(steps=steps)
