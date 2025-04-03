#!/usr/bin/env python
"""
Experiment: Cooldown Data Mix Ablations
Author: Will Held
Date: 2025-04-02

Tests four data mixes for cooldown:
1. Original (DCLM + StarCoder)
2. Nemotron only
3. Nemotron + Code + Dolmino
4. Full mix (Nemotron + Code + Dolmino + Arxiv + Wikipedia + Stackexchange)

Metrics: Paloma Loss, Tulu3 Validation Loss, MMLU Accuracy
"""

# Core imports for experiment functionality
from experiments.cooldown_quality import default_quality_ablation, QualityAblationConfig
from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron_steps
from experiments.dolmino.tokenize_dolmino import tokenize_dolmino_steps, dolmino_math_tokenized_llama3
from marin.execution.executor import executor_main, ExecutorStep, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.processing.tokenize import TokenizeConfig, tokenize
from experiments.llama import llama3_tokenizer

# Existing tokenized datasets
from experiments.exp846_arxiv_cooldown import markdownified_arxiv_tokenized
from experiments.exp845_wikipedia_cooldown import wikipedia_cooldown_ablation
from experiments.exp649_stack_exchange_training import tokenized_stackexchange


# Data mix definitions
def get_original_mix():
    """Original mix: DCLM + StarCoder + ProofPile"""
    return lm_mixture_data_config(
        components={**dclm_components_llama3, "tulu_sft": tulu3_flat_llama_tokenized_as_validation},
        weights=DCLM_MIXTURE_WEIGHTS,
    )


def get_nemotron_only_mix():
    """Nemotron-only mix"""
    nemotron_steps = tokenize_nemotron_steps()
    return lm_mixture_data_config(
        components={**nemotron_steps, "tulu_sft": tulu3_flat_llama_tokenized_as_validation}, weights=NEMOTRON_WEIGHTS
    )


def get_nemotron_code_dolmino_mix():
    """Nemotron + Code + Dolmino mix"""
    # Gather components
    components = {
        **tokenize_nemotron_steps(),
        "starcoderdata": dclm_components_llama3["starcoderdata"],
        "proofpile_2": dclm_components_llama3["proofpile_2"],
        "all_math": dolmino_math_tokenized_llama3,
        "tulu_sft": tulu3_flat_llama_tokenized_as_validation,
        **tokenize_dolmino_steps(),
    }

    # Define weights
    weights = {
        **{k: v * 0.6 for k, v in NEMOTRON_WEIGHTS.items()},
        "starcoderdata": 0.25,
        "proofpile_2": 0.25,
        "flan": 0.017 * 10,
        "dolmino/pes2o": 0.0581 * 10,
        "dolmino/stackexchange": 0.0171 * 10,
        "dolmino/wiki": 0.00365 * 10,
        "all_math": 0.00422 * 10,
    }

    return lm_mixture_data_config(components=components, weights=weights)


def get_full_mix():
    """Full mix: Nemotron + Code + Dolmino + Additional Data"""
    # Combine all components
    medu_mmlu_science_qa_tokenized = default_tokenize(
        name="medu-mmlu-science-qa",
        dataset="gs://marin-us-east1/documents/medu-mmlu-science-llama8b-qa-whole-1a419d",
        tokenizer=llama3_tokenizer,
    )
    
    components = {
        **tokenize_nemotron_steps(),
        "starcoderdata": dclm_components_llama3["starcoderdata"],
        "proofpile_2": dclm_components_llama3["proofpile_2"],
        "all_math": dolmino_math_tokenized_llama3,
        "arxiv_markdownified": markdownified_arxiv_tokenized,
        "wikipedia_markdown": output_path_of(wikipedia_cooldown_ablation),
        "stackexchange_custom": tokenized_stackexchange,
        "medu_science_qa": medu_mmlu_science_qa_tokenized,
        "tulu_sft": tulu3_flat_llama_tokenized_as_validation,
        **tokenize_dolmino_steps(),
    }

    # Define weights
    weights = {
        **{k: v * 0.6 for k, v in NEMOTRON_WEIGHTS.items()},
        "starcoderdata": 0.25,
        "proofpile_2": 0.25,
        "flan": 0.017 * 10,
        "dolmino/pes2o": 0.0581 * 5,
        "dolmino/stackexchange": 0.0171 * 5,
        "dolmino/wiki": 0.00365 * 5,
        "all_math": 0.00422 * 10,
        "arxiv_markdownified": 0.0581 * 5,
        "stackexchange_custom": 0.0171 * 5,
        "wikipedia_markdown": 0.00365 * 5,
        "medu_science_qa": 0.0012 * 5
    }

    return lm_mixture_data_config(components=components, weights=weights)


def run_cooldown_ablation():
    """Run the cooldown data mix ablation experiment"""
    # Define data mixes to test
    data_mixes = {
        "original_pt_mix": get_original_mix(),
        "nemotron_pt_only": get_nemotron_only_mix(),
        "nemotron_code_dolmino": get_nemotron_code_dolmino_mix(),
        "nemotron_code_dolmino_misc": get_full_mix(),
    }

    # Configure experiment
    config = QualityAblationConfig(
        baseline_weight=0.0,
        mcq_weight=0.0,
        candidate_weight=1.0,  # 100% focus on data mix
        tpu_type="v4-128",
        model_name_prefix="pretrain_v_hq_ablation",
    )

    # Run ablation for each mix
    results = {}
    for mix_name, data_mix in data_mixes.items():
        # Run ablation
        ablation_model = default_quality_ablation(candidate_tokenized=data_mix, config=config, mix_name=mix_name)

        results[mix_name] = ablation_model

    return results


if __name__ == "__main__":
    executor_main(run_cooldown_ablation)
