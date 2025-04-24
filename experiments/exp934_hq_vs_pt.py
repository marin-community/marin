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

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3
from experiments.defaults import default_anneal, default_tokenize
from experiments.dolmino.tokenize_dolmino import dolmino_math_tokenized_llama3, tokenize_dolmino_steps
from experiments.instruction_datasets import tulu3_flat_llama_tokenized_as_validation
from experiments.llama import llama3_tokenizer
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron_steps
from marin.execution.executor import InputName, executor_main
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_mixture_data_config

# 1. Original mix: DCLM + StarCoder + ProofPile
original_mix = lm_mixture_data_config(
    components={**dclm_components_llama3},
    weights=DCLM_MIXTURE_WEIGHTS,
)

# 2. Nemotron-only mix
nemotron_steps = tokenize_nemotron_steps()
nemotron_only_mix = lm_mixture_data_config(components={**nemotron_steps}, weights=NEMOTRON_WEIGHTS)

# 3. Nemotron + Code + Dolmino mix
nemotron_code_dolmino_components = {
    **tokenize_nemotron_steps(),
    "starcoderdata": dclm_components_llama3["starcoderdata"],
    "proofpile_2": dclm_components_llama3["proofpile_2"],
    "all_math": dolmino_math_tokenized_llama3,
    **tokenize_dolmino_steps(),
}

nemotron_code_dolmino_weights = {
    **{k: v * 0.6 for k, v in NEMOTRON_WEIGHTS.items()},
    "starcoderdata": 0.25,
    "proofpile_2": 0.25,
    "dolmino/flan": 0.017 * 10,
    "dolmino/pes2o": 0.0581 * 10,
    "dolmino/stackexchange": 0.0171 * 10,
    "dolmino/wiki": 0.00365 * 10,
    "all_math": 0.00422 * 10,
}

nemotron_code_dolmino_mix = lm_mixture_data_config(
    components=nemotron_code_dolmino_components, weights=nemotron_code_dolmino_weights
)

# 4. Full mix with everything
# Create the medu science QA dataset
# MMLU Science QA tokenization
medu_mmlu_science_qa_tokenized = default_tokenize(
    name="medu-mmlu-science-qa",
    dataset="gs://marin-us-east1/documents/medu-mmlu-science-llama8b-qa-whole-1a419d",
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/medu-mmlu-science-qa-c64fda")

# Wikipedia tokenization
md_wiki_tokenized = default_tokenize(
    name="wikipedia",
    dataset=InputName.hardcoded("documents/wikipedia-resiliparse-custom-fork-2569de/20241201/"),
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/wikipedia-6980f2")

# Arxiv tokenization
md_arxiv_tokenized = default_tokenize(
    name="arxiv-no-problem",
    dataset=InputName.hardcoded("documents/ar5iv/ar5iv-04-2024-no-problem-3971ff/resiliparse-custom-fork"),
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/arxiv-no-problem-a3e054")

# Stackexchange tokenization
md_stackexchange_tokenized = default_tokenize(
    name="stackexchange",
    dataset="gs://marin-us-central2/documents/stackexchange-resiliparse-custom-fork-ab41ad",
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/stackexchange-621b94")

full_mix_components = {
    **tokenize_nemotron_steps(),
    "starcoderdata": dclm_components_llama3["starcoderdata"],
    "proofpile_2": dclm_components_llama3["proofpile_2"],
    "all_math": dolmino_math_tokenized_llama3,
    "arxiv_markdownified": md_arxiv_tokenized,
    "wikipedia_markdown": md_wiki_tokenized,
    "stackexchange_custom": md_stackexchange_tokenized,
    "medu_science_qa": medu_mmlu_science_qa_tokenized,
    **tokenize_dolmino_steps(),
}

# weights based on either compressed TB or teratokens, which is roughly equivalent
# scale of 5 to oversample higher quality data
full_mix_weights = {
    **{k: v * 0.6 for k, v in NEMOTRON_WEIGHTS.items()},
    "starcoderdata": 0.25,
    "proofpile_2": 0.25,
    "dolmino/flan": 0.017 * 10,
    "dolmino/pes2o": 0.0581 * 5,
    "dolmino/stackexchange": 0.0171 * 5,
    "dolmino/wiki": 0.00365 * 5,
    "all_math": 0.00422 * 10,
    "arxiv_markdownified": 0.0581 * 5,
    "stackexchange_custom": 0.0171 * 5,
    "wikipedia_markdown": 0.00365 * 5,
    "medu_science_qa": 0.0012 * 5,
}

full_mix = lm_mixture_data_config(components=full_mix_components, weights=full_mix_weights)

# Dictionary of all mixes
data_mixes = {
    "original_pt_mix": original_mix,
    "nemotron_pt_only": nemotron_only_mix,
    "nemotron_code_dolmino": nemotron_code_dolmino_mix,
    "nemotron_code_dolmino_misc": full_mix,
}

# Default parameters for annealing
anneal_tokens = 50_000_000_000  # 50B tokens
tpu_type = "v4-128"
node_count = 4
checkpoint = "gs://marin-us-central2/checkpoints/llama-8b-tootsie-adept-phoenix/checkpoints/step-1240000"


def run_cooldown_ablation():
    # Apply annealing to each mix
    results = []
    for mix_name, data_mix in data_mixes.items():
        # Create AnnealConfig
        anneal_config = AnnealConfig(
            initialize_from_checkpoint_path=checkpoint,
            dataset_config=add_validation_sets_to_mixture(
                data_mix, {"tulu_sft": tulu3_flat_llama_tokenized_as_validation}
            ),
            num_anneal_training_tokens=anneal_tokens,
            tpu_type=tpu_type,
            node_count=node_count,
            train_batch_size=2048,
        )

        # Run annealing
        model_name = f"pretrain_v_hq_ablation-phx1.24M-{mix_name}"
        results.append(
            default_anneal(
                name=model_name,
                anneal_config=anneal_config,
            )
        )

    return results


if __name__ == "__main__":
    executor_main(
        steps=run_cooldown_ablation(),
        description="Profiling Cooldowns on Pretraining data moving towards increasingly HQ Data",
    )
