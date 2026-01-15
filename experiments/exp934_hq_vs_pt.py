#!/usr/bin/env python
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
# In this experiment:
# PT = Pretraining
# HQ = High Quality

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, versioned
from marin.execution import step, StepContext, StepRef, deferred, output
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from marin.schemas.web.selectors import ARXIV_BLACKLISTED_SELECTORS, WIKI_BLACKLISTED_SELECTORS
from marin.transform.ar5iv.transform_ar5iv import Ar5ivExtractionConfig
from marin.transform.ar5iv.transform_ar5iv import process_ar5iv_dump as _process_ar5iv_dump
from marin.transform.wikipedia.transform_wikipedia import WikiExtractionConfig
from marin.transform.wikipedia.transform_wikipedia import process_wiki_dump as _process_wiki_dump

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal, default_tokenize
from experiments.exp822_stackexchange_markdownify import stackexchange_text_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer
from experiments.posttrain.instruction_datasets import tulu3_flat_llama_tokenized_as_validation
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3
from experiments.pretraining_datasets.dolmino import tokenize_dolmino, tokenize_dolmino_math

# Mark library functions as deferred
process_wiki_dump = deferred(_process_wiki_dump)
process_ar5iv_dump = deferred(_process_ar5iv_dump)

# 1. Original mix: DCLM + StarCoder + ProofPile
original_mix = lm_mixture_data_config(
    components={**dclm_components_llama3},
    weights=DCLM_MIXTURE_WEIGHTS,
    permutation_type="linear",
)

# 2. Nemotron-only mix
nemotron_steps = tokenize_nemotron()
nemotron_only_mix = lm_mixture_data_config(
    components={**nemotron_steps},
    weights=NEMOTRON_WEIGHTS,
    permutation_type="linear",
)

# 3. Nemotron + Code + Dolmino mix
nemotron_code_dolmino_components = {
    **tokenize_nemotron(),
    "starcoderdata": dclm_components_llama3["starcoderdata"],
    "proofpile_2": dclm_components_llama3["proofpile_2"],
    "all_math": tokenize_dolmino_math(),
    **tokenize_dolmino(),
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
    components=nemotron_code_dolmino_components,
    weights=nemotron_code_dolmino_weights,
    permutation_type="linear",
)

# 4. Full mix with everything

# Wikipedia resiliparse custom fork step (data already exists at hardcoded path)
@step(name="documents/wikipedia-resiliparse-custom-fork")
def wikipedia_resiliparse_custom_fork():
    return process_wiki_dump(WikiExtractionConfig(
        input_path="gs://marin-us-central2/raw/wikipedia-a7dad0/20241201",
        revision=versioned("20241201"),
        output_path=output(),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            links=False,
            skip_elements=WIKI_BLACKLISTED_SELECTORS,
            markdownify_config=HtmlToMarkdownConfig(include_images=False, include_links=False),
        ),
        remove_reference_section=versioned(True),
        digit_threshold=versioned(50),
        word_threshold=versioned(70),
        special_char_threshold=versioned(50),
    )).with_output_path("documents/wikipedia-resiliparse-custom-fork-2569de").cd("20241201")

# ar5iv resiliparse custom fork step (data already exists at hardcoded path)
@step(name="documents/ar5iv/ar5iv-04-2024-no-problem")
def ar5iv_no_problem_resiliparse_custom_fork():
    return process_ar5iv_dump(Ar5ivExtractionConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/ar5iv-04-2024-no-problem-49c4e3/202404",
        revision="042024",
        output_path=output() / "resiliparse-custom-fork",
        extract_method=versioned("resiliparse"),
        extract_config=ResiliparseConfig(
            links=versioned(False),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
        ),
        remove_reference_section=versioned(True),
    )).with_output_path("documents/ar5iv/ar5iv-04-2024-no-problem-3971f")

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
    dataset=wikipedia_resiliparse_custom_fork(),
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/wikipedia-6980f2")

# Arxiv tokenization
md_arxiv_tokenized = default_tokenize(
    name="arxiv-no-problem",
    dataset=ar5iv_no_problem_resiliparse_custom_fork(),
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/arxiv-no-problem-a3e054")

# Stackexchange tokenization
md_stackexchange_tokenized = default_tokenize(
    name="stackexchange",
    dataset=stackexchange_text_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/stackexchange-621b94")

pt_vs_hq_components = {
    **tokenize_nemotron(),
    "starcoderdata": dclm_components_llama3["starcoderdata"],
    "proofpile_2": dclm_components_llama3["proofpile_2"],
    "all_math": tokenize_dolmino_math(),
    "arxiv_markdownified": md_arxiv_tokenized,
    "wikipedia_markdown": md_wiki_tokenized,
    "stackexchange_custom": md_stackexchange_tokenized,
    "medu_science_qa": medu_mmlu_science_qa_tokenized,
    **tokenize_dolmino(),
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

full_mix = lm_mixture_data_config(
    components=pt_vs_hq_components,
    weights=full_mix_weights,
    permutation_type="linear",
)

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
            resources=ResourceConfig.with_tpu(tpu_type, slice_count=node_count),
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
