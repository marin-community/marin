# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Dataset definitions for high-quality data sources used in cooldown/midtraining.

This module provides the `pt_vs_hq_components` dictionary containing tokenized
datasets used by various training experiments.
"""

from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from marin.schemas.web.selectors import ARXIV_BLACKLISTED_SELECTORS, WIKI_BLACKLISTED_SELECTORS
from marin.transform.ar5iv.transform_ar5iv import Ar5ivExtractionConfig, process_ar5iv_dump
from marin.transform.stackexchange.transform_stackexchange import (
    StackExchangeExtractionConfig,
    process_stackexchange_dump,
)
from marin.transform.wikipedia.transform_wikipedia import WikiExtractionConfig, process_wiki_dump

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.dolmino import tokenize_dolmino, tokenize_dolmino_math

# Stack Exchange resiliparse custom fork (inlined from deleted exp822)
stackexchange_text_resiliparse_custom_fork = ExecutorStep(
    name="documents/stackexchange-resiliparse-custom-fork",
    fn=process_stackexchange_dump,
    config=StackExchangeExtractionConfig(
        input_path=versioned("gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-complete"),
        output_path=this_output_path(),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            links=False,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
    ),
).with_output_path("documents/stackexchange-resiliparse-custom-fork-ab41ad")

# Wikipedia resiliparse custom fork step (data already exists at hardcoded path)
wikipedia_resiliparse_custom_fork = (
    ExecutorStep(
        name="documents/wikipedia-resiliparse-custom-fork",
        fn=process_wiki_dump,
        config=WikiExtractionConfig(
            input_path="gs://marin-us-central2/raw/wikipedia-a7dad0/20241201",
            revision=versioned("20241201"),
            output_path=this_output_path(),
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
        ),
    )
    .with_output_path("documents/wikipedia-resiliparse-custom-fork-2569de")
    .cd("20241201")
)

# ar5iv resiliparse custom fork step (data already exists at hardcoded path)
ar5iv_no_problem_resiliparse_custom_fork = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/ar5iv-04-2024-no-problem-49c4e3/202404",
        revision="042024",
        output_path=this_output_path("resiliparse-custom-fork"),
        extract_method=versioned("resiliparse"),
        extract_config=ResiliparseConfig(
            links=versioned(False),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
        ),
        remove_reference_section=versioned(True),
    ),
).with_output_path("documents/ar5iv/ar5iv-04-2024-no-problem-3971f")

# MMLU Science QA tokenization
medu_mmlu_science_qa_tokenized = default_tokenize(
    name="medu-mmlu-science-qa",
    dataset="gs://marin-us-east1/documents/medu-mmlu-science-llama8b-qa-whole-1a419d",
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/medu-mmlu-science-qa-c64fda")

# Wikipedia tokenization
md_wiki_tokenized = default_tokenize(
    name="wikipedia",
    dataset=wikipedia_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/wikipedia-6980f2")

# Arxiv tokenization
md_arxiv_tokenized = default_tokenize(
    name="arxiv-no-problem",
    dataset=ar5iv_no_problem_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/arxiv-no-problem-a3e054")

# Stackexchange tokenization
md_stackexchange_tokenized = default_tokenize(
    name="stackexchange",
    dataset=stackexchange_text_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/stackexchange-621b94")

# Main export: high-quality data components for cooldown/midtraining
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
