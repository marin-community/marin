# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Dataset definitions for high-quality data sources used in cooldown/midtraining.

This module provides the `pt_vs_hq_components` dictionary containing tokenized
datasets used by various training experiments.
"""

from marin.datakit.download.ar5iv import ar5iv_step
from marin.datakit.download.wikipedia import download_wikipedia_step
from marin.execution.executor import ExecutorStep, mirrored, this_output_path, versioned
from marin.execution.step_spec import StepSpec
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
        input_path=mirrored(versioned("documents/stackexchange/v2024-04-02/md-complete"), budget_gb=50),
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

_wikipedia_download = download_wikipedia_step()

# Wikipedia resiliparse custom fork step
_wikipedia_transform = StepSpec(
    name="documents/wikipedia-resiliparse-custom-fork",
    fn=lambda output_path: process_wiki_dump(
        WikiExtractionConfig(
            input_path=f"{_wikipedia_download.output_path}/20241201",
            revision="20241201",
            output_path=output_path,
            extract_method="resiliparse",
            extract_config=ResiliparseConfig(
                links=False,
                skip_elements=WIKI_BLACKLISTED_SELECTORS,
                markdownify_config=HtmlToMarkdownConfig(include_images=False, include_links=False),
            ),
            remove_reference_section=True,
            digit_threshold=50,
            word_threshold=70,
            special_char_threshold=50,
        )
    ),
    deps=[_wikipedia_download],
    hash_attrs={"revision": "20241201", "extract_method": "resiliparse"},
    override_output_path="documents/wikipedia-resiliparse-custom-fork-2569de",
)
wikipedia_resiliparse_custom_fork = _wikipedia_transform.as_executor_step().cd("20241201")

_ar5iv_download = ar5iv_step(
    input_path="gs://marin-us-central2/raw/ar5iv/ar5iv-04-2024-no-problem.zip",
    override_output_path="raw/ar5iv/ar5iv-04-2024-no-problem-49c4e3",
)

# ar5iv resiliparse custom fork step
_ar5iv_transform = StepSpec(
    name="documents/ar5iv/ar5iv-04-2024-no-problem",
    fn=lambda output_path: process_ar5iv_dump(
        Ar5ivExtractionConfig(
            input_path=f"{_ar5iv_download.output_path}/202404",
            revision="042024",
            output_path=output_path,
            extract_method="resiliparse",
            extract_config=ResiliparseConfig(
                links=False,
                prepend_title=True,
                skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            ),
            remove_reference_section=True,
        )
    ),
    deps=[_ar5iv_download],
    hash_attrs={"revision": "042024", "extract_method": "resiliparse"},
    override_output_path="documents/ar5iv/ar5iv-04-2024-no-problem-3971f",
)
ar5iv_no_problem_resiliparse_custom_fork = _ar5iv_transform.as_executor_step()

# MMLU Science QA tokenization
medu_mmlu_science_qa_tokenized = default_tokenize(
    name="medu-mmlu-science-qa",
    dataset=mirrored("documents/medu-mmlu-science-llama8b-qa-whole-1a419d", budget_gb=30),
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
