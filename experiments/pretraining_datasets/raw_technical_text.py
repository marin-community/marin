# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw technical-text tranche.

A unioned mixture of surface-preserving technical sources: raw source files,
repo-adjacent technical text (READMEs, PEPs, IRC logs, GitHub-archive issues,
StackExchange threads, Kaggle notebooks), and raw-form scientific text
(LaTeX/arXiv, open-web-math, algebraic-stack). Everything listed here is
tokenized directly from the upstream record without HTML-to-markdown
conversion or whitespace compaction, so literal URLs, identifiers, numbers,
and whitespace/layout survive into training.

Motivation: perplexity-gap reports against Llama 3.1 8B and Qwen3 8B show
Marin underperforms on raw technical surface forms (github_cpp, github_python,
dolma_100_programing_languages, URL/number/whitespace byte buckets). Most
non-code mixture components in the current pretraining mix are cleaned or
normalized, so the model rarely sees literal technical surface forms. See
issue #4961.
"""

from experiments.common_pile.tokenize_common_pile import common_pile_tokenized, stackv2
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.dolma import tokenize_dolma
from experiments.pretraining_datasets.simple import tokenized as simple_tokenized
from experiments.pretraining_datasets.starcoder2_extras import tokenize_starcoder2_extras
from marin.datakit.download.starcoder2_extras import SUBSETS as _STARCODER2_EXTRAS_SUBSET_ORDER
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.data_configs import TokenizerStep

# StarCoder2-extras subsets that carry raw source / repo-adjacent technical
# text. These are tokenized from the upstream parquet ``content`` field; the
# normalize step applied upstream preserves the source verbatim.
STARCODER2_EXTRAS_SUBSETS: tuple[str, ...] = (
    "ir_cpp",
    "ir_python",
    "ir_rust",
    "ir_low_resource",
    "documentation",
    "kaggle",
)

# Common Pile splits that preserve raw technical surface forms:
#   - ``stackv2`` / ``stackv2_edu``: raw source files
#   - ``stackv2_html``: raw HTML extracted from the Stack v2
#   - ``github_archive``: raw GitHub issues / PR bodies / commits
#   - ``ubuntu_irc``: raw IRC logs (preserved timestamps, nicks, URLs)
#   - ``python_enhancement_proposals``: PEPs (raw RST/markdown)
#   - ``stackexchange``: raw StackExchange dumps with code fences preserved
COMMON_PILE_SUBSETS: tuple[str, ...] = (
    "stackv2_edu",
    "stackv2_html",
    "github_archive",
    "ubuntu_irc",
    "python_enhancement_proposals",
    "stackexchange",
)

# Dolma 1.7 splits that are raw-ish and retain layout:
#   - ``algebraic-stack`` / ``open-web-math``: raw LaTeX/math surface form
#   - ``arxiv``: arXiv sources, including raw LaTeX
#   - ``stackexchange``: Q&A threads with code fences preserved
DOLMA_SUBSETS: tuple[str, ...] = (
    "algebraic-stack",
    "arxiv",
    "open-web-math",
    "stackexchange",
)


def raw_technical_text_components(*, tokenizer: str = llama3_tokenizer) -> dict[str, TokenizerStep]:
    """Return the tokenized components that make up the raw technical-text tranche.

    Keys mirror the existing registry naming:
      - ``starcoderdata``
      - ``starcoder2_extras/<subset>`` for STARCODER2_EXTRAS_SUBSETS
      - ``common_pile/<subset>`` for COMMON_PILE_SUBSETS + ``common_pile/stackv2``
      - ``dolma/<subset>`` for DOLMA_SUBSETS
    """
    components: dict[str, TokenizerStep] = {}

    # starcoderdata — reuse the pinned llama3 tokenized path when available;
    # otherwise build a fresh tokenization step using the same text_key.
    if tokenizer == llama3_tokenizer:
        components["starcoderdata"] = simple_tokenized["starcoderdata"]
    else:
        from levanter.data.text import TextLmDatasetFormat

        from experiments.pretraining_datasets.simple import downloads as simple_downloads

        components["starcoderdata"] = default_tokenize(
            name="starcoderdata",
            dataset=simple_downloads["starcoderdata"],
            tokenizer=tokenizer,
            format=TextLmDatasetFormat(text_key="content"),
        )

    # starcoder2_extras — tokenize_starcoder2_extras iterates its internal
    # SUBSETS list in order, so zip back to names.
    starcoder2_steps = tokenize_starcoder2_extras(tokenizer=tokenizer)
    starcoder2_by_subset = dict(zip(_STARCODER2_EXTRAS_SUBSET_ORDER, starcoder2_steps, strict=True))
    for subset in STARCODER2_EXTRAS_SUBSETS:
        components[f"starcoder2_extras/{subset}"] = starcoder2_by_subset[subset]

    # Common Pile subsets. stackv2 (full) is intentionally not registered in
    # COMMON_PILE_DATASETS (it is license-restricted and excluded from the
    # default Comma mix), so tokenize it explicitly here.
    common_pile_steps = common_pile_tokenized(tokenizer=tokenizer)
    for subset in COMMON_PILE_SUBSETS:
        components[f"common_pile/{subset}"] = common_pile_steps[f"common_pile/{subset}"]
    components["common_pile/stackv2"] = default_tokenize(
        name="common_pile/stackv2",
        dataset=stackv2,
        tokenizer=tokenizer,
    )

    # Dolma raw splits.
    dolma_steps = tokenize_dolma(tokenizer=tokenizer)
    for subset in DOLMA_SUBSETS:
        components[f"dolma/{subset}"] = dolma_steps[f"dolma/{subset}"]

    return components


# Approximate token counts for each component (in billions of tokens). These
# are used as *relative* sampling weights within the tranche; absolute
# magnitudes only matter when embedding the tranche in a larger mixture.
#
# Sources:
#   - starcoderdata: https://huggingface.co/datasets/bigcode/starcoderdata (≈250B)
#   - Common Pile: derived from comma_v0.1_training_dataset main-stage weights
#     (teratoken counts) in experiments.common_pile.tokenize_common_pile.
#   - Dolma: sampling proportions from DOLMA_OLMO_MIXTURE_WEIGHTS (billions).
#   - Starcoder2-extras: approximate sizes from the bigcode dataset card;
#     these are order-of-magnitude and should be refined once per-subset
#     token counts are measured against the Marin tokenizer.
RAW_TECHNICAL_TEXT_TOKEN_COUNTS: dict[str, float] = {
    # Raw code files (B tokens).
    "starcoderdata": 250.0,
    # starcoder2_extras (B tokens, approximate).
    "starcoder2_extras/ir_cpp": 8.0,
    "starcoder2_extras/ir_python": 4.0,
    "starcoder2_extras/ir_rust": 2.0,
    "starcoder2_extras/ir_low_resource": 2.0,
    "starcoder2_extras/documentation": 3.0,
    "starcoder2_extras/kaggle": 3.0,
    # Common Pile (B tokens; COMMA main-stage teratokens * 1000).
    "common_pile/stackv2": 200.0,  # upper-bound estimate; stackv2 is large
    "common_pile/stackv2_edu": 135.6,
    "common_pile/stackv2_html": 2.4,
    "common_pile/github_archive": 66.0,
    "common_pile/ubuntu_irc": 11.4,
    "common_pile/python_enhancement_proposals": 0.02,
    "common_pile/stackexchange": 143.4,
    # Dolma raw splits (B tokens; DOLMA_OLMO_MIXTURE_WEIGHTS are billions).
    "dolma/algebraic-stack": 12.6,
    "dolma/arxiv": 28.0,
    "dolma/open-web-math": 12.6,
    "dolma/stackexchange": 19.6,
}


# Llama3-tokenized components + a mixture config, mirroring the
# ``dclm_components_llama3`` / ``dclm_mixture_config_llama3`` pair in
# ``experiments.pretraining_datasets.dclm``.
raw_technical_text_components_llama3 = raw_technical_text_components(tokenizer=llama3_tokenizer)
raw_technical_text_mixture_config_llama3 = lm_mixture_data_config(
    components=raw_technical_text_components_llama3,
    weights=RAW_TECHNICAL_TEXT_TOKEN_COUNTS,
)


if __name__ == "__main__":
    executor_main(steps=list(raw_technical_text_components_llama3.values()))
