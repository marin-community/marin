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
Canonical raw and tokenized pretraining datasets.

This module defines both raw dataset downloads and their canonical tokenization.
For multi-split datasets (dolmino, nemotron_cc, dolma), helper functions are provided
to generate tokenized versions of individual splits.
"""

import os.path

from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheOptions

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.download.nemotron_cc.download_nemotron_cc import NemotronIngressConfig, download_nemotron_cc
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

fineweb = ExecutorStep(
    name="raw/fineweb",
    fn=download_hf,
    config=DownloadConfig(hf_dataset_id="HuggingFaceFW/fineweb", revision="cd85054"),
    override_output_path="raw/fineweb",
)

fineweb_edu = ExecutorStep(
    name="raw/fineweb-edu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["data/**/*.parquet"],
    ),
    override_output_path="raw/fineweb-edu-c2beb4",
).cd("data")

slimpajama = ExecutorStep(
    name="raw/SlimPajama-627B",
    fn=download_hf,
    config=DownloadConfig(hf_dataset_id="cerebras/SlimPajama-627B", revision="2d0accd", append_sha_to_path=True),
    override_output_path="raw/SlimPajama-627B-262830",
).cd("2d0accd")

slimpajama_6b = ExecutorStep(
    name="raw/SlimPajama-6B",
    fn=download_hf,
    config=DownloadConfig(hf_dataset_id="DKYoon/SlimPajama-6B", revision="b5f90f4"),
    override_output_path="raw/SlimPajama-6B-be35b7",
)

dolma = ExecutorStep(
    name="raw/dolma",
    fn=download_hf,
    config=DownloadConfig(hf_dataset_id="allenai/dolma", revision="7f48140"),
    override_output_path="raw/dolma",
)

dclm_baseline_wrong = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        append_sha_to_path=True,
    ),
    override_output_path="raw/dclm_WRONG_20250211/",
).cd("a3b142c")

dclm_baseline = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=this_output_path(),
        append_sha_to_path=True,
    ),
    override_output_path="raw/dclm",
).cd("a3b142c")

the_stack_dedup = ExecutorStep(
    name="raw/the-stack-dedup",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="bigcode/the-stack-dedup",
        revision="17cad72",
        append_sha_to_path=True,
    ),
    override_output_path="raw/the-stack-dedup-4ba450",
).cd("17cad72")

proofpile_2 = ExecutorStep(
    name="raw/proof-pile-2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="EleutherAI/proof-pile-2",
        revision="901a927",
        append_sha_to_path=True,
    ),
    override_output_path="raw/proof-pile-2-f1b1d8",
).cd("901a927")

the_pile_openwebtext2 = ExecutorStep(
    name="raw/the_pile_openwebtext2",
    fn=download_hf,
    config=DownloadConfig(hf_dataset_id="vietgpt/the_pile_openwebtext2", revision="1de27c6"),
    override_output_path="raw/the_pile_openwebtext2",
)

starcoderdata = ExecutorStep(
    name="raw/starcoderdata",
    fn=download_hf,
    config=DownloadConfig(hf_dataset_id="bigcode/starcoderdata", revision="9fc30b5"),
    override_output_path="raw/starcoderdata-720c8c",
)

dolmino = (
    ExecutorStep(
        name="raw/dolmino-mix-1124",
        fn=download_hf,
        config=DownloadConfig(hf_dataset_id="allenai/dolmino-mix-1124", revision="bb54cab", append_sha_to_path=True),
    )
    .with_output_path("raw/dolmino-mix-1124-157960")
    .cd("bb54cab")
)

nemotron_cc = ExecutorStep(
    name="raw/nemotro-cc",
    fn=download_nemotron_cc,
    config=NemotronIngressConfig(),
    pip_dependency_groups=["download_transform"],
)


# ============================================================================
# TOKENIZED DATASETS
# ============================================================================
# The sections below define canonical tokenizations for the raw datasets above.
# These use llama3 tokenizer by default but can be customized with different
# tokenizer parameters.


# Lazy import to avoid circular dependencies
def _get_llama3_tokenizer():
    from experiments.llama import llama3_tokenizer

    return llama3_tokenizer


# For dolma 1.7, we hardcode the path since it was added before versioning
_DOLMA_V1_7_PATH = "raw/dolma/v1.7"


# ----------------------------------------------------------------------------
# DOLMA tokenization
# ----------------------------------------------------------------------------

# Sampling proportion comes from https://huggingface.co/datasets/allenai/dolma
DOLMA_OLMO_MIXTURE_WEIGHTS = {
    "dolma/algebraic-stack": 12.6,  # 12.6 * 1.0
    "dolma/arxiv": 28.0,  # 28.0 * 1.0
    "dolma/gutenberg": 5.3,  # 5.3 * 1.0
    "dolma/c4": 124.95,  # 249.9 * 0.5
    "dolma/cc": 597.75,  # 1,195.5 * 0.5
    "dolma/cc-news": 14.3,  # 1.0
    "dolma/falcon": 456.4,  # 1.0, refined web
    "dolma/megawika": 4.6,  # 1.0
    "dolma/open-web-math": 12.6,  # 1.0
    "dolma/pes2o": 57.2,  # 1.0
    "dolma/reddit": 79.9,  # 1.0
    "dolma/stackexchange": 19.6,  # 1.0
    "dolma/starcoder": 263.8,  # 1.0
    "dolma/flan": 16.5,  # 6.5 * 1.0
    "dolma/wiki": 7.4,  # 3.7 * 2.0
}

DOLMA_DATASETS = {
    "algebraic-stack": ["algebraic-stack-train-{0000..0015}.json.gz"],
    "arxiv": ["arxiv-{0000..0099}.json.gz"],
    "gutenberg": ["books-{0000..0002}.json.gz"],
    "c4": ["c4-{0000..0170}.json.gz"],
    "cc": [
        "cc_en_head-{0000..0274}.json.gz",
        "cc_en_middle-{0000..0238}.json.gz",
        "cc_en_middle-{0240..0379}.json.gz",
        "cc_en_tail-{0000..0152}.json.gz",
        "cc_en_tail-{0154..0444}.json.gz",
    ],
    "cc-news": ["cc_news_head-{0000..0004}.json.gz", "cc_news_middle-{0000..0002}.json.gz", "cc_news_tail-0000.json.gz"],
    "falcon": ["falcon-{0000..0499}.json.gz"],
    "megawika": ["megawika-{0000..0261}.json.gz"],
    "open-web-math": ["open-web-math-train-{0000..0012}.json.gz"],
    "pes2o": ["pes2o-{0000..0025}.json.gz"],
    "reddit": ["reddit-{0000..0077}.json.gz"],
    "stackexchange": ["stackexchange-{0000..0025}.json.gz"],
    "starcoder": ["starcoder-{0000..0048}.json.gz"],
    "flan": ["tulu_flan-{0000..0065}.json.gz"],
    "wiki": ["wiki-{0000..0001}.json.gz"],
}

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
DOLMA_LLAMA3_OVERRIDES = {
    "c4": "tokenized/dolma/c4-e0e5ec",
    "cc": "tokenized/dolma/cc-74b017",
    "cc-news": "tokenized/dolma/cc-news-625d3e",
    "falcon": "tokenized/dolma/falcon-da8fd0",
    "flan": "tokenized/dolma/flan-a99cb2",
    "gutenberg": "tokenized/dolma/gutenberg-f9eb99",
    "reddit": "tokenized/dolma/reddit-62a64a",
    "starcoder": "tokenized/dolma/starcoder-8b6089",
    "algebraic-stack": "tokenized/dolma/algebraic-stack-cc00cf",
    "arxiv": "tokenized/dolma/arxiv-07a51f",
    "megawika": "tokenized/dolma/megawika-34abf2",
    "open-web-math": "tokenized/dolma/open-web-math-79823d",
    "pes2o": "tokenized/dolma/pes2o-538363",
    "stackexchange": "tokenized/dolma/stackexchange-adfc49",
    "wiki": "tokenized/dolma/wiki-212315",
}


def tokenize_dolma_steps(*, base_path="tokenized/", tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """
    Generate tokenization steps for all Dolma 1.7 dataset splits.

    Args:
        base_path: Base directory for tokenized output
        tokenizer: Tokenizer name (defaults to llama3)

    Returns:
        Dictionary mapping split names (e.g., "dolma/c4") to tokenization steps
    """
    if tokenizer is None:
        tokenizer = _get_llama3_tokenizer()

    dolma_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, files in DOLMA_DATASETS.items():
        step = ExecutorStep(
            name=os.path.join(base_path, "dolma", dataset),
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=[os.path.join(_DOLMA_V1_7_PATH, file) for file in files],
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
            pip_dependency_groups=["sentencepiece"],
        )

        if tokenizer == _get_llama3_tokenizer() and dataset in DOLMA_LLAMA3_OVERRIDES:
            step = step.with_output_path(DOLMA_LLAMA3_OVERRIDES[dataset])
        dolma_steps[os.path.join("dolma", dataset)] = step

    return dolma_steps


# ----------------------------------------------------------------------------
# DOLMINO tokenization
# ----------------------------------------------------------------------------

_dolmino_base_dir = dolmino.cd("data")

# The following dataset splits define file patterns for each split.
DOLMINO_DATASETS = {
    "dclm": ["**/*.json.zst"],
    "flan": ["**/*.json.gz"],
    "math/codesearchnet-owmfilter": ["**/*.jsonl.gz"],
    "math/dolmino_math_synth": ["**/*.jsonl"],
    "math/gsm8k": ["**/*.jsonl.zst"],
    "math/mathcoder2-synthmath": ["**/*.jsonl"],
    "math/metamath-owmfilter": ["**/*.jsonl.gz"],
    "math/tinyGSM-MIND": ["**/*.jsonl.gz"],
    "math/tulu_math": ["**/*.jsonl"],
    "pes2o": ["**/*.json.gz"],
    "stackexchange": ["**/*.json.gz"],
    "wiki": ["**/*.json.gz"],
}

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
DOLMINO_LLAMA3_OVERRIDES = {
    "dclm": "tokenized/dolmino/dclm-6c18eb",
    "flan": "tokenized/dolmino/flan-d71ec1",
    "math/codesearchnet-owmfilter": "tokenized/dolmino/math/codesearchnet-owmfilter-fd2640",
    "math/dolmino_math_synth": "tokenized/dolmino/math/dolmino_math_synth-11f876",
    "math/gsm8k": "tokenized/dolmino/math/gsm8k-902e8b",
    "math/mathcoder2-synthmath": "tokenized/dolmino/math/mathcoder2-synthmath-bc8dd2",
    "math/metamath-owmfilter": "tokenized/dolmino/math/metamath-owmfilter-fafa84",
    "math/tinyGSM-MIND": "tokenized/dolmino/math/tinyGSM-MIND-6c3016",
    "math/tulu_math": "tokenized/dolmino/math/tulu_math-414a4d",
    "pes2o": "tokenized/dolmino/pes2o-d22243",
    "stackexchange": "tokenized/dolmino/stackexchange-271a84",
    "wiki": "tokenized/dolmino/wiki-c31b74",
    "dolmino_dclm": "tokenized/dolmino/dclm-6c18eb",
}


def _get_dolmino_split_paths(split: str):
    """Helper to get file paths for a dolmino split."""
    patterns = DOLMINO_DATASETS[split]
    dolmino_split_input_base_path = _dolmino_base_dir / split
    return [dolmino_split_input_base_path / pattern for pattern in patterns]


def tokenize_dolmino_steps(*, base_path="tokenized/", tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """
    Generate tokenization steps for all Dolmino dataset splits.

    Args:
        base_path: Base directory for tokenized output
        tokenizer: Tokenizer name (defaults to llama3)

    Returns:
        Dictionary mapping split names (e.g., "dolmino/dclm") to tokenization steps
    """
    if tokenizer is None:
        tokenizer = _get_llama3_tokenizer()

    dolmino_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in DOLMINO_DATASETS:
        dolmino_split_output_path = os.path.join(base_path, "dolmino", split)
        dolmino_split_paths = _get_dolmino_split_paths(split)
        step = ExecutorStep(
            name=dolmino_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=dolmino_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
            pip_dependency_groups=["sentencepiece"],
        )

        if tokenizer == _get_llama3_tokenizer() and split in DOLMINO_LLAMA3_OVERRIDES:
            step = step.with_output_path(DOLMINO_LLAMA3_OVERRIDES[split])
        dolmino_steps[os.path.join("dolmino", split)] = step

    return dolmino_steps


def get_dolmino_step_llama3(split: str) -> ExecutorStep[TokenizeConfig]:
    """
    Get a specific dolmino split tokenized with llama3.

    Args:
        split: Split name (e.g., "dclm", "flan", "math/gsm8k")

    Returns:
        ExecutorStep for the tokenized split
    """
    assert (
        split in DOLMINO_DATASETS
    ), f"Split {split} not found in {DOLMINO_DATASETS}, \
        Check marin.experiments.pretraining_datasets.DOLMINO_DATASETS for which splits are supported."
    return tokenize_dolmino_steps()[f"dolmino/{split}"]


# Special combined math split
_all_dolmino_math_files = [
    path for split in DOLMINO_DATASETS if "math" in split for path in _get_dolmino_split_paths(split)
]

dolmino_math_tokenized_llama3 = ExecutorStep(
    name="tokenized/dolmino/all_math",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=_all_dolmino_math_files,
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(_get_llama3_tokenizer()),
        cache_options=CacheOptions(num_shard_groups=32),
    ),
    pip_dependency_groups=["sentencepiece"],
).with_output_path("tokenized/dolmino/all_math-9d507c")


# ----------------------------------------------------------------------------
# NEMOTRON CC tokenization
# ----------------------------------------------------------------------------

_nemotron_cc_path = output_path_of(nemotron_cc, "contrib/Nemotron/Nemotron-CC/data-jsonl/")

# The following dataset splits define file patterns for each split.
NEMOTRON_DATASETS = {
    "hq_actual": ["quality=high/kind=actual/**/*.jsonl.gz"],
    "hq_synth": ["quality=high/kind=synthetic/**/*.jsonl.gz"],
    "medium_high": ["quality=medium-high/**/*.jsonl.gz"],
    "medium": ["quality=medium/**/*.jsonl.gz"],
    "medium_low": ["quality=medium-low/**/*.jsonl.gz"],
    "low_actual": ["quality=low/kind=actual/**/*.jsonl.gz"],
    "low_synth": ["quality=low/kind=synthetic/**/*.jsonl.gz"],
}

# Weights for each split based on their size in TiB/GiB
# Converted GiB to TiB for consistency
NEMOTRON_WEIGHTS = {
    "nemotron_cc/hq_actual": 935.43 / 1024,  # 935.43 GiB
    "nemotron_cc/hq_synth": 2.72,  # 2.72 TiB
    "nemotron_cc/medium_high": 844.51 / 1024,  # 844.51 GiB
    "nemotron_cc/medium": 3.38,  # 3.38 TiB
    "nemotron_cc/medium_low": 1.54,  # 1.54 TiB
    "nemotron_cc/low_actual": 718.06 / 1024,  # 718.06 GiB
    "nemotron_cc/low_synth": 642.78 / 1024,  # 642.78 GiB
}

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
NEMOTRON_LLAMA3_OVERRIDES = {
    "hq_actual": "tokenized/nemotron_cc/hq_actual-5af4cc",
    "hq_synth": "tokenized/nemotron_cc/hq_synth-3525e2",
    "low_actual": "tokenized/nemotron_cc/low_actual-cb3f2c",
    "low_synth": "tokenized/nemotron_cc/low_synth-3c57b3",
    "medium": "tokenized/nemotron_cc/medium-d86506",
    "medium_high": "tokenized/nemotron_cc/medium_high-d21701",
    "medium_low": "tokenized/nemotron_cc/medium_low-0fdb07",
}


def _get_nemotron_split_paths(split: str):
    """Helper to get file paths for a nemotron split."""
    patterns = NEMOTRON_DATASETS[split]
    return [_nemotron_cc_path / pattern for pattern in patterns]


def tokenize_nemotron_steps(*, base_path="tokenized/", tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """
    Generate tokenization steps for all Nemotron CC dataset splits.

    Args:
        base_path: Base directory for tokenized output
        tokenizer: Tokenizer name (defaults to llama3)

    Returns:
        Dictionary mapping split names (e.g., "nemotron_cc/hq_actual") to tokenization steps
    """
    if tokenizer is None:
        tokenizer = _get_llama3_tokenizer()

    nemotron_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in NEMOTRON_DATASETS:
        nemotron_split_output_path = os.path.join(base_path, "nemotron_cc", split)
        nemotron_split_paths = _get_nemotron_split_paths(split)
        step = ExecutorStep(
            name=nemotron_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=nemotron_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                cache_options=CacheOptions(num_shard_groups=256),
            ),
            pip_dependency_groups=["sentencepiece"],
        )

        if tokenizer == _get_llama3_tokenizer() and split in NEMOTRON_LLAMA3_OVERRIDES:
            step = step.with_output_path(NEMOTRON_LLAMA3_OVERRIDES[split])

        nemotron_steps[os.path.join("nemotron_cc", split)] = step

    assert nemotron_steps.keys() == NEMOTRON_WEIGHTS.keys()
    return nemotron_steps


def get_nemotron_step(split: str) -> ExecutorStep[TokenizeConfig]:
    """
    Get a specific nemotron split tokenized with llama3.

    Args:
        split: Split name (e.g., "hq_actual", "medium", "low_synth")

    Returns:
        ExecutorStep for the tokenized split
    """
    assert (
        split in NEMOTRON_DATASETS
    ), f"Split {split} not found in {NEMOTRON_DATASETS}, \
        Check marin.experiments.pretraining_datasets.NEMOTRON_DATASETS for which splits are supported."
    return tokenize_nemotron_steps()[f"nemotron_cc/{split}"]


# ----------------------------------------------------------------------------
# Simple single-corpus tokenized datasets
# ----------------------------------------------------------------------------


def _tokenize_simple(
    name: str,
    raw_dataset: ExecutorStep,
    tokenizer: str | None = None,
    override_path: str | None = None,
    format: TextLmDatasetFormat = TextLmDatasetFormat(),
    cache_options: CacheOptions | None = None,
) -> ExecutorStep[TokenizeConfig]:
    """Helper to create a simple tokenized dataset."""
    if tokenizer is None:
        tokenizer = _get_llama3_tokenizer()

    config = TokenizeConfig(
        train_paths=[raw_dataset],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(tokenizer),
        format=format,
    )

    if cache_options is not None:
        config = TokenizeConfig(
            train_paths=config.train_paths,
            validation_paths=config.validation_paths,
            cache_path=config.cache_path,
            tokenizer=config.tokenizer,
            format=config.format,
            cache_options=cache_options,
        )

    step = ExecutorStep(
        name=os.path.join("tokenized", name),
        fn=tokenize,
        config=config,
        pip_dependency_groups=["sentencepiece"],
    )

    if override_path is not None:
        step = step.with_output_path(override_path)

    return step


# DCLM baseline
dclm_baseline_tokenized_llama3 = _tokenize_simple(
    "dclm_baseline",
    dclm_baseline,
    override_path="tokenized/dclm_baseline-0206f1/",
)

# StarCoder data (uses "content" as text key)
starcoderdata_tokenized_llama3 = _tokenize_simple(
    "starcoderdata",
    starcoderdata,
    format=TextLmDatasetFormat(text_key="content"),
    override_path="tokenized/starcoderdata-12f018/",
)

# ProofPile 2
proofpile_2_tokenized_llama3 = _tokenize_simple(
    "proofpile_2",
    proofpile_2,
    override_path="tokenized/proofpile_2-4a35c7/",
)

# SlimPajama 6B
slimpajama_6b_tokenized_llama3 = _tokenize_simple(
    "SlimPajama-6B",
    slimpajama_6b,
)

# FineWeb-Edu
fineweb_edu_tokenized_llama3 = _tokenize_simple(
    "fineweb-edu",
    fineweb_edu,
)
