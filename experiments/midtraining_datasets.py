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

from experiments.common_pile.tokenize_common_pile import stackv2_edu_filtered
from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import deferred, output, step, versioned
from marin.processing.tokenize import lm_mixture_data_config
from marin.transform.common_pile.filter_by_extension import FilterByMetadataExtensionConfig
from marin.transform.common_pile.filter_by_extension import (
    filter_dataset_by_metadata_extension as _filter_dataset_by_metadata_extension,
)
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig
from marin.transform.medical.lavita_to_dolma import convert_lavita_split_to_dolma as _convert_lavita_split_to_dolma

# Mark library functions as deferred
download_hf = deferred(_download_hf)
filter_dataset_by_metadata_extension = deferred(_filter_dataset_by_metadata_extension)
convert_lavita_split_to_dolma = deferred(_convert_lavita_split_to_dolma)

finemath_commit_hash = "8f233cf"


@step(name="raw/finemath")
def finemath():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="HuggingFaceTB/finemath",
            revision=finemath_commit_hash,
            gcs_output_path=output(),
            wait_for_completion=True,
        )
    )


finemath_3_plus = finemath().cd("finemath-3plus")


def tokenize_finemath_3_plus():
    return default_tokenize(
        name="finemath_3_plus",
        dataset=finemath_3_plus,
        tokenizer=llama3_tokenizer,
    ).with_output_path("tokenized/finemath_3_plus-a26b0f/")


STACKV2_EDU_PYTHON_EXTENSIONS = (".py", ".pyw", ".pyi")


@step(name="documents/common_pile/stackv2_edu_filtered_python")
def stackv2_edu_filtered_python():
    return filter_dataset_by_metadata_extension(
        FilterByMetadataExtensionConfig(
            input_path=stackv2_edu_filtered,
            output_path=output(),
            allowed_extensions=STACKV2_EDU_PYTHON_EXTENSIONS,
            input_glob="stack-edu-*.json.gz",
        )
    )


def tokenize_stackv2_edu_filtered_python():
    return default_tokenize(
        name="common_pile_stackv2_edu_filtered_python",
        dataset=stackv2_edu_filtered_python(),
        tokenizer=llama3_tokenizer,
    )


# Define MegaMath dataset source
megamath_source = default_download(
    name="raw/llm360/megamath",
    hf_dataset_id="llm360/MegaMath",
    revision=versioned("3cbc64616594d6bc8759abaa0b2a71858f880f0d"),
    override_output_path="raw/llm360/megamath",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# Megamath is partitioned into 6 sources. We expose each of them as a separate step.
megamath_split_paths = {
    # Code just seems to be metadata, not actual code files.
    # "megamath/code": megamath_source / "megamath-code/*.parquet",
    "megamath/qa": megamath_source / "megamath-qa/**/*.parquet",
    "megamath/text_code_block": megamath_source / "megamath-text-code-block/*.parquet",
    "megamath/translated_code": megamath_source / "megamath-translated-code/*.parquet",
    "megamath/web_pro": megamath_source / "megamath-web-pro/*.parquet",
    "megamath/web": megamath_source / "megamath-web/*/*.parquet",
}


def tokenize_megamath():
    return {
        name: default_tokenize(
            name=name,
            dataset=path,
            tokenizer=llama3_tokenizer,
        )
        for name, path in megamath_split_paths.items()
    }


# source: https://huggingface.co/datasets/LLM360/MegaMath#detailed-statistics
# in teratokens
megamath_token_counts = {
    # "Real"
    "megamath/web": 0.2639,  # 263.9B
    "megamath/web_pro": 0.0151,  # 15.1B
    # Synthetic
    "megamath/text_code_block": 0.0503,  # 50.3B
    "megamath/translated_code": 0.0072,  # 7.2B
    "megamath/qa": 0.0070,  # 7.0B
}


def megamath_mixture():
    return lm_mixture_data_config(
        components=tokenize_megamath(),
        weights=megamath_token_counts,
    )


def megamath_real_only():
    tokenized = tokenize_megamath()
    return lm_mixture_data_config(
        components={
            "megamath/web": tokenized["megamath/web"],
            "megamath/web_pro": tokenized["megamath/web_pro"],
        },
        weights={
            "megamath/web": megamath_token_counts["megamath/web"],
            "megamath/web_pro": megamath_token_counts["megamath/web_pro"],
        },
        permutation_type="feistel",
    )


pile_pubmed_abstracts_validation = default_download(
    name="raw/pile_pubmed_abstracts",
    hf_dataset_id="suolyer/pile_pubmed-abstracts",
    revision="139fdbf",
    override_output_path="raw/pile_pubmed_abstracts",
    hf_urls_glob=["val.json"],
)

pile_pubmed_central_validation = default_download(
    name="raw/pile_pubmed_central",
    hf_dataset_id="suolyer/pile_pubmed-central",
    revision="783dc95",
    override_output_path="raw/pile_pubmed_central",
    hf_urls_glob=["val.json"],
)


def tokenize_pile_pubmed_central_validation():
    return default_tokenize(
        name="pile_pubmed_central",
        dataset=pile_pubmed_central_validation,
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


def tokenize_pile_pubmed_abstracts_validation():
    return default_tokenize(
        name="pile_pubmed_abstracts",
        dataset=pile_pubmed_abstracts_validation,
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


# Medical QA datasets
lavita_medical_qa_datasets = default_download(
    name="raw/lavita_medical_qa",
    hf_dataset_id="lavita/medical-qa-datasets",
    revision="59d48e2",
    override_output_path="raw/lavita_medical_qa",
)


@step(name="documents/lavita_pubmed")
def lavita_pubmed():
    return convert_lavita_split_to_dolma(
        LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets,
            output_path=output(),
            subset="pubmed-qa",
            split="train",
        )
    )


@step(name="documents/lavita_medmcqa")
def lavita_medmcqa():
    return convert_lavita_split_to_dolma(
        LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets,
            output_path=output(),
            subset="medmcqa",
            split="train",
        )
    )


@step(name="documents/lavita_allprocessed")
def lavita_allprocessed():
    return convert_lavita_split_to_dolma(
        LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets,
            output_path=output(),
            subset="all-processed",
            split="train",
        )
    )


@step(name="documents/lavita_pubmed_validation")
def lavita_pubmed_validation():
    return convert_lavita_split_to_dolma(
        LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets,
            output_path=output(),
            subset="pubmed-qa",
            split="validation",
        )
    )


@step(name="documents/lavita_medmcqa_validation")
def lavita_medmcqa_validation():
    return convert_lavita_split_to_dolma(
        LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets,
            output_path=output(),
            subset="medmcqa",
            split="validation",
        )
    )


def tokenize_lavita_allprocessed():
    return default_tokenize(
        "tokenized/lavita_allprocessed",
        lavita_allprocessed(),
        tokenizer=llama3_tokenizer,
    )


def tokenize_lavita_medmcqa():
    return default_tokenize(
        "tokenized/lavita_medmcqa",
        lavita_medmcqa(),
        tokenizer=llama3_tokenizer,
    )


def tokenize_lavita_pubmedqa():
    return default_tokenize(
        "tokenized/lavita_pubmedqa",
        lavita_pubmed(),
        tokenizer=llama3_tokenizer,
    )


def tokenize_lavita_pubmedqa_validation():
    return default_tokenize(
        "tokenized/lavita_pubmedqa_validation",
        lavita_pubmed_validation(),
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


def tokenize_lavita_medmcqa_validation():
    return default_tokenize(
        "tokenized/lavita_medmcqa_validation",
        lavita_medmcqa_validation(),
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


# Backward compatibility aliases for files that import the module-level variables
# These are deprecated and should be replaced with function calls inside @step functions
class _LazyTokenized:
    """Provides backward-compatible access to tokenized datasets.

    This class allows importing module-level variables like `finemath_3_plus_tokenized`
    while deferring the actual step creation until the variable is used.
    """

    def __init__(self, factory):
        self._factory = factory
        self._cached = None

    def __call__(self):
        if self._cached is None:
            self._cached = self._factory()
        return self._cached

    # Make it behave like an ExecutorStep when accessed directly
    def __getattr__(self, name):
        return getattr(self(), name)


# These are for backward compatibility - new code should call the functions directly
finemath_3_plus_tokenized = _LazyTokenized(tokenize_finemath_3_plus)
stackv2_edu_filtered_python_tokenized = _LazyTokenized(tokenize_stackv2_edu_filtered_python)
megamath_tokenized = _LazyTokenized(tokenize_megamath)
pile_pubmed_central_validation_tokenized = _LazyTokenized(tokenize_pile_pubmed_central_validation)
pile_pubmed_abstracts_validation_tokenized = _LazyTokenized(tokenize_pile_pubmed_abstracts_validation)
lavita_allprocessed_tokenized = _LazyTokenized(tokenize_lavita_allprocessed)
lavita_medmcqa_tokenized = _LazyTokenized(tokenize_lavita_medmcqa)
lavita_pubmedqa_tokenized = _LazyTokenized(tokenize_lavita_pubmedqa)
lavita_pubmedqa_validation_tokenized = _LazyTokenized(tokenize_lavita_pubmedqa_validation)
lavita_medmcqa_validation_tokenized = _LazyTokenized(tokenize_lavita_medmcqa_validation)
