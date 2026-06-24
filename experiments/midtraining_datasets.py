# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution import versioned
from marin.execution.types import ExecutorStep, InputName, this_output_path
from marin.processing.tokenize import lm_mixture_data_config
from marin.transform.common_pile.filter_by_extension import (
    FilterByMetadataExtensionConfig,
    filter_dataset_by_metadata_extension,
)
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma

from experiments.common_pile.tokenize_common_pile import stackv2_edu_filtered
from experiments.llama import llama3_tokenizer
from experiments.tokenization import default_download, default_tokenize

finemath_commit_hash = "8f233cf"


def finemath() -> ExecutorStep:
    return ExecutorStep(
        name="raw/finemath",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="HuggingFaceTB/finemath",
            revision=finemath_commit_hash,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    )


def finemath_3_plus() -> ExecutorStep:
    return finemath().cd("finemath-3plus")


def finemath_3_plus_tokenized() -> ExecutorStep:
    return default_tokenize(
        name="finemath_3_plus",
        dataset=finemath_3_plus(),
        tokenizer=llama3_tokenizer,
    ).with_output_path("tokenized/finemath_3_plus-a26b0f/")


STACKV2_EDU_PYTHON_EXTENSIONS = (".py", ".pyw", ".pyi")


def stackv2_edu_filtered_python() -> ExecutorStep:
    return ExecutorStep(
        name="documents/common_pile/stackv2_edu_filtered_python",
        fn=filter_dataset_by_metadata_extension,
        config=FilterByMetadataExtensionConfig(
            input_path=stackv2_edu_filtered(),
            output_path=this_output_path(),
            allowed_extensions=STACKV2_EDU_PYTHON_EXTENSIONS,
            input_glob="stack-edu-*.json.gz",
        ),
    )


def stackv2_edu_filtered_python_tokenized() -> ExecutorStep:
    return default_tokenize(
        name="common_pile_stackv2_edu_filtered_python",
        dataset=stackv2_edu_filtered_python(),
        tokenizer=llama3_tokenizer,
    )


def megamath_source() -> ExecutorStep:
    """MegaMath dataset source download."""
    return default_download(
        name="raw/llm360/megamath",
        hf_dataset_id="llm360/MegaMath",
        revision=versioned("3cbc64616594d6bc8759abaa0b2a71858f880f0d"),
        override_output_path="raw/llm360/megamath",
        hf_urls_glob=["**/*.parquet", "*.md"],
    )


def megamath_split_paths() -> dict[str, InputName]:
    """Megamath is partitioned into 6 sources. We expose each of them as a separate step."""
    source = megamath_source()
    return {
        # Code just seems to be metadata, not actual code files.
        # "megamath/code": source / "megamath-code/*.parquet",
        "megamath/qa": source / "megamath-qa/**/*.parquet",
        "megamath/text_code_block": source / "megamath-text-code-block/*.parquet",
        "megamath/translated_code": source / "megamath-translated-code/*.parquet",
        "megamath/web_pro": source / "megamath-web-pro/*.parquet",
        "megamath/web": source / "megamath-web/*/*.parquet",
    }


def megamath_tokenized() -> dict[str, ExecutorStep]:
    return {
        name: default_tokenize(
            name=name,
            dataset=path,
            tokenizer=llama3_tokenizer,
        )
        for name, path in megamath_split_paths().items()
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
        components=megamath_tokenized(),
        weights=megamath_token_counts,
    )


def megamath_real_only():
    tokenized = megamath_tokenized()
    return lm_mixture_data_config(
        components={
            "megamath/web": tokenized["megamath/web"],
            "megamath/web_pro": tokenized["megamath/web_pro"],
        },
        weights={
            "megamath/web": megamath_token_counts["megamath/web"],
            "megamath/web_pro": megamath_token_counts["megamath/web_pro"],
        },
    )


def pile_pubmed_abstracts_validation() -> ExecutorStep:
    return default_download(
        name="raw/pile_pubmed_abstracts",
        hf_dataset_id="suolyer/pile_pubmed-abstracts",
        revision="139fdbf",
        override_output_path="raw/pile_pubmed_abstracts",
        hf_urls_glob=["val.json"],
    )


def pile_pubmed_central_validation() -> ExecutorStep:
    return default_download(
        name="raw/pile_pubmed_central",
        hf_dataset_id="suolyer/pile_pubmed-central",
        revision="783dc95",
        override_output_path="raw/pile_pubmed_central",
        hf_urls_glob=["val.json"],
    )


def pile_pubmed_central_validation_tokenized() -> ExecutorStep:
    return default_tokenize(
        name="pile_pubmed_central",
        dataset=pile_pubmed_central_validation(),
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


def pile_pubmed_abstracts_validation_tokenized() -> ExecutorStep:
    return default_tokenize(
        name="pile_pubmed_abstracts",
        dataset=pile_pubmed_abstracts_validation(),
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


# Medical QA datasets
def lavita_medical_qa_datasets() -> ExecutorStep:
    return default_download(
        name="raw/lavita_medical_qa",
        hf_dataset_id="lavita/medical-qa-datasets",
        revision="59d48e2",
        override_output_path="raw/lavita_medical_qa",
    )


def lavita_pubmed() -> ExecutorStep:
    return ExecutorStep(
        name="documents/lavita_pubmed",
        fn=convert_lavita_split_to_dolma,
        config=LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets(), output_path=this_output_path(), subset="pubmed-qa", split="train"
        ),
    )


def lavita_medmcqa() -> ExecutorStep:
    return ExecutorStep(
        name="documents/lavita_medmcqa",
        fn=convert_lavita_split_to_dolma,
        config=LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets(), output_path=this_output_path(), subset="medmcqa", split="train"
        ),
    )


def lavita_allprocessed() -> ExecutorStep:
    return ExecutorStep(
        name="documents/lavita_allprocessed",
        fn=convert_lavita_split_to_dolma,
        config=LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets(),
            output_path=this_output_path(),
            subset="all-processed",
            split="train",
        ),
    )


def lavita_pubmed_validation() -> ExecutorStep:
    return ExecutorStep(
        name="documents/lavita_pubmed_validation",
        fn=convert_lavita_split_to_dolma,
        config=LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets(),
            output_path=this_output_path(),
            subset="pubmed-qa",
            split="validation",
        ),
    )


def lavita_medmcqa_validation() -> ExecutorStep:
    return ExecutorStep(
        name="documents/lavita_medmcqa_validation",
        fn=convert_lavita_split_to_dolma,
        config=LavitaToDolmaConfig(
            input_path=lavita_medical_qa_datasets(),
            output_path=this_output_path(),
            subset="medmcqa",
            split="validation",
        ),
    )


def lavita_allprocessed_tokenized() -> ExecutorStep:
    return default_tokenize(
        "tokenized/lavita_allprocessed",
        lavita_allprocessed(),
        tokenizer=llama3_tokenizer,
    )


def lavita_medmcqa_tokenized() -> ExecutorStep:
    return default_tokenize(
        "tokenized/lavita_medmcqa",
        lavita_medmcqa(),
        tokenizer=llama3_tokenizer,
    )


def lavita_pubmedqa_tokenized() -> ExecutorStep:
    return default_tokenize(
        "tokenized/lavita_pubmedqa",
        lavita_pubmed(),
        tokenizer=llama3_tokenizer,
    )


def lavita_pubmedqa_validation_tokenized() -> ExecutorStep:
    return default_tokenize(
        "tokenized/lavita_pubmedqa_validation",
        lavita_pubmed_validation(),
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )


def lavita_medmcqa_validation_tokenized() -> ExecutorStep:
    return default_tokenize(
        "tokenized/lavita_medmcqa_validation",
        lavita_medmcqa_validation(),
        tokenizer=llama3_tokenizer,
        is_validation=True,
    )
