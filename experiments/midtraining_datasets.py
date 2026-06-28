# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy dataset handles for mid-training: math, code, medical QA, and pile validation."""

from marin.execution.artifact import Dataset
from marin.execution.lazy import Lazy, derived
from marin.experiment.data import hf_download, tokenized
from marin.transform.common_pile.filter_by_extension import (
    FilterByMetadataExtensionConfig,
    filter_dataset_by_metadata_extension,
)
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma

from experiments.common_pile.tokenize_common_pile import stackv2_edu_filtered_download
from experiments.llama import llama3_tokenizer

STACKV2_EDU_PYTHON_EXTENSIONS = (".py", ".pyw", ".pyi")

# Source: https://huggingface.co/datasets/LLM360/MegaMath#detailed-statistics
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


def finemath_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """Finemath-3plus tokenized dataset handle."""
    finemath_download = hf_download(
        "raw/finemath",
        hf_id="HuggingFaceTB/finemath",
        revision="8f233cf",
    )
    return {
        "finemath_3_plus": tokenized(
            "finemath_3_plus",
            tokenizer=tokenizer,
            raw=finemath_download,
            glob="finemath-3plus/**/*.parquet",
            pin="tokenized/finemath_3_plus-a26b0f/",
        ),
    }


def stackv2_edu_python_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """Tokenized Python-filtered stackv2_edu Common Pile split."""
    stackv2_edu_dl = stackv2_edu_filtered_download()
    python_filter = derived(
        "documents/common_pile/stackv2_edu_filtered_python",
        fn=filter_dataset_by_metadata_extension,
        build_config=lambda ctx: FilterByMetadataExtensionConfig(
            input_path=ctx.path(stackv2_edu_dl),
            output_path=ctx.out,
            allowed_extensions=STACKV2_EDU_PYTHON_EXTENSIONS,
            input_glob="stack-edu-*.json.gz",
        ),
        deps=(stackv2_edu_dl,),
        kind=Dataset,
    )
    return {
        "common_pile_stackv2_edu_filtered_python": tokenized(
            "common_pile_stackv2_edu_filtered_python",
            tokenizer=tokenizer,
            raw=python_filter,
            glob="**/*.jsonl.gz",
        ),
    }


def megamath_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """Tokenized MegaMath splits, one handle per source partition."""
    megamath_download = hf_download(
        "raw/llm360/megamath",
        hf_id="llm360/MegaMath",
        revision="3cbc64616594d6bc8759abaa0b2a71858f880f0d",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/llm360/megamath",
    )
    _split_globs = {
        "megamath/qa": "megamath-qa/**/*.parquet",
        "megamath/text_code_block": "megamath-text-code-block/*.parquet",
        "megamath/translated_code": "megamath-translated-code/*.parquet",
        "megamath/web_pro": "megamath-web-pro/*.parquet",
        "megamath/web": "megamath-web/*/*.parquet",
    }
    return {
        name: tokenized(name, tokenizer=tokenizer, raw=megamath_download, glob=glob)
        for name, glob in _split_globs.items()
    }


def pile_validation_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """Pile PubMed validation tokenized dataset handles."""
    pile_pubmed_abstracts_download = hf_download(
        "raw/pile_pubmed_abstracts",
        hf_id="suolyer/pile_pubmed-abstracts",
        revision="139fdbf",
        urls_glob=["val.json"],
        pin="raw/pile_pubmed_abstracts",
    )
    pile_pubmed_central_download = hf_download(
        "raw/pile_pubmed_central",
        hf_id="suolyer/pile_pubmed-central",
        revision="783dc95",
        urls_glob=["val.json"],
        pin="raw/pile_pubmed_central",
    )
    return {
        "pile_pubmed_central": tokenized(
            "pile_pubmed_central",
            tokenizer=tokenizer,
            raw=pile_pubmed_central_download,
            glob="val.json",
            validation=True,
        ),
        "pile_pubmed_abstracts": tokenized(
            "pile_pubmed_abstracts",
            tokenizer=tokenizer,
            raw=pile_pubmed_abstracts_download,
            glob="val.json",
            validation=True,
        ),
    }


def lavita_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """Medical QA (lavita) tokenized dataset handles."""
    lavita_download = hf_download(
        "raw/lavita_medical_qa",
        hf_id="lavita/medical-qa-datasets",
        revision="59d48e2",
        pin="raw/lavita_medical_qa",
    )

    def _lavita_derived(name: str, subset: str, split: str) -> Lazy[Dataset]:
        return derived(
            name,
            fn=convert_lavita_split_to_dolma,
            build_config=lambda ctx: LavitaToDolmaConfig(
                input_path=ctx.path(lavita_download), output_path=ctx.out, subset=subset, split=split
            ),
            deps=(lavita_download,),
            kind=Dataset,
        )

    lavita_pubmed = _lavita_derived("documents/lavita_pubmed", "pubmed-qa", "train")
    lavita_medmcqa = _lavita_derived("documents/lavita_medmcqa", "medmcqa", "train")
    lavita_allprocessed = _lavita_derived("documents/lavita_allprocessed", "all-processed", "train")
    lavita_pubmed_validation = _lavita_derived("documents/lavita_pubmed_validation", "pubmed-qa", "validation")
    lavita_medmcqa_validation = _lavita_derived("documents/lavita_medmcqa_validation", "medmcqa", "validation")

    return {
        "lavita_allprocessed": tokenized(
            "lavita_allprocessed", tokenizer=tokenizer, raw=lavita_allprocessed, glob="**/*.parquet"
        ),
        "lavita_medmcqa": tokenized("lavita_medmcqa", tokenizer=tokenizer, raw=lavita_medmcqa, glob="**/*.parquet"),
        "lavita_pubmedqa": tokenized("lavita_pubmedqa", tokenizer=tokenizer, raw=lavita_pubmed, glob="**/*.parquet"),
        "lavita_pubmedqa_validation": tokenized(
            "lavita_pubmedqa_validation",
            tokenizer=tokenizer,
            raw=lavita_pubmed_validation,
            glob="**/*.parquet",
            validation=True,
        ),
        "lavita_medmcqa_validation": tokenized(
            "lavita_medmcqa_validation",
            tokenizer=tokenizer,
            raw=lavita_medmcqa_validation,
            glob="**/*.parquet",
            validation=True,
        ),
    }
