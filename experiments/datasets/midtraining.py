# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ArtifactStep dataset handles for mid-training: math, code, medical QA, and pile validation."""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.transform.common_pile.filter_by_extension import (
    FilterByMetadataExtensionConfig,
    filter_dataset_by_metadata_extension,
)
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma

from experiments.datasets.common_pile import stackv2_edu_filtered_download
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


def finemath_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Finemath-3plus tokenized dataset handle."""
    finemath_download = hf_download(
        "raw/finemath",
        hf_id="HuggingFaceTB/finemath",
        revision="8f233cf",
        version="2026.06.28",
    )
    return {
        "finemath_3_plus": tokenized(
            "finemath_3_plus",
            tokenizer=tokenizer,
            raw=finemath_download,
            glob="finemath-3plus/**/*.parquet",
            pin="tokenized/finemath_3_plus-a26b0f/",
            version="2026.06.28",
        ),
    }


def stackv2_edu_python_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Tokenized Python-filtered stackv2_edu Common Pile split."""
    stackv2_edu_dl = stackv2_edu_filtered_download()
    python_filter = ArtifactStep(
        name="documents/common_pile/stackv2_edu_filtered_python",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=filter_dataset_by_metadata_extension,
        build_config=lambda ctx: FilterByMetadataExtensionConfig(
            input_path=ctx.artifact_path(stackv2_edu_dl),
            output_path=ctx.output_path,
            allowed_extensions=STACKV2_EDU_PYTHON_EXTENSIONS,
            input_glob="stack-edu-*.json.gz",
        ),
        deps=(stackv2_edu_dl,),
    )
    return {
        "common_pile_stackv2_edu_filtered_python": tokenized(
            "common_pile_stackv2_edu_filtered_python",
            tokenizer=tokenizer,
            raw=python_filter,
            glob="**/*.jsonl.gz",
            version="2026.06.28",
        ),
    }


def _megamath_download() -> ArtifactStep[TokenizedCache]:
    """Pinned raw download of the full MegaMath HF repo (all partitions share it)."""
    return hf_download(
        "raw/llm360/megamath",
        hf_id="llm360/MegaMath",
        revision="3cbc64616594d6bc8759abaa0b2a71858f880f0d",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/llm360/megamath",
        version="2026.06.28",
    )


def megamath_slice(
    name: str, glob: str, pin: str | None = None, *, tokenizer: str = llama3_tokenizer
) -> ArtifactStep[TokenizedCache]:
    """One MegaMath partition (the parquet under ``glob`` of the pinned download) as a tokenized handle.

    With the llama3 tokenizer and ``pin`` set, resolves to that marin-executor cache instead of
    re-tokenizing; any other tokenizer tokenizes fresh.
    """
    return tokenized(
        name,
        tokenizer=tokenizer,
        raw=_megamath_download(),
        glob=glob,
        pin=pin if tokenizer == llama3_tokenizer else None,
        version="2026.06.28",
    )


# split -> (source glob under the pinned megamath download, existing llama3 tokenized cache).
MEGAMATH_SLICES: dict[str, tuple[str, str]] = {
    "megamath/qa": ("megamath-qa/**/*.parquet", "tokenized/megamath/qa-851f70"),
    "megamath/text_code_block": ("megamath-text-code-block/*.parquet", "tokenized/megamath/text_code_block-4df313"),
    "megamath/translated_code": ("megamath-translated-code/*.parquet", "tokenized/megamath/translated_code-358b52"),
    "megamath/web_pro": ("megamath-web-pro/*.parquet", "tokenized/megamath/web_pro-d2b548"),
    "megamath/web": ("megamath-web/*/*.parquet", "tokenized/megamath/web-2e2440"),
}


def megamath_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Tokenized MegaMath splits, one handle per source partition."""
    return {name: megamath_slice(name, glob, pin, tokenizer=tokenizer) for name, (glob, pin) in MEGAMATH_SLICES.items()}


def pile_validation_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Pile PubMed validation tokenized dataset handles."""
    pile_pubmed_abstracts_download = hf_download(
        "raw/pile_pubmed_abstracts",
        hf_id="suolyer/pile_pubmed-abstracts",
        revision="139fdbf",
        urls_glob=["val.json"],
        pin="raw/pile_pubmed_abstracts",
        version="2026.06.28",
    )
    pile_pubmed_central_download = hf_download(
        "raw/pile_pubmed_central",
        hf_id="suolyer/pile_pubmed-central",
        revision="783dc95",
        urls_glob=["val.json"],
        pin="raw/pile_pubmed_central",
        version="2026.06.28",
    )
    return {
        "pile_pubmed_central": tokenized(
            "pile_pubmed_central",
            tokenizer=tokenizer,
            raw=pile_pubmed_central_download,
            glob="val.json",
            validation=True,
            version="2026.06.28",
        ),
        "pile_pubmed_abstracts": tokenized(
            "pile_pubmed_abstracts",
            tokenizer=tokenizer,
            raw=pile_pubmed_abstracts_download,
            glob="val.json",
            validation=True,
            version="2026.06.28",
        ),
    }


def lavita_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Medical QA (lavita) tokenized dataset handles."""
    lavita_download = hf_download(
        "raw/lavita_medical_qa",
        hf_id="lavita/medical-qa-datasets",
        revision="59d48e2",
        pin="raw/lavita_medical_qa",
        version="2026.06.28",
    )

    def _lavita_derived(name: str, subset: str, split: str) -> ArtifactStep[TokenizedCache]:
        return ArtifactStep(
            name=name,
            version="2026.06.28",
            artifact_type=TokenizedCache,
            run=convert_lavita_split_to_dolma,
            build_config=lambda ctx: LavitaToDolmaConfig(
                input_path=ctx.artifact_path(lavita_download), output_path=ctx.output_path, subset=subset, split=split
            ),
            deps=(lavita_download,),
        )

    lavita_pubmed = _lavita_derived("documents/lavita_pubmed", "pubmed-qa", "train")
    lavita_medmcqa = _lavita_derived("documents/lavita_medmcqa", "medmcqa", "train")
    lavita_allprocessed = _lavita_derived("documents/lavita_allprocessed", "all-processed", "train")
    lavita_pubmed_validation = _lavita_derived("documents/lavita_pubmed_validation", "pubmed-qa", "validation")
    lavita_medmcqa_validation = _lavita_derived("documents/lavita_medmcqa_validation", "medmcqa", "validation")

    return {
        "lavita_allprocessed": tokenized(
            "lavita_allprocessed",
            tokenizer=tokenizer,
            raw=lavita_allprocessed,
            glob="**/*.parquet",
            version="2026.06.28",
        ),
        "lavita_medmcqa": tokenized(
            "lavita_medmcqa",
            tokenizer=tokenizer,
            raw=lavita_medmcqa,
            glob="**/*.parquet",
            version="2026.06.28",
        ),
        "lavita_pubmedqa": tokenized(
            "lavita_pubmedqa",
            tokenizer=tokenizer,
            raw=lavita_pubmed,
            glob="**/*.parquet",
            version="2026.06.28",
        ),
        "lavita_pubmedqa_validation": tokenized(
            "lavita_pubmedqa_validation",
            tokenizer=tokenizer,
            raw=lavita_pubmed_validation,
            glob="**/*.parquet",
            validation=True,
            version="2026.06.28",
        ),
        "lavita_medmcqa_validation": tokenized(
            "lavita_medmcqa_validation",
            tokenizer=tokenizer,
            raw=lavita_medmcqa_validation,
            glob="**/*.parquet",
            validation=True,
            version="2026.06.28",
        ),
    }


if __name__ == "__main__":
    dataset_main(
        {
            **finemath_datasets(),
            **megamath_datasets(),
            **lavita_datasets(),
            **pile_validation_datasets(),
            **stackv2_edu_python_datasets(),
        }
    )
