# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
FinePDFs and FinePDFs-edu dataset definitions for long-context experiments.

Uses the HuggingFace repos HuggingFaceFW/finepdfs and HuggingFaceFW/finepdfs-edu.
Step-functions return per-language tokenized Dataset handles.
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

FINEPDFS_HF_ID = "HuggingFaceFW/finepdfs"
FINEPDFS_REVISION = "89f5411"

FINEPDFS_EDU_HF_ID = "HuggingFaceFW/finepdfs-edu"
FINEPDFS_EDU_REVISION = "9cfabe2"

# ~206,917,202 docs * ~3,600 tokens/doc from manual audit ≈ 7.45e11 tokens
finepdfs_token_counts = {
    "eng_Latn": 7.45e11,
}

# ~140B tokens for English
finepdfs_edu_token_counts = {"eng_Latn": 140e9}


def _finepdfs_raw() -> ArtifactStep[TokenizedCache]:
    return hf_download(
        "finepdfs_eng_Latn",
        hf_id=FINEPDFS_HF_ID,
        revision=FINEPDFS_REVISION,
        urls_glob=["data/eng_Latn/*/*.parquet"],
        pin="finepdfs_eng_Latn",
        version="2026.06.28",
    )


def _finepdfs_edu_raw() -> ArtifactStep[TokenizedCache]:
    return hf_download(
        "finepdfs_edu_eng_Latn",
        hf_id=FINEPDFS_EDU_HF_ID,
        revision=FINEPDFS_EDU_REVISION,
        urls_glob=["data/eng_Latn/train/*.parquet"],
        pin="finepdfs_edu_eng_Latn",
        version="2026.06.28",
    )


def finepdfs_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Tokenized FinePDFs train data per language (English only for now)."""
    raw = _finepdfs_raw()
    return {
        "eng_Latn": tokenized(
            "finepdfs/eng_Latn",
            tokenizer=tokenizer,
            raw=raw,
            glob="data/eng_Latn/train/*.parquet",
            version="2026.06.28",
        )
    }


def finepdfs_validation_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Tokenized FinePDFs validation data per language."""
    raw = _finepdfs_raw()
    return {
        "eng_Latn": tokenized(
            "finepdfs/eng_Latn_val",
            tokenizer=tokenizer,
            raw=raw,
            glob="data/eng_Latn/test/*.parquet",
            validation=True,
            version="2026.06.28",
        )
    }


def finepdfs_edu_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Tokenized FinePDFs-edu train data per language."""
    raw = _finepdfs_edu_raw()
    return {
        "eng_Latn": tokenized(
            "finepdfs_edu/eng_Latn",
            tokenizer=tokenizer,
            raw=raw,
            glob="data/eng_Latn/train/*.parquet",
            version="2026.06.28",
        )
    }


if __name__ == "__main__":
    # Both families key on the language code (eng_Latn), so namespace them when merging.
    dataset_main(
        {
            **{f"finepdfs/{lang}": handle for lang, handle in finepdfs_datasets().items()},
            **{f"finepdfs_edu/{lang}": handle for lang, handle in finepdfs_edu_datasets().items()},
        }
    )
