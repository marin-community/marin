# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PleIAs/SEC dataset as a lazy Dataset handle.

SEC annual report (10-K) filings from 1993-2024, sourced from EDGAR via
the EDGAR-Crawler toolkit. ~7.2 billion words across ~245K filings,
stored as one Parquet file per year.
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

SEC_HF_ID = "PleIAs/SEC"
SEC_REVISION = "b09d02e"


def _sec_raw() -> ArtifactStep[TokenizedCache]:
    return hf_download(
        "sec",
        hf_id=SEC_HF_ID,
        revision=SEC_REVISION,
        urls_glob=["*.parquet"],
        pin="sec",
        version="2026.09.18",
    )


def sec_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """SEC 10-K filings (1993-2024) as a tokenized Dataset handle."""
    raw = _sec_raw()
    return tokenized(
        "sec",
        tokenizer=tokenizer,
        raw=raw,
        glob="*.parquet",
        version="2026.09.18",
    )


if __name__ == "__main__":
    dataset_main({"sec": sec_dataset()})
