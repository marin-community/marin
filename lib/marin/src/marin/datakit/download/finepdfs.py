# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FinePDFs download definitions."""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

FINEPDFS_HF_ID = "HuggingFaceFW/finepdfs"
FINEPDFS_REVISION = "89f5411"
FINEPDFS_EXTRA_LANGS = [
    "spa_Latn",
    "deu_Latn",
    "fra_Latn",
    "rus_Cyrl",
    "jpn_Jpan",
    "ita_Latn",
    "por_Latn",
    "pol_Latn",
    "nld_Latn",
    "hun_Latn",
    "cmn_Hani",
    "ces_Latn",
    "arb_Arab",
    "ukr_Cyrl",
    "swe_Latn",
    "ron_Latn",
    "ind_Latn",
    "tha_Thai",
]


def download_finepdfs_step(language: str) -> StepSpec:
    return download_hf_step(
        f"raw/finepdfs_{language}",
        hf_dataset_id=FINEPDFS_HF_ID,
        revision=FINEPDFS_REVISION,
        hf_urls_glob=[f"data/{language}/*/*.parquet"],
    )
