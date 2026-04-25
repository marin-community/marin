# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFaceFW/finepdfs download + normalize helpers.

The upstream is a single HF repo with 19 language-scoped subsets; Marin stages
each language under its own ``raw/finepdfs_<lang>_<sha>`` directory (each dir
already holds only that language's shards). One chain per language, returned
by :func:`finepdfs_normalize_steps`.
"""

from __future__ import annotations

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

FINEPDFS_HF_ID = "HuggingFaceFW/finepdfs"
FINEPDFS_REVISION = "89f5411"

# Language subset -> content-hash suffix on the staged directory. The staged
# layout is ``raw/finepdfs_<lang>_<sha>/`` (no nested subdir; the directory
# content IS the per-language parquet).
FINEPDFS_LANGUAGES: dict[str, str] = {
    "eng_Latn": "1a6e7def",
    "arb_Arab": "d45e1edc",
    "ces_Latn": "b3371d5c",
    "cmn_Hani": "07be0dc4",
    "deu_Latn": "ce5aaacd",
    "fra_Latn": "35a75de8",
    "hun_Latn": "e906b5de",
    "ind_Latn": "8ba9e288",
    "ita_Latn": "c8fa2fa7",
    "jpn_Jpan": "7b65dbec",
    "nld_Latn": "a60bc417",
    "pol_Latn": "2558940c",
    "por_Latn": "cdf5ff50",
    "ron_Latn": "c41b1d50",
    "rus_Cyrl": "6e14b64d",
    "spa_Latn": "89be7172",
    "swe_Latn": "eac6cc36",
    "tha_Thai": "2921d58a",
    "ukr_Cyrl": "be1fb148",
}


def finepdfs_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every FinePDFs language subset.

    Keyed by the registry name convention: ``"finepdfs"`` for English (the
    default subset; no suffix) and ``"finepdfs/<lang>"`` for the 18 multilingual
    subsets.
    """
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for lang, sha in FINEPDFS_LANGUAGES.items():
        marin_name = "finepdfs" if lang == "eng_Latn" else f"finepdfs/{lang}"
        chains[marin_name] = hf_normalize_steps(
            marin_name=marin_name,
            hf_dataset_id=FINEPDFS_HF_ID,
            revision=FINEPDFS_REVISION,
            staged_path=f"raw/finepdfs_{lang}_{sha}",
        )
    return chains
