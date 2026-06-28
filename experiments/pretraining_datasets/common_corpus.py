# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Corpus dataset definitions and tokenization."""

from fray.types import ResourceConfig
from marin.datakit.download.common_corpus import HF_DATASET_ID, filter_common_corpus
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import Dataset
from marin.experiment.data import derived, hf_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer

# Pinned HF revision and URL glob for the raw download.
_HF_REVISION = "b78a5c1"
_URLS_GLOB = ["common_corpus_*/*.parquet"]


def _run_filter(cfg: dict) -> None:
    filter_common_corpus(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        id_field=cfg["id_field"],
    )


def common_corpus_datasets(*, tokenizer: str = marin_tokenizer) -> Dataset:
    """Tokenize the filtered Common Corpus (English, open types)."""
    dl = hf_download("raw/common_corpus", hf_id=HF_DATASET_ID, revision=_HF_REVISION, urls_glob=_URLS_GLOB)
    filtered = derived(
        "raw/common_corpus_english_filtered",
        fn=_run_filter,
        build_config=lambda ctx: {"input_path": ctx.path(dl), "output_path": ctx.out},
        deps=(dl,),
        kind=Dataset,
    )
    norm = derived(
        "normalized/common_corpus_english_filtered",
        fn=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.path(filtered),
            "output_path": ctx.out,
            "id_field": "identifier",
        },
        deps=(filtered,),
        kind=Dataset,
    )
    return tokenized(
        "common_corpus_english",
        tokenizer=tokenizer,
        raw=norm,
        glob="outputs/main/*.parquet",
        resources=ResourceConfig(ram="40g", disk="5g"),
    )
