# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Corpus dataset definitions and tokenization."""

from fray.types import ResourceConfig
from marin.datakit.download.common_corpus import HF_DATASET_ID, filter_common_corpus
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

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


def common_corpus_datasets(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """Tokenize the filtered Common Corpus (English, open types)."""
    dl = hf_download(
        "raw/common_corpus", hf_id=HF_DATASET_ID, revision=_HF_REVISION, urls_glob=_URLS_GLOB, version="2026.06.28"
    )
    filtered = ArtifactStep(
        name="raw/common_corpus_english_filtered",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_filter,
        build_config=lambda ctx: {"input_path": ctx.artifact_path(dl), "output_path": ctx.output_path},
        deps=(dl,),
    )
    norm = ArtifactStep(
        name="normalized/common_corpus_english_filtered",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(filtered),
            "output_path": ctx.output_path,
            "id_field": "identifier",
        },
        deps=(filtered,),
    )
    return tokenized(
        "common_corpus_english",
        tokenizer=tokenizer,
        raw=norm,
        glob="outputs/main/*.parquet",
        resources=ResourceConfig(ram="40g", disk="5g"),
        version="2026.06.28",
    )
