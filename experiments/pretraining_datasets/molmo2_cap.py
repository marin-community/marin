# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-Cap dataset tokenization."""

from fray.types import ResourceConfig
from marin.datakit.download.molmo2_cap import HF_DATASET_ID, HF_REVISION, transform
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.artifact import Dataset
from marin.execution.lazy import Lazy, derived
from marin.experiment.data import hf_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
    )


def molmo2_cap_datasets(*, tokenizer: str = marin_tokenizer) -> Lazy[Dataset]:
    """Tokenize the normalized Molmo2-Cap captions."""
    dl = hf_download(
        "raw/molmo2-cap",
        hf_id=HF_DATASET_ID,
        revision=HF_REVISION,
        urls_glob=["data/train-*.parquet"],
    )
    processed = derived(
        "processed/molmo2-cap",
        fn=_run_transform,
        build_config=lambda ctx: {
            "input_path": ctx.path(dl),
            "output_path": ctx.out,
            "schema_version": "v1",
        },
        deps=(dl,),
        kind=Dataset,
    )
    norm = derived(
        "normalized/molmo2-cap",
        fn=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.path(processed), "output_path": ctx.out},
        deps=(processed,),
        kind=Dataset,
    )
    return tokenized(
        "molmo2_cap",
        tokenizer=tokenizer,
        raw=norm,
        glob="outputs/main/*.parquet",
        resources=ResourceConfig(ram="16g", disk="5g"),
    )
