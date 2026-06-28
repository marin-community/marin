# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SVG (nyuuzyou/svgfind) dataset download, normalization, and tokenization."""

from marin.datakit.download.svgfind import CC_GLOBS, HF_DATASET_ID, HF_REVISION, transform_svgfind_creativecommons
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import Dataset
from marin.experiment.data import derived, hf_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform_svgfind_creativecommons(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
    )


def svg_datasets(*, tokenizer: str = marin_tokenizer) -> Dataset:
    """SVG Creative Commons corpus as a tokenized Dataset handle."""
    dl = hf_download(
        "raw/svgfind-creativecommons",
        hf_id=HF_DATASET_ID,
        revision=HF_REVISION,
        urls_glob=list(CC_GLOBS),
    )
    processed = derived(
        "processed/svgfind-creativecommons",
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
        "normalized/svgfind-creativecommons",
        fn=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.path(processed), "output_path": ctx.out},
        deps=(processed,),
        kind=Dataset,
    )
    return tokenized("svg", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet")
