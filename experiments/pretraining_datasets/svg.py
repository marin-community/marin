# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SVG (nyuuzyou/svgfind) dataset download, normalization, and tokenization."""

from marin.datakit.download.svgfind import CC_GLOBS, HF_DATASET_ID, HF_REVISION, transform_svgfind_creativecommons
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform_svgfind_creativecommons(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
    )


def svg_datasets(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """SVG Creative Commons corpus as a tokenized Dataset handle."""
    dl = hf_download(
        "raw/svgfind-creativecommons",
        hf_id=HF_DATASET_ID,
        revision=HF_REVISION,
        urls_glob=list(CC_GLOBS),
        version="2026.06.28",
    )
    processed = ArtifactStep(
        name="processed/svgfind-creativecommons",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_transform,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(dl),
            "output_path": ctx.output_path,
            "schema_version": "v1",
        },
        deps=(dl,),
    )
    norm = ArtifactStep(
        name="normalized/svgfind-creativecommons",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.artifact_path(processed), "output_path": ctx.output_path},
        deps=(processed,),
    )
    return tokenized("svg", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet", version="2026.06.28")
