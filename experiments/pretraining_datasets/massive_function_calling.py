# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MASSIVE → function-calling dataset tokenization."""

from marin.datakit.download.massive import (
    MASSIVE_TARBALL_URL,
    MASSIVE_VERSION,
    stage_massive_raw,
    transform_staged_massive,
)
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import raw_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def _run_stage(cfg: dict) -> None:
    stage_massive_raw(output_path=cfg["output_path"])


def _run_transform(cfg: dict) -> None:
    transform_staged_massive(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        text_field=cfg["text_field"],
        id_field=cfg["id_field"],
        file_extensions=tuple(cfg["file_extensions"]),
    )


def massive_function_calling_datasets(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """MASSIVE function-calling corpus as a tokenized Dataset handle."""
    staged = raw_download(
        "raw/massive",
        fn=_run_stage,
        build_config=lambda ctx: {
            "tarball_url": MASSIVE_TARBALL_URL,
            "version": MASSIVE_VERSION,
            "output_path": ctx.output_path,
        },
        version="2026.06.28",
    )
    transformed = ArtifactStep(
        name="processed/massive_function_calling",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_transform,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(staged),
            "output_path": ctx.output_path,
            "schema_version": "v1",
        },
        deps=(staged,),
    )
    norm = ArtifactStep(
        name="normalized/massive_function_calling",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(transformed),
            "output_path": ctx.output_path,
            "text_field": "text",
            "id_field": "id",
            "file_extensions": [".parquet"],
        },
        deps=(transformed,),
    )
    return tokenized(
        "massive_function_calling", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet", version="2026.06.28"
    )
