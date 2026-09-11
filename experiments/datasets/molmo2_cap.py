# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-Cap dataset tokenization."""

from fray.types import ResourceConfig
from marin.datakit.download.molmo2_cap import HF_DATASET_ID, HF_REVISION, transform
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
    )


def molmo2_cap_dataset(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """Tokenize the normalized Molmo2-Cap captions."""
    dl = hf_download(
        "raw/molmo2-cap",
        hf_id=HF_DATASET_ID,
        revision=HF_REVISION,
        urls_glob=["data/train-*.parquet"],
        version="2026.06.28",
    )
    processed = ArtifactStep(
        name="processed/molmo2-cap",
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
        name="normalized/molmo2-cap",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.artifact_path(processed), "output_path": ctx.output_path},
        deps=(processed,),
    )
    return tokenized(
        "molmo2_cap",
        tokenizer=tokenizer,
        raw=norm,
        glob="outputs/main/*.parquet",
        resources=ResourceConfig(ram="16g", disk="5g"),
        version="2026.06.28",
    )


if __name__ == "__main__":
    dataset_main({"molmo2_cap": molmo2_cap_dataset()})
