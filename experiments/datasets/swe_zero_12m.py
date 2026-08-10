# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO-12M-trajectories pre-training dataset as a lazy Dataset handle.

12.29M execution-free agentic-coding trajectories from
AlienKevin/SWE-ZERO-12M-trajectories, rendered via the mini-swe-agent v1 format,
normalized, and tokenized.
"""

from marin.datakit.download.swe_zero_12m import HF_DATASET_ID, HF_REVISION, transform
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(input_path=cfg["input_path"], output_path=cfg["output_path"])


def swe_zero_12m_dataset(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """SWE-ZERO-12M-trajectories as a tokenized Dataset handle."""
    dl = hf_download("raw/swe-zero-12m-trajectories", hf_id=HF_DATASET_ID, revision=HF_REVISION, version="2026.06.28")
    processed = ArtifactStep(
        name="processed/swe-zero-12m-trajectories",
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
        name="normalized/swe-zero-12m",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.artifact_path(processed), "output_path": ctx.output_path},
        deps=(processed,),
    )
    return tokenized("swe-zero-12m", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet", version="2026.06.28")


if __name__ == "__main__":
    dataset_main({"swe-zero-12m": swe_zero_12m_dataset()})
