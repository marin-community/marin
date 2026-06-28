# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO-12M-trajectories pre-training dataset as a lazy Dataset handle.

12.29M execution-free agentic-coding trajectories from
AlienKevin/SWE-ZERO-12M-trajectories, rendered via the mini-swe-agent v1 format,
normalized, and tokenized.
"""

from marin.datakit.download.swe_zero_12m import HF_DATASET_ID, HF_REVISION, transform
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.artifact import Dataset
from marin.execution.lazy import Lazy, derived
from marin.experiment.data import hf_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(input_path=cfg["input_path"], output_path=cfg["output_path"])


def swe_zero_12m_datasets(*, tokenizer: str = marin_tokenizer) -> Lazy[Dataset]:
    """SWE-ZERO-12M-trajectories as a tokenized Dataset handle."""
    dl = hf_download("raw/swe-zero-12m-trajectories", hf_id=HF_DATASET_ID, revision=HF_REVISION)
    processed = derived(
        "processed/swe-zero-12m-trajectories",
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
        "normalized/swe-zero-12m",
        fn=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.path(processed), "output_path": ctx.out},
        deps=(processed,),
        kind=Dataset,
    )
    return tokenized("swe-zero-12m", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet")
