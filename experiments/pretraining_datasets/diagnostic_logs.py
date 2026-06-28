# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-log dataset as a lazy ``Dataset`` handle.

The normalized GHALogs training partition (a 3-step Zephyr pipeline: materialize
parquet → filter to train partition → normalize) is expressed as a single raw-data
handle. The tokenize step reads the normalized parquet from ``outputs/main/``.
"""

from marin.datakit.download.diagnostic_logs import (
    DEFAULT_GHALOGS_MATERIALIZE_SHARDS,
    GHALOGS_STAGED_PREFIX,
    DiagnosticPartition,
    materialize_ghalogs_partition_to_parquet,
    materialize_ghalogs_to_parquet,
)
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.artifact import Dataset
from marin.execution.lazy import Lazy
from marin.experiment.data import raw_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer


def _run_ghalogs_normalize_pipeline(output_path: str) -> None:
    """Materialize, partition, and normalize GHALogs into ``output_path/outputs/main/``."""
    materialized_path = f"{output_path}/_wip/materialized"
    train_path = f"{output_path}/_wip/train"
    materialize_ghalogs_to_parquet(
        GHALOGS_STAGED_PREFIX,
        materialized_path,
        num_shards=DEFAULT_GHALOGS_MATERIALIZE_SHARDS,
    )
    materialize_ghalogs_partition_to_parquet(
        materialized_path,
        train_path,
        partition=DiagnosticPartition.TRAIN,
    )
    normalize_to_parquet(
        input_path=train_path,
        output_path=output_path,
        text_field="text",
        id_field="id",
        file_extensions=(".parquet",),
    )


def _ghalogs_normalized() -> Lazy[Dataset]:
    """The normalized GHALogs public training partition as a raw-data handle."""
    return raw_download(
        "normalized/ghalogs/public",
        fn=_run_ghalogs_normalize_pipeline,
        build_config=lambda ctx: ctx.out,
    )


def tokenize_ghalogs(*, tokenizer: str = marin_tokenizer) -> Lazy[Dataset]:
    """Tokenized GHALogs public training partition."""
    return tokenized(
        "ghalogs_public",
        tokenizer=tokenizer,
        raw=_ghalogs_normalized(),
        glob="outputs/main/*.parquet",
    )
