# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-log dataset definitions and tokenization."""

from fray import ResourceConfig
from marin.datakit.download.diagnostic_logs import (
    DiagnosticPartition,
    materialize_ghalogs_partition_step,
    materialize_ghalogs_step,
)
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

_GHALOGS_MATERIALIZED_STEP = materialize_ghalogs_step()
_GHALOGS_TRAIN_STEP = materialize_ghalogs_partition_step(
    materialized=_GHALOGS_MATERIALIZED_STEP,
    partition=DiagnosticPartition.TRAIN,
)
_GHALOGS_DEV_STEP = materialize_ghalogs_partition_step(
    materialized=_GHALOGS_MATERIALIZED_STEP,
    partition=DiagnosticPartition.DEV,
)

ghalogs_download = _GHALOGS_TRAIN_STEP.as_executor_step()
ghalogs_dev = _GHALOGS_DEV_STEP.as_executor_step()


def tokenize_ghalogs(*, tokenizer: str | None = None) -> ExecutorStep[TokenizeConfig]:
    """Tokenize the train/dev GHALogs parquet shards."""
    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    return ExecutorStep(
        name="tokenized/ghalogs_public",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[ghalogs_download.as_input_name() / "*.parquet"],
            validation_paths=[ghalogs_dev.as_input_name() / "*.parquet"],
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            worker_resources=ResourceConfig(ram="40g", disk="5g"),
        ),
    )
