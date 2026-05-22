# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-log dataset definitions and tokenization."""

from marin.datakit.download.diagnostic_logs import ghalogs_public_normalize_steps
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

ghalogs_normalized = ghalogs_public_normalize_steps()[-1].as_executor_step()


def tokenize_ghalogs(*, tokenizer: str | None = None) -> ExecutorStep[TokenizeConfig]:
    """Tokenize the normalized GHALogs public training partition."""
    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    return ExecutorStep(
        name="tokenized/ghalogs_public",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[ghalogs_normalized.as_input_name() / "outputs/main/*.parquet"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )
