# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NSF awards dataset download, normalization, and tokenization."""

from marin.datakit.download.nsf_awards import download_nsf_awards_step, normalize_nsf_awards_step
from marin.execution.types import ExecutorStep, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_tokenizer import marin_tokenizer


def nsf_awards_download() -> ExecutorStep:
    return normalize_nsf_awards_step(download_nsf_awards_step()).as_executor_step()


def nsf_awards_tokenized() -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/nsf_awards",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[output_path_of(nsf_awards_download(), "outputs/main/*.parquet")],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(marin_tokenizer),
        ),
    )
