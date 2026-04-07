# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NSF awards dataset download and tokenization."""

from experiments.marin_models import marin_tokenizer
from marin.datakit.download.nsf_awards import download_nsf_awards_step
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

nsf_awards_download = download_nsf_awards_step().as_executor_step()

nsf_awards_tokenized = ExecutorStep(
    name="tokenized/nsf_awards",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(nsf_awards_download, "*.parquet")],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(marin_tokenizer),
    ),
)
