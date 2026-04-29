# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Corpus dataset definitions and tokenization."""

from fray import ResourceConfig

from marin.execution.dag import ExecutorStep, this_output_path, versioned
from marin.datakit.download.common_corpus import (
    download_common_corpus_raw_step,
    filter_common_corpus_step,
    normalize_common_corpus_step,
)
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

common_corpus_download = normalize_common_corpus_step(
    filter_common_corpus_step(download_common_corpus_raw_step())
).as_executor_step()


def tokenize_common_corpus(*, tokenizer: str | None = None) -> TokenizerStep:
    """Tokenize the filtered Common Corpus (English, open types)."""
    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    return ExecutorStep(
        name="tokenized/common_corpus_english",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[common_corpus_download.as_input_name() / "outputs/main/*.parquet"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            worker_resources=ResourceConfig(ram="40g", disk="5g"),
        ),
    )
