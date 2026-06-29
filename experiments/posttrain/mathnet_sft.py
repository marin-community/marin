# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Text-only SFT tokenization step for ShadenA/MathNet."""

from levanter.data.text import ChatLmDatasetFormat
from marin.datakit.download.mathnet import mathnet_text_sft_primary_step
from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer

mathnet_text_sft_primary = mathnet_text_sft_primary_step()
mathnet_text_sft_primary_executor = mathnet_text_sft_primary.as_executor_step()

mathnet_text_sft_primary_tokenized = default_tokenize(
    name="mathnet-v0-text-sft-primary-marin-tokenizer",
    dataset=mathnet_text_sft_primary_executor / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=ChatLmDatasetFormat(),
)


if __name__ == "__main__":
    executor_main(steps=[mathnet_text_sft_primary_executor, mathnet_text_sft_primary_tokenized])
