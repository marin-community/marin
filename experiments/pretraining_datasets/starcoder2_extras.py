# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2 data extras: download and tokenize ir_cpp, ir_python, ir_rust, ir_low_resource, documentation."""

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from levanter.data.text.formats import TextLmDatasetFormat
from marin.datakit.download.starcoder2_extras import (
    SUBSETS,
    download_starcoder2_extras_step,
)
from marin.datakit.normalize import normalize_step
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import TokenizerStep


def tokenize_starcoder2_extras(*, tokenizer: str = marin_tokenizer) -> list[TokenizerStep]:
    """Download, normalize, and tokenize all selected starcoder2data-extras subsets."""
    steps = []
    for subset in SUBSETS:
        download = download_starcoder2_extras_step(subset)
        normalized = normalize_step(
            name=f"normalized/starcoder2_extras/{subset}",
            download=download,
            text_field="content",
            file_extensions=(".parquet",),
            # documentation contains very large records (e.g. full OpenJDK docs at 64MB);
            # split them to avoid OOM during tokenization
            max_record_size=10_000_000 if subset == "documentation" else None,
        )
        steps.append(
            default_tokenize(
                name=f"starcoder2_extras/{subset}",
                dataset=normalized.as_executor_step(),
                tokenizer=tokenizer,
                format=TextLmDatasetFormat(text_key="text"),
                levanter_batch_size=128,
            )
        )
    return steps


if __name__ == "__main__":
    executor_main(steps=tokenize_starcoder2_extras())
