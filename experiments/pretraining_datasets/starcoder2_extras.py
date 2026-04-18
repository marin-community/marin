# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2 data extras: download and tokenize ir_cpp, ir_python, ir_rust, ir_low_resource, documentation."""

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from fray.v2 import ResourceConfig
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
        )
        # documentation contains a single 64MB OpenJDK record that peaks at ~9GB RSS
        # during tokenization; bump memory to 32GB for that subset
        doc_resources = ResourceConfig(ram="32g", disk="10g") if subset == "documentation" else None
        steps.append(
            default_tokenize(
                name=f"starcoder2_extras/{subset}",
                # Normalize splits main/dup outputs; only tokenize the main branch.
                dataset=normalized.as_executor_step() / "outputs/main/*.parquet",
                tokenizer=tokenizer,
                format=TextLmDatasetFormat(text_key="text"),
                levanter_batch_size=128,
                worker_resources=doc_resources,
            )
        )
    return steps


if __name__ == "__main__":
    executor_main(steps=tokenize_starcoder2_extras())
