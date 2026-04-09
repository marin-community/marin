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
    reshard_starcoder2_extras_step,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import TokenizerStep

WORKER_RAM = {"ir_low_resource": "80g"}
DEFAULT_WORKER_RAM = "40g"


def tokenize_starcoder2_extras(*, tokenizer: str = marin_tokenizer) -> list[TokenizerStep]:
    """Download and tokenize all selected starcoder2data-extras subsets."""
    steps = []
    RESHARD_SUBSETS = {"ir_low_resource"}
    for subset in SUBSETS:
        if subset in RESHARD_SUBSETS:
            download = reshard_starcoder2_extras_step(subset)
        else:
            download = download_starcoder2_extras_step(subset)
        ram = WORKER_RAM.get(subset, DEFAULT_WORKER_RAM)
        steps.append(
            default_tokenize(
                name=f"starcoder2_extras/{subset}",
                dataset=download.as_executor_step(),
                tokenizer=tokenizer,
                format=TextLmDatasetFormat(text_key="content"),
                worker_resources=ResourceConfig(ram=ram, disk="10g"),
            )
        )
    return steps


if __name__ == "__main__":
    executor_main(steps=tokenize_starcoder2_extras())
