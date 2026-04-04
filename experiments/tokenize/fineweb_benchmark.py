# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark HF vs tokie tokenizer backends on fineweb-edu 10BT sample.

Runs two tokenization jobs over the same data — one with the HF backend,
one with tokie — so throughput and memory behavior can be compared.

Usage:
    uv run iris --config=lib/iris/examples/marin-dev.yaml job run -- \
        python experiments/tokenize/fineweb_benchmark.py
"""

import logging
import os

from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

from levanter.tokenizers import TokenizerBackend
from marin.datakit.download.huggingface import download_hf_step
from marin.execution.executor import ExecutorStep, versioned
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

logger = logging.getLogger(__name__)

TOKENIZER = "meta-llama/Llama-3.1-8B"
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "tokenizer-benchmark")


def build_steps() -> list[StepSpec | ExecutorStep]:
    download = download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
    )

    raw_train_paths = [os.path.join(download.output_path, "sample/10BT")]
    prefix = f"{marin_prefix()}/tmp/{OUTPUT_PREFIX}"

    hf_config = TokenizeConfig(
        train_paths=raw_train_paths,
        validation_paths=versioned([]),
        cache_path=f"{prefix}/hf",
        tokenizer=versioned(TOKENIZER),
        tokenizer_backend=TokenizerBackend.HF,
    )

    tokie_config = TokenizeConfig(
        train_paths=raw_train_paths,
        validation_paths=versioned([]),
        cache_path=f"{prefix}/tokie",
        tokenizer=versioned(TOKENIZER),
        tokenizer_backend=TokenizerBackend.TOKIE,
    )

    hf_step = ExecutorStep(
        name="tokenize/fineweb-benchmark-hf",
        fn=tokenize,
        config=hf_config,
    )

    tokie_step = ExecutorStep(
        name="tokenize/fineweb-benchmark-tokie",
        fn=tokenize,
        config=tokie_config,
    )

    return [download, hf_step, tokie_step]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
