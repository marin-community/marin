# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark HF vs tokie tokenizer backends on fineweb-edu 10BT sample.

Runs two tokenization jobs over the same data — one with the HF backend,
one with tokie — so throughput and memory behavior can be compared.

Usage (local, tiny synthetic dataset for smoke testing):
    MARIN_PREFIX=/tmp/marin BENCHMARK_TINY=1 uv run python experiments/tokenize/fineweb_benchmark.py

Usage (cluster, full 10BT):
    uv run iris --config=lib/iris/examples/marin-dev.yaml job run -- \
        python experiments/tokenize/fineweb_benchmark.py
"""

import json
import logging
import os
import tempfile

from levanter.data.text import TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from rigging.log_setup import configure_logging

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

logger = logging.getLogger(__name__)

TOKENIZER = "meta-llama/Llama-3.1-8B"
TINY = os.environ.get("BENCHMARK_TINY", "") == "1"


def _make_tiny_dataset() -> str:
    """Create a tiny JSONL file for local smoke testing."""
    tmpdir = tempfile.mkdtemp(prefix="tokenizer-benchmark-")
    path = os.path.join(tmpdir, "tiny.jsonl")
    with open(path, "w") as f:
        for i in range(200):
            f.write(json.dumps({"text": f"The quick brown fox jumps over the lazy dog. Sentence number {i}."}) + "\n")
    return path


def _build_hf_and_tokie_steps(
    raw_train_paths: list[str],
    sample_count: int | None,
) -> list[ExecutorStep]:
    hf_tokenize = ExecutorStep(
        name="tokenize/benchmark-hf",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=raw_train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(TOKENIZER),
            tokenizer_backend=TokenizerBackend.HF,
            format=TextLmDatasetFormat(),
            sample_count=sample_count,
        ),
    )

    tokie_tokenize = ExecutorStep(
        name="tokenize/benchmark-tokie",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=raw_train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(TOKENIZER),
            tokenizer_backend=TokenizerBackend.TOKIE,
            format=TextLmDatasetFormat(),
            sample_count=sample_count,
        ),
    )

    return [hf_tokenize, tokie_tokenize]


if TINY:
    tiny_path = _make_tiny_dataset()
    steps = _build_hf_and_tokie_steps([tiny_path], sample_count=200)
else:
    dataset = download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
    )
    steps = [
        dataset,
        *_build_hf_and_tokie_steps(
            [os.path.join(dataset.output_path, "sample/10BT")],
            sample_count=None,
        ),
    ]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    executor_main(steps=steps)
