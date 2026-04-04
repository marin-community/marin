# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark HF vs tokie vs kitoken tokenizer backends on fineweb-edu 10BT sample.

Runs three tokenization jobs over the same data — one per backend —
so throughput and memory behavior can be compared.

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
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from fray.v2 import ResourceConfig
from marin.processing.tokenize import TokenizeConfig, tokenize

logger = logging.getLogger(__name__)

TOKENIZER = "meta-llama/Llama-3.1-8B"
TINY = os.environ.get("BENCHMARK_TINY", "") == "1"

# Pre-downloaded fineweb-edu 10BT sample (revision 87f0914).
FINEWEB_10BT = "gs://marin-us-central1/raw/fineweb-edu-87f0914/sample/10BT"


def _make_tiny_dataset() -> str:
    """Create a tiny JSONL file for local smoke testing."""
    tmpdir = tempfile.mkdtemp(prefix="tokenizer-benchmark-")
    path = os.path.join(tmpdir, "tiny.jsonl")
    with open(path, "w") as f:
        for i in range(200):
            f.write(json.dumps({"text": f"The quick brown fox jumps over the lazy dog. Sentence number {i}."}) + "\n")
    return path


def _tokenize_spec(
    name: str,
    train_path: str,
    backend: TokenizerBackend,
    sample_count: int | None = None,
) -> StepSpec:
    prefix = f"{marin_prefix()}/tmp/tokenizer-benchmark"
    return StepSpec(
        name=name,
        output_path_prefix=f"{prefix}/{name}",
        fn=lambda op: tokenize(
            TokenizeConfig(
                train_paths=[train_path],
                validation_paths=[],
                cache_path=op,
                tokenizer=TOKENIZER,
                tokenizer_backend=backend,
                format=TextLmDatasetFormat(),
                sample_count=sample_count,
                worker_resources=ResourceConfig(ram="16g", disk="5g"),
            )
        ),
    )


def build_steps() -> list[StepSpec]:
    train_path = _make_tiny_dataset() if TINY else FINEWEB_10BT
    sample_count = 200 if TINY else None
    return [
        _tokenize_spec("benchmark-hf", train_path, TokenizerBackend.HF, sample_count=sample_count),
        _tokenize_spec("benchmark-tokie", train_path, TokenizerBackend.TOKIE, sample_count=sample_count),
        _tokenize_spec("benchmark-kitoken", train_path, TokenizerBackend.KITOKEN, sample_count=sample_count),
    ]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
