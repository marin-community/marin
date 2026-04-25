# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark HF vs kitoken tokenizer backends on fineweb-edu 10BT sample.

Runs tokenization jobs over the same data — one per backend —
so throughput and memory behavior can be compared. Also runs a validation
job that compares kitoken against HF and outputs any documents
where the tokenization differs.

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
from collections.abc import Iterator, Sequence

from levanter.tokenizers import TokenizerBackend, load_tokenizer
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext, zephyr_worker_ctx
from zephyr.readers import load_file

from fray import ResourceConfig
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.tokenize import _get_filepaths_to_tokenize
from levanter.data.text import TextLmDatasetFormat

logger = logging.getLogger(__name__)

TOKENIZER = "meta-llama/Llama-3.1-8B"
TINY = os.environ.get("BENCHMARK_TINY", "") == "1"

# All data and outputs live in us-central1. Force the prefix so that
# orchestrator and workers all write to the same region regardless of
# where Iris initially places the parent task.
DATA_REGION = "us-central1"
os.environ.setdefault("MARIN_PREFIX", f"gs://marin-{DATA_REGION}")

# Pre-downloaded fineweb-edu 10BT sample (revision 87f0914).
FINEWEB_10BT = f"gs://marin-{DATA_REGION}/raw/fineweb-edu-87f0914/sample/10BT"


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
                worker_resources=ResourceConfig(ram="16g", disk="5g", regions=[DATA_REGION]),
            )
        ),
    )


def _compare_batches(batches: Iterator[Sequence[dict]], _shard_id: int) -> Iterator[dict]:
    """Compare tokenization of each document with alt backend vs HF.

    Shared context must contain 'tokenizer_name' and 'alt_backend'.
    Yields documents where the two backends produce different token IDs.
    """
    ctx = zephyr_worker_ctx()
    name = ctx.get_shared("tokenizer_name")
    alt_backend = ctx.get_shared("alt_backend")

    hf_tok = load_tokenizer(name, backend=TokenizerBackend.HF)
    alt_tok = load_tokenizer(name, backend=alt_backend)

    total = 0
    mismatches = 0

    for batch in batches:
        for doc in batch:
            text = doc.get("text", "")
            total += 1

            hf_ids = hf_tok.encode(text, add_special_tokens=False)
            alt_ids = alt_tok.encode(text, add_special_tokens=False)

            if hf_ids != alt_ids:
                mismatches += 1
                first_diff = next(
                    (i for i, (a, b) in enumerate(zip(hf_ids, alt_ids, strict=False)) if a != b),
                    min(len(hf_ids), len(alt_ids)),
                )
                yield {
                    "text": text,
                    "hf_tokens": len(hf_ids),
                    "alt_tokens": len(alt_ids),
                    "alt_backend": alt_backend.value,
                    "first_diff_pos": first_diff,
                    "hf_ids_at_diff": hf_ids[max(0, first_diff - 2) : first_diff + 3],
                    "alt_ids_at_diff": alt_ids[max(0, first_diff - 2) : first_diff + 3],
                }

    logger.info(f"Compared {total} docs against {alt_backend.value}: {mismatches} mismatches")


def _compare_spec(
    name: str,
    train_path: str,
    alt_backend: TokenizerBackend,
    sample_count: int | None = None,
) -> StepSpec:
    prefix = f"{marin_prefix()}/tmp/tokenizer-benchmark"

    def run(output_path: str) -> None:
        ctx = ZephyrContext(
            max_workers=32,
            resources=ResourceConfig(cpu=1, ram="16g", regions=[DATA_REGION]),
            name=name,
        )
        ctx.put("tokenizer_name", TOKENIZER)
        ctx.put("alt_backend", alt_backend)

        files = _get_filepaths_to_tokenize([train_path])
        ds = Dataset.from_list(files).flat_map(load_file)
        if sample_count is not None:
            ds = ds.take_per_shard(sample_count)

        ctx.execute(
            ds.window(64)
            .map_shard(_compare_batches)
            .write_jsonl(f"{output_path}/mismatches-{{shard:05d}}-of-{{total:05d}}.jsonl"),
        )

    return StepSpec(
        name=name,
        output_path_prefix=f"{prefix}/{name}",
        fn=run,
    )


def build_steps() -> list[StepSpec]:
    train_path = _make_tiny_dataset() if TINY else FINEWEB_10BT
    sample_count = 200 if TINY else None
    return [
        # _tokenize_spec("benchmark-hf", train_path, TokenizerBackend.HF, sample_count=sample_count),
        # _tokenize_spec("benchmark-kitoken", train_path, TokenizerBackend.KITOKEN, sample_count=sample_count),
        _compare_spec("compare-kitoken-vs-hf", train_path, TokenizerBackend.KITOKEN, sample_count=sample_count),
    ]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
