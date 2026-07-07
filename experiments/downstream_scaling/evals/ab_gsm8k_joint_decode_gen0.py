# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generation-0 side of the joint-decode A/B gate: standalone GSM8K generation.

Runs the generation-0 joint-decode-avg engine pair (from
``joint_decode_avg_xregion``, the only gen-0 variant runnable at this
branch's pins) directly on two local TPU chips — no executor, no cluster
jobs — over a pre-staged prompts file (the production GSM8K prompts
artifact, e.g. ``prompts/gsm8k-99b4ac``), writes chunk files for later
scoring, and prints per-chunk timing plus finish-reason and
completion-length summaries. The new-engine counterpart
(``ab_gsm8k_joint_decode_new.py`` on the ``jd-unify-phase3`` branch) has
the identical structure and sampling configuration; the summaries — and,
once graded, the stored chunks' accuracies — are the A/B comparison.
Grading is deferred: the canonical lm_eval grader is broken under current
pins (see the plan doc), and the chunk files keep until it is fixed.

Run on a TPU worker (vllm extra), with prompts.jsonl.gz staged in
--output-dir first:

    uv run --no-sync python \\
      experiments/downstream_scaling/evals/ab_gsm8k_joint_decode_gen0.py \\
      --decoder-model <hf-checkpoint-path> --advisor-model <hf-model-path> \\
      --output-dir ~/ab_gsm8k/gen0
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xregion import (
    JointDecodeModelConfig,
    JointDecoder,
    JointDecodeSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.schema import prompts_file, read_prompt_rows

logger = logging.getLogger(__name__)

STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-model", required=True)
    parser.add_argument("--advisor-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-problems", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--chip-a", type=int, default=0)
    parser.add_argument("--chip-b", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-k-a", type=int, default=16)
    parser.add_argument("--top-k-b", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--advisor-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--barrier-timeout-s", type=float, default=1200.0)
    return parser.parse_args()


def load_prompts(output_dir: str, n_problems: int) -> list[dict]:
    path = prompts_file(output_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; stage the production GSM8K prompts artifact there first "
            "(e.g. downstream_scaling/evals/prompts/gsm8k-99b4ac/prompts.jsonl.gz)"
        )
    rows = list(read_prompt_rows(path))
    if len(rows) != n_problems:
        raise ValueError(f"{path} has {len(rows)} prompts; expected {n_problems}")
    for row in rows:
        metadata = row["metadata"]
        if metadata["num_fewshot"] != NUM_FEWSHOT or metadata["fewshot_seed"] != FEWSHOT_SEED:
            raise ValueError(f"prompt {row['id']} has fewshot config {metadata}; expected {NUM_FEWSHOT}/{FEWSHOT_SEED}")
    return rows


def chunk_paths(chunks_dir: Path, chunk_id: int) -> tuple[Path, Path]:
    return chunks_dir / f"chunk-{chunk_id:06d}.jsonl.gz", chunks_dir / f"chunk-{chunk_id:06d}.SUCCESS"


def generate(args: argparse.Namespace, rows: list[dict], chunks_dir: Path) -> None:
    prompt_ids = [row["id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    total = len(rows) * args.n_samples
    n_chunks = (total + args.chunk_size - 1) // args.chunk_size

    sampling = JointDecodeSamplingConfig(
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        top_k_a=args.top_k_a,
        top_k_b=args.top_k_b,
        seed=args.seed,
        temperature=args.temperature,
        stop=STOP_TOKENS,
        advisor_weight=args.advisor_weight,
    )

    with JointDecoder(
        decoder_model_path=args.decoder_model,
        advisor_model_path=args.advisor_model,
        sampling=sampling,
        # Decoder is delphi-shaped (needs the RPA patch); advisor is not.
        decoder_model=JointDecodeModelConfig(max_model_len=args.max_model_len, apply_rpa_block_size_patch=True),
        advisor_model=JointDecodeModelConfig(max_model_len=args.max_model_len),
        chip_a=args.chip_a,
        chip_b=args.chip_b,
        barrier_timeout_s=args.barrier_timeout_s,
        microbatch_size=args.chunk_size,
        assets_cache_dir=chunks_dir.parent / "cache",
    ) as decoder:
        for chunk_id in range(n_chunks):
            output_path, success_path = chunk_paths(chunks_dir, chunk_id)
            if success_path.exists():
                logger.info("chunk %d/%d already done; skipping", chunk_id + 1, n_chunks)
                continue
            start = chunk_id * args.chunk_size
            end = min(start + args.chunk_size, total)
            chunk_prompts = [prompts[i // args.n_samples] for i in range(start, end)]

            chunk_start = time.monotonic()
            outputs = decoder.generate(chunk_prompts)

            with gzip.open(output_path, "wt") as f:
                for i, output in zip(range(start, end), outputs, strict=True):
                    f.write(
                        json.dumps(
                            {
                                "id": prompt_ids[i // args.n_samples],
                                "completion_index": i % args.n_samples,
                                "completion": {
                                    "text": output.text,
                                    "metadata": {"finish_reason": output.finish_reason},
                                },
                            }
                        )
                        + "\n"
                    )
            success_path.write_text("ok\n")
            logger.info("chunk %d/%d done in %.1fs", chunk_id + 1, n_chunks, time.monotonic() - chunk_start)


def summarize(chunks_dir: Path) -> None:
    finish_reasons: Counter[str] = Counter()
    lengths: list[int] = []
    for chunk_file in sorted(chunks_dir.glob("chunk-*.jsonl.gz")):
        with gzip.open(chunk_file, "rt") as f:
            for line in f:
                record = json.loads(line)
                finish_reasons[str(record["completion"]["metadata"]["finish_reason"])] += 1
                lengths.append(len(record["completion"]["text"]))
    quantiles = statistics.quantiles(lengths, n=10)
    print(f"completions: {len(lengths)}")
    print(f"finish_reasons: {dict(finish_reasons.most_common())}")
    print(
        f"completion chars: mean={statistics.mean(lengths):.0f} "
        f"p10={quantiles[0]:.0f} p50={quantiles[4]:.0f} p90={quantiles[8]:.0f}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    chunks_dir = Path(output_dir) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    rows = load_prompts(output_dir, args.n_problems)
    generate(args, rows, chunks_dir)
    summarize(chunks_dir)


if __name__ == "__main__":
    main()
