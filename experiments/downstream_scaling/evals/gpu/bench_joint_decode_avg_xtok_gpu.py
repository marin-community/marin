# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the 1e22 cross-tokenizer joint-decode-avg workload on two GPUs.

Runs one chunk of the run_delphi_gsm8k_joint_decode_avg_xtok_gpu workload
(same models, sampling constants, and batch shape, imported from that script)
against the real GSM8K prompts materialized by a prior run (PROMPTS_PATH),
and decomposes where each decision round's wall time goes:

- ``coordinator resolve``: serial time inside the coordinator resolving each
  round (selector math for every live row runs here, under the coordinator
  lock, while both engines idle). The ``selector`` line is the portion spent
  inside the selection function itself.
- ``advisor drain depth``: per-round max of len(tokens_b); the advisor must
  locally emit the round's longest forced segmentation before the next
  barrier, so (depth - 1) estimates its extra forward passes per round.
- ``round gap residual``: median inter-round gap minus resolve time — engine
  steps, drain steps, and HTTP/JSON transport.

Run twice to isolate the prefix-mass cost: ``--selector anchored`` (the real
rule) vs ``--selector cheap`` (argmax over A's candidates; identical
byte-mapping, forcing, drain, and transport, near-zero scoring math).

Example:
    .venv/bin/python experiments/downstream_scaling/evals/gpu/\
bench_joint_decode_avg_xtok_gpu.py --num-requests 512 --selector anchored
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from joint_decode.config import JointDecodeSamplingConfig
from joint_decode.coordinator import Coordinator
from joint_decode.gpu.config import JointDecodeConfig, JointDecodeModelConfig
from joint_decode.gpu.decoder import joint_decoder
from transformers import AutoTokenizer

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.framework.schema import read_prompt_rows
from experiments.downstream_scaling.evals.gpu.run_delphi_gsm8k_joint_decode_avg_xtok_gpu import (
    ADVISOR_MAX_TOKENS,
    ADVISOR_MODEL,
    BARRIER_TIMEOUT_S,
    GPU_MEMORY_UTILIZATION,
    MAX_MICROBATCH_SIZE,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    N_SAMPLES,
    PREFIX_CREDIT,
    SEED,
    STOP_TOKENS,
    TEMPERATURE,
    TOP_K_A,
)
from experiments.downstream_scaling.models.delphi import DELPHI_HF_REPOS

DECODER_MODEL = DELPHI_HF_REPOS["1e22"]
ADVISOR_WEIGHT = 0.5
WARMUP_REQUESTS = 16
# The GSM8K prompts step output from a prior runner invocation (5-shot,
# seed 1234, 256 problems). Adjust the prefix root for the machine: the run
# prefix on the GPU box, or its rsync copy next to the repo.
PROMPTS_PATH = "/juice4/scr4/nlp/model-tracing/marin/marin-prefix/downstream_scaling/evals/prompts/gsm8k-fc535b/prompts.jsonl.gz"


@dataclass
class RoundStats:
    """Per-round decomposition, filled by the coordinator/selector wrappers.

    Selector calls happen synchronously inside _resolve_decode_pair (under the
    coordinator lock), so per-round accumulation needs no extra locking."""

    resolve_starts: list[float] = field(default_factory=list)
    resolve_walls: list[float] = field(default_factory=list)
    round_rows: list[int] = field(default_factory=list)
    round_max_len_b: list[int] = field(default_factory=list)
    selector_total: float = 0.0
    selector_calls: int = 0
    tokens_a_total: int = 0
    tokens_b_total: int = 0
    _current_max_len_b: int = 0

    def record_selection(self, elapsed: float, tokens_a: list[int], tokens_b: list[int]) -> None:
        self.selector_total += elapsed
        self.selector_calls += 1
        self.tokens_a_total += len(tokens_a)
        self.tokens_b_total += len(tokens_b)
        self._current_max_len_b = max(self._current_max_len_b, len(tokens_b))

    def begin_round(self) -> None:
        self._current_max_len_b = 0

    def end_round(self, start: float, wall: float, rows: int) -> None:
        self.resolve_starts.append(start)
        self.resolve_walls.append(wall)
        self.round_rows.append(rows)
        self.round_max_len_b.append(self._current_max_len_b)

    def reset(self) -> None:
        self.__init__()


STATS = RoundStats()

_ORIG_RESOLVE = Coordinator._resolve_decode_pair


def _timed_resolve(self: Coordinator, entry_a: Any, entry_b: Any) -> None:
    start = time.perf_counter()
    STATS.begin_round()
    _ORIG_RESOLVE(self, entry_a, entry_b)
    STATS.end_round(start, time.perf_counter() - start, rows=len(entry_a.request_ids))


def _make_select_token(selector: str, vocab_a: xtok_selection.Vocab, vocab_b: xtok_selection.Vocab):
    def anchored(a_topk, b_topk, *, rng, request_index):
        del request_index
        return xtok_selection.select_avg_anchored(
            a_topk,
            b_topk,
            advisor_weight=ADVISOR_WEIGHT,
            temperature=TEMPERATURE,
            prefix_credit=PREFIX_CREDIT,
            rng=rng,
            vocab_a=vocab_a,
            vocab_b=vocab_b,
        )

    def cheap(a_topk, b_topk, *, rng, request_index):
        # Same byte mapping, forcing, drain, and transport as anchored, with
        # the softmax/prefix-mass scoring replaced by an A-side argmax: the
        # anchored-vs-cheap throughput delta isolates the scoring math.
        del rng, request_index
        a = xtok_selection.candidates(vocab_a, a_topk)
        b = xtok_selection.candidates(vocab_b, b_topk)
        key = max(a, key=lambda k: a[k].logit)
        return xtok_selection.force(vocab_a, key, a), xtok_selection.force(vocab_b, key, b)

    rule = anchored if selector == "anchored" else cheap

    def select_token(a_topk, b_topk, *, rng, request_index):
        start = time.perf_counter()
        tokens_a, tokens_b = rule(a_topk, b_topk, rng=rng, request_index=request_index)
        STATS.record_selection(time.perf_counter() - start, tokens_a, tokens_b)
        return tokens_a, tokens_b

    return select_token


def load_prompts() -> list[str]:
    """The runner's request stream: each prompt row repeated N_SAMPLES times,
    matching the composition of a real chunk."""
    rows = list(read_prompt_rows(PROMPTS_PATH))
    return [row["prompt"] for row in rows for _ in range(N_SAMPLES)]


def _report(label: str, wall: float) -> None:
    gaps = [b - a for a, b in zip(STATS.resolve_starts, STATS.resolve_starts[1:], strict=False)]
    resolve_total = sum(STATS.resolve_walls)
    rounds = len(STATS.resolve_walls)
    print(f"\n=== {label} ===")
    print(f"wall {wall:.1f}s; rounds {rounds}; selector calls {STATS.selector_calls}")
    print(
        f"tokens: A {STATS.tokens_a_total} ({STATS.tokens_a_total / wall:.1f}/s), "
        f"B {STATS.tokens_b_total} ({STATS.tokens_b_total / wall:.1f}/s), "
        f"B/A fertility {STATS.tokens_b_total / max(STATS.tokens_a_total, 1):.2f}"
    )
    if not rounds:
        return
    print(
        f"coordinator resolve: total {resolve_total:.1f}s ({100 * resolve_total / wall:.1f}% of wall); "
        f"median {1e3 * statistics.median(STATS.resolve_walls):.1f}ms/round"
    )
    print(
        f"  selector: total {STATS.selector_total:.1f}s ({100 * STATS.selector_total / wall:.1f}% of wall); "
        f"mean {1e6 * STATS.selector_total / max(STATS.selector_calls, 1):.0f}us/call"
    )
    if gaps:
        print(
            f"round gap: median {1e3 * statistics.median(gaps):.1f}ms, p90 "
            f"{1e3 * statistics.quantiles(gaps, n=10)[-1]:.1f}ms "
            f"(gap - resolve = engines + drain + transport)"
        )
    print(
        f"advisor drain depth (per-round max len_b): mean "
        f"{statistics.mean(STATS.round_max_len_b):.2f}, max {max(STATS.round_max_len_b)} "
        f"(mean - 1 ~= extra B forward passes per round)"
    )
    print(f"rows per round: median {statistics.median(STATS.round_rows):.0f}, max {max(STATS.round_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--selector", choices=("anchored", "cheap"), default="anchored")
    parser.add_argument("--top-k-b", type=int, default=64)
    args = parser.parse_args()

    Coordinator._resolve_decode_pair = _timed_resolve

    vocab_a = xtok_selection.load_vocab(AutoTokenizer.from_pretrained(DECODER_MODEL))
    vocab_b = xtok_selection.load_vocab(AutoTokenizer.from_pretrained(ADVISOR_MODEL))
    select_token = _make_select_token(args.selector, vocab_a, vocab_b)

    decode_config = JointDecodeConfig(
        model_a=JointDecodeModelConfig(
            model_path=DECODER_MODEL,
            gpu_index=0,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=False,
            enforce_eager=True,
        ),
        model_b=JointDecodeModelConfig(
            model_path=ADVISOR_MODEL,
            gpu_index=1,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=False,
            enforce_eager=True,
        ),
        sampling=JointDecodeSamplingConfig(
            max_tokens_a=MAX_TOKENS,
            max_tokens_b=ADVISOR_MAX_TOKENS,
            top_k_a=TOP_K_A,
            top_k_b=args.top_k_b,
            barrier_timeout_s=BARRIER_TIMEOUT_S,
            seed=SEED,
            stop=STOP_TOKENS,
            max_microbatch_size=MAX_MICROBATCH_SIZE,
            max_num_batched_tokens=MAX_MODEL_LEN + MAX_MICROBATCH_SIZE,
        ),
    )

    flat = load_prompts()
    if args.num_requests > len(flat):
        raise ValueError(f"--num-requests {args.num_requests} exceeds the {len(flat)} requests in the prompts file")
    prompts = flat[: args.num_requests]
    warmup = flat[:WARMUP_REQUESTS]

    with joint_decoder(decode_config, select_token=select_token) as decoder:
        print(f"warmup: {WARMUP_REQUESTS} requests")
        decoder.generate(warmup, warmup)
        STATS.reset()

        print(f"benchmark: {args.num_requests} requests, selector={args.selector}, top_k_b={args.top_k_b}")
        started = time.perf_counter()
        outputs = decoder.generate(prompts, prompts)
        wall = time.perf_counter() - started

    finished = sum(1 for output in outputs if output.text)
    print(f"completed {len(outputs)} requests ({finished} non-empty)")
    _report(f"selector={args.selector} top_k_b={args.top_k_b} n={args.num_requests}", wall)


if __name__ == "__main__":
    main()
