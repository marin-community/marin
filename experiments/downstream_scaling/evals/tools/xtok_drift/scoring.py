# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score advisor drift on recorded xtok rollouts.

For each selected rollout, run the advisor (Qwen) over the same byte prefixes
under two conditionings — the exact forced token history the pipeline fed it,
and the canonical tokenization of the same bytes — and measure the divergence
of its next-step predictions at every chunk boundary. Writes one ``.npz`` +
``.json`` pair per rollout to ``--cache-dir`` (browsed by ``app``) and prints
the summary that answers whether forced-segmentation history corruption is
material. See ``.agents/projects/20260715_xtok_advisor_drift_viz_plan.md``.

``--slug`` resolves the GSM8K sweep's step paths from the executor graph;
run on a single >=16 GB CUDA GPU:

    python -m experiments.downstream_scaling.evals.tools.xtok_drift.scoring \\
        --slug 1e22 --weights 0.4 --num-rollouts 5 --cache-dir ~/xtok_drift_cache
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.tools.xtok_drift.rollouts import (
    DEFAULT_EXECUTOR_PREFIX,
    Rollout,
    decoded_boundaries,
    load_rollouts,
    resolve_step_paths,
    verify_token_paths,
)

# Advisor pin and selector parameters, mirroring
# run_delphi_gsm8k_joint_decode_avg_xtok.py (the sweep being probed).
ADVISOR_MODEL = "Qwen/Qwen3-4B-Base"
ADVISOR_REVISION = "906bfd4"
TOP_K_B = 64
PREFIX_CREDIT = 1.0

# Top entries stored per boundary and conditioning, for the app's tables.
TOPK_STORE = 256
# Null check: identical ids evaluated in two differently-batched forwards may
# differ by kernel nondeterminism; more than this is a replay bug, not numerics.
COINCIDE_KL_MAX = 1e-2


@dataclass(frozen=True)
class Comparison:
    """Divergence between one forced/canonical logit pair (fp32, full vocab)."""

    kl_canonical_forced: float
    kl_forced_canonical: float
    entropy_forced: float
    entropy_canonical: float
    topk_forced_ids: np.ndarray
    topk_forced_logprobs: np.ndarray
    topk_canonical_ids: np.ndarray
    topk_canonical_logprobs: np.ndarray


@dataclass(frozen=True)
class ScoredBoundary:
    step_index: int
    byte_offset: int
    coincide: bool
    comparison: Comparison
    # NaN when the next commit is EOS or the rollout ends here.
    prefix_mass_forced: float
    prefix_mass_canonical: float


@dataclass(frozen=True)
class ScoredRollout:
    rollout: Rollout
    boundaries: list[ScoredBoundary]
    skipped_offsets: list[int]


def load_advisor(model: str, revision: str, device: str) -> tuple[Any, Any]:
    """Load tokenizer + model; ``model`` may be the HF repo id or a local path
    (``revision`` ignored for local paths)."""
    kwargs = {} if os.path.isdir(model) else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kwargs).to(device)
    lm.eval()
    return tokenizer, lm


def token_label(token_id: int, vocab: xtok_selection.Vocab, tokenizer: Any) -> str:
    piece = vocab.token_bytes[token_id] if token_id < len(vocab.token_bytes) else None
    if piece is not None:
        return repr(piece)
    return str(tokenizer.convert_ids_to_tokens(token_id))


def compare_logits(forced: torch.Tensor, canonical: torch.Tensor) -> Comparison:
    log_f = torch.log_softmax(forced.float(), dim=-1)
    log_c = torch.log_softmax(canonical.float(), dim=-1)
    p_f, p_c = log_f.exp(), log_c.exp()
    top_f = torch.topk(log_f, TOPK_STORE)
    top_c = torch.topk(log_c, TOPK_STORE)
    return Comparison(
        kl_canonical_forced=float(torch.sum(p_c * (log_c - log_f))),
        kl_forced_canonical=float(torch.sum(p_f * (log_f - log_c))),
        entropy_forced=float(-torch.sum(p_f * log_f)),
        entropy_canonical=float(-torch.sum(p_c * log_c)),
        topk_forced_ids=top_f.indices.numpy().astype(np.int32),
        topk_forced_logprobs=top_f.values.numpy().astype(np.float32),
        topk_canonical_ids=top_c.indices.numpy().astype(np.int32),
        topk_canonical_logprobs=top_c.values.numpy().astype(np.float32),
    )


def committed_prefix_mass(chunk: bytes, logits: torch.Tensor, vocab: xtok_selection.Vocab) -> float:
    """The selector-visible quantity: P(next text starts with ``chunk``) from
    the top-``TOP_K_B`` raw logits, through the production candidates ->
    softmax -> prefix_mass path (mirrors ``select_avg_anchored``)."""
    values, ids = torch.topk(logits, TOP_K_B)
    rows = [
        {"token_id": token_id, "logit": value} for token_id, value in zip(ids.tolist(), values.tolist(), strict=True)
    ]
    cands = xtok_selection.candidates(vocab, rows)
    max_logit = max(candidate.logit for candidate in cands.values())
    exps = {key: math.exp(candidate.logit - max_logit) for key, candidate in cands.items()}
    total = sum(exps.values())
    probs = {key: value / total for key, value in exps.items()}
    return xtok_selection.prefix_mass(chunk, probs, credit=PREFIX_CREDIT)


def last_position_logits(model: Any, sequences: list[list[int]], *, batch_size: int) -> torch.Tensor:
    """Final-real-token logits for each id sequence, ``[len(sequences), vocab]``
    on CPU in fp32. Left-padded so ``logits_to_keep=1`` reads every row's true
    last token; explicit position_ids keep rotary positions unshifted."""
    device = model.device
    out: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            mask[row, width - len(ids) :] = 1
        position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
                position_ids=position_ids.to(device),
                logits_to_keep=1,
            ).logits[:, -1, :]
        out.append(logits.float().cpu())
    return torch.cat(out, dim=0)


def forced_boundary_logits(
    model: Any, prompt_ids: list[int], rollout: Rollout, boundary_steps: list[int]
) -> torch.Tensor:
    """One teacher-forced pass over the full forced sequence; row i is the
    next-token logits after 1-based step ``boundary_steps[i]``."""
    ids = list(prompt_ids)
    position_after_step: dict[int, int] = {}
    for step_index, step in enumerate(rollout.steps, start=1):
        ids.extend(step.tokens_b)
        position_after_step[step_index] = len(ids) - 1
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], dtype=torch.long, device=model.device)).logits[0]
    return logits[[position_after_step[k] for k in boundary_steps]].float().cpu()


def score_rollout(
    model: Any,
    tokenizer: Any,
    vocab_b: xtok_selection.Vocab,
    rollout: Rollout,
    *,
    batch_size: int,
) -> ScoredRollout:
    # Bare encode: parity with the production worker's prompt tokenization
    # (worker_loop.py in the pinned joint-decode package).
    prompt_ids = tokenizer.encode(rollout.prompt)
    boundaries, skipped = decoded_boundaries(rollout)
    if not boundaries:
        raise ValueError(f"{rollout.problem_id} completion {rollout.completion_index}: every boundary is mid-codepoint")

    forced_prefix_by_step: dict[int, list[int]] = {}
    forced_gen: list[int] = []
    for step_index, step in enumerate(rollout.steps, start=1):
        forced_gen = forced_gen + list(step.tokens_b)
        forced_prefix_by_step[step_index] = forced_gen

    canonical_gen = [tokenizer.encode(boundary.text, add_special_tokens=False) for boundary in boundaries]
    forced_rows = forced_boundary_logits(model, prompt_ids, rollout, [b.step_index for b in boundaries])
    canonical_rows = last_position_logits(model, [prompt_ids + gen for gen in canonical_gen], batch_size=batch_size)

    scored: list[ScoredBoundary] = []
    for i, boundary in enumerate(boundaries):
        coincide = canonical_gen[i] == forced_prefix_by_step[boundary.step_index]
        comparison = compare_logits(forced_rows[i], canonical_rows[i])
        worst = max(comparison.kl_canonical_forced, comparison.kl_forced_canonical)
        if coincide and worst > COINCIDE_KL_MAX:
            raise RuntimeError(
                f"null check failed at step {boundary.step_index} of {rollout.problem_id} "
                f"completion {rollout.completion_index}: KL={worst:.4f} nats on identical ids — replay bug"
            )
        next_step = rollout.steps[boundary.step_index] if boundary.step_index < len(rollout.steps) else None
        if next_step is None:
            mass_forced = mass_canonical = float("nan")
        else:
            mass_forced = committed_prefix_mass(next_step.chunk, forced_rows[i], vocab_b)
            mass_canonical = committed_prefix_mass(next_step.chunk, canonical_rows[i], vocab_b)
        scored.append(
            ScoredBoundary(
                step_index=boundary.step_index,
                byte_offset=boundary.byte_offset,
                coincide=coincide,
                comparison=comparison,
                prefix_mass_forced=mass_forced,
                prefix_mass_canonical=mass_canonical,
            )
        )
    return ScoredRollout(rollout=rollout, boundaries=scored, skipped_offsets=skipped)


def score_chunked_prefix(
    model: Any,
    tokenizer: Any,
    vocab_b: xtok_selection.Vocab,
    prompt: str,
    chunks: list[bytes],
) -> tuple[Comparison, bool]:
    """The app's counterfactual probe: pipeline-style forcing of an arbitrary
    chunking (greedy segmentation per chunk) vs canonical encoding of the same
    text. Returns the comparison and whether the two id sequences coincide."""
    forced_gen = [token_id for chunk in chunks for token_id in xtok_selection.segment(vocab_b, chunk)]
    text = b"".join(chunks).decode("utf-8")
    canonical_gen = tokenizer.encode(text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt)
    rows = last_position_logits(model, [prompt_ids + forced_gen, prompt_ids + canonical_gen], batch_size=2)
    return compare_logits(rows[0], rows[1]), forced_gen == canonical_gen


def cache_key(rollout: Rollout) -> str:
    return f"{rollout.problem_id.replace('/', '-')}__w{rollout.advisor_weight:g}__i{rollout.completion_index}"


def save_scored(
    cache_dir: str,
    scored: ScoredRollout,
    *,
    model_name: str,
    revision: str,
    vocab_b: xtok_selection.Vocab,
    tokenizer: Any,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(scored.rollout)
    bounds = scored.boundaries
    arrays = {
        "step_index": np.array([b.step_index for b in bounds], dtype=np.int32),
        "byte_offset": np.array([b.byte_offset for b in bounds], dtype=np.int32),
        "coincide": np.array([b.coincide for b in bounds], dtype=bool),
        "kl_canonical_forced": np.array([b.comparison.kl_canonical_forced for b in bounds], dtype=np.float32),
        "kl_forced_canonical": np.array([b.comparison.kl_forced_canonical for b in bounds], dtype=np.float32),
        "entropy_forced": np.array([b.comparison.entropy_forced for b in bounds], dtype=np.float32),
        "entropy_canonical": np.array([b.comparison.entropy_canonical for b in bounds], dtype=np.float32),
        "prefix_mass_forced": np.array([b.prefix_mass_forced for b in bounds], dtype=np.float32),
        "prefix_mass_canonical": np.array([b.prefix_mass_canonical for b in bounds], dtype=np.float32),
        "topk_forced_ids": np.stack([b.comparison.topk_forced_ids for b in bounds]),
        "topk_forced_logprobs": np.stack([b.comparison.topk_forced_logprobs for b in bounds]),
        "topk_canonical_ids": np.stack([b.comparison.topk_canonical_ids for b in bounds]),
        "topk_canonical_logprobs": np.stack([b.comparison.topk_canonical_logprobs for b in bounds]),
    }
    np.savez_compressed(os.path.join(cache_dir, key + ".npz"), **arrays)

    # Labels for every token id appearing in the stored top-ks, so the app can
    # browse a cache without loading the tokenizer or the model.
    shown_ids = np.unique(np.concatenate([arrays["topk_forced_ids"], arrays["topk_canonical_ids"]], axis=None))
    labels = {str(int(token_id)): token_label(int(token_id), vocab_b, tokenizer) for token_id in shown_ids}
    meta = {
        "problem_id": scored.rollout.problem_id,
        "advisor_weight": scored.rollout.advisor_weight,
        "completion_index": scored.rollout.completion_index,
        "prompt": scored.rollout.prompt,
        "chunks_hex": [step.chunk.hex() for step in scored.rollout.steps],
        "tokens_b": [list(step.tokens_b) for step in scored.rollout.steps],
        "ended_with_eos": scored.rollout.ended_with_eos,
        "skipped_offsets": scored.skipped_offsets,
        "token_labels": labels,
        "model": model_name,
        "revision": revision,
    }
    with open(os.path.join(cache_dir, key + ".json"), "w") as f:
        json.dump(meta, f)
    return key


def print_summary(scored_rollouts: list[ScoredRollout]) -> None:
    pooled: list[float] = []
    print()
    print(
        f"{'rollout':<52} {'bnds':>5} {'coinc':>6} {'skip':>5} "
        f"{'KLmean':>8} {'KLp50':>8} {'KLp90':>8} {'coincmax':>9}"
    )
    for scored in scored_rollouts:
        drifted = [b.comparison.kl_canonical_forced for b in scored.boundaries if not b.coincide]
        coinciding = [b.comparison.kl_canonical_forced for b in scored.boundaries if b.coincide]
        pooled.extend(drifted)
        if drifted:
            stats = f"{np.mean(drifted):8.4f} {np.percentile(drifted, 50):8.4f} {np.percentile(drifted, 90):8.4f}"
        else:
            stats = f"{'-':>8} {'-':>8} {'-':>8}"
        coincmax = f"{max(coinciding):9.4f}" if coinciding else f"{'-':>9}"
        print(
            f"{cache_key(scored.rollout):<52} {len(scored.boundaries):>5} {len(coinciding):>6} "
            f"{len(scored.skipped_offsets):>5} {stats} {coincmax}"
        )
    if pooled:
        print(
            f"\nnon-coinciding boundaries pooled: n={len(pooled)} mean={np.mean(pooled):.4f} "
            f"p50={np.percentile(pooled, 50):.4f} p90={np.percentile(pooled, 90):.4f} nats KL(canonical||forced)"
        )
    else:
        print("\nall boundaries coincide: the forced histories were canonical everywhere (no drift exposure)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", help="Delphi checkpoint key (e.g. 1e22); resolves the sweep's step paths")
    parser.add_argument("--step-output", help="completions step dir; overrides --slug resolution (with --prompts)")
    parser.add_argument("--prompts", help="prompts step dir; overrides --slug resolution (with --step-output)")
    parser.add_argument(
        "--prefix", default=DEFAULT_EXECUTOR_PREFIX, help="executor prefix the sweep ran under (--slug resolution)"
    )
    parser.add_argument("--weights", nargs="+", type=float, required=True, help="advisor weights to select")
    parser.add_argument("--problems", nargs="+", default=None, help="prompt ids; default all problems")
    parser.add_argument(
        "--samples", nargs="+", type=int, default=[0], help="sample ranks within each (problem, weight) group"
    )
    parser.add_argument("--num-rollouts", type=int, default=None, help="keep only the first N selected rollouts")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default=ADVISOR_MODEL, help="HF repo id or local model path")
    parser.add_argument("--revision", default=ADVISOR_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="canonical-pass sequences per forward")
    args = parser.parse_args()
    explicit = args.step_output is not None or args.prompts is not None
    if args.slug is not None and explicit:
        parser.error("--slug and --step-output/--prompts are mutually exclusive")
    if args.slug is None and (args.step_output is None or args.prompts is None):
        parser.error("pass --slug, or both --step-output and --prompts")
    return args


def main() -> None:
    args = parse_args()
    if args.slug is not None:
        completions_output, prompts_output = resolve_step_paths(args.slug, prefix=args.prefix)
        print(f"resolved slug {args.slug}:\n  completions: {completions_output}\n  prompts:     {prompts_output}")
    else:
        completions_output, prompts_output = args.step_output, args.prompts
    rollouts = load_rollouts(
        completions_output,
        prompts_output=prompts_output,
        advisor_weights=args.weights,
        sample_ranks=args.samples,
        problem_ids=args.problems,
        limit=args.num_rollouts,
        cache_dir=args.cache_dir,
    )
    print(f"loaded {len(rollouts)} rollouts from {completions_output}")
    tokenizer, model = load_advisor(args.model, args.revision, args.device)
    vocab_b = xtok_selection.load_vocab(tokenizer)
    scored_rollouts: list[ScoredRollout] = []
    for rollout in rollouts:
        verify_token_paths(rollout, vocab_b)
        scored = score_rollout(model, tokenizer, vocab_b, rollout, batch_size=args.batch_size)
        key = save_scored(
            args.cache_dir, scored, model_name=args.model, revision=args.revision, vocab_b=vocab_b, tokenizer=tokenizer
        )
        print(f"scored {key}: {len(scored.boundaries)} boundaries, {len(scored.skipped_offsets)} skipped")
        scored_rollouts.append(scored)
    print_summary(scored_rollouts)


if __name__ == "__main__":
    main()
