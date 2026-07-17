# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score recorded same-tokenizer A/B rollouts for the blend visualizer."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.tools.blend_viz.rollouts import (
    DEFAULT_EXECUTOR_PREFIX,
    Rollout,
    load_rollouts,
    resolve_step_paths,
    verify_shared_tokenizer_path,
)

logger = logging.getLogger(__name__)

ADVISOR_MODEL = "meta-llama/Llama-3.1-8B"
ADVISOR_REVISION = "d04e592"
DECODER_REVISION = "main"
TOP_K_A = 16
TOP_K_B = 16
TEMPERATURE = 0.4
TOPK_STORE = 256
CACHE_DIR_ENV_VAR = "BLEND_VIZ_CACHE"

ROW_TOKEN = 0
ROW_EOS = 1
ROW_CUT = 2


@dataclass(frozen=True)
class ScoredRows:
    row_kind: np.ndarray
    committed_ids: np.ndarray
    a_topk_ids: np.ndarray
    a_topk_logprobs: np.ndarray
    b_logprobs_at_a_topk: np.ndarray
    b_topk_ids: np.ndarray
    b_topk_logprobs: np.ndarray
    a_logprobs_at_b_topk: np.ndarray
    a_entropy: np.ndarray
    b_entropy: np.ndarray
    a_committed_logprob: np.ndarray
    b_committed_logprob: np.ndarray
    a_committed_rank: np.ndarray
    b_committed_rank: np.ndarray
    kl_a_b: np.ndarray


def resolve_cache_dir(flag_value: str | None) -> str:
    if flag_value is not None:
        return flag_value
    cache_dir = os.environ.get(CACHE_DIR_ENV_VAR)
    if cache_dir is None:
        raise ValueError(f"no cache dir: pass --cache-dir or set {CACHE_DIR_ENV_VAR}")
    return cache_dir


def _pretrained_kwargs(model: str, revision: str) -> dict[str, str]:
    return {} if os.path.isdir(model) else {"revision": revision}


def load_models(
    decoder_model: str,
    advisor_model: str,
    *,
    device: str,
    decoder_revision: str = DECODER_REVISION,
    advisor_revision: str = ADVISOR_REVISION,
) -> tuple[Any, xtok_selection.Vocab, Any, Any]:
    """Load the shared tokenizer and both resident causal LMs."""
    tokenizer_a = AutoTokenizer.from_pretrained(decoder_model, **_pretrained_kwargs(decoder_model, decoder_revision))
    tokenizer_b = AutoTokenizer.from_pretrained(advisor_model, **_pretrained_kwargs(advisor_model, advisor_revision))
    if tokenizer_a.get_vocab() != tokenizer_b.get_vocab():
        raise ValueError("decoder and advisor tokenizers do not have identical vocabularies")
    vocab = xtok_selection.load_vocab(tokenizer_a)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        dtype=dtype,
        **_pretrained_kwargs(decoder_model, decoder_revision),
    ).to(device)
    advisor = AutoModelForCausalLM.from_pretrained(
        advisor_model,
        dtype=dtype,
        **_pretrained_kwargs(advisor_model, advisor_revision),
    ).to(device)
    decoder.eval()
    advisor.eval()
    return tokenizer_a, vocab, decoder, advisor


def token_label(token_id: int, vocab: xtok_selection.Vocab, tokenizer: Any) -> str:
    piece = vocab.token_bytes[token_id] if token_id < len(vocab.token_bytes) else None
    if piece is not None:
        return repr(piece)
    return str(tokenizer.convert_ids_to_tokens(token_id))


def _score_logit_rows(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    committed_ids: list[int],
    *,
    eos_token_id: int,
    topk_store: int,
) -> ScoredRows:
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"A/B logit shapes differ: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}")
    if logits_a.ndim != 2 or logits_a.shape[0] != len(committed_ids):
        raise ValueError(f"expected {len(committed_ids)} logit rows, got {tuple(logits_a.shape)}")

    stored_k = min(topk_store, logits_a.shape[1])
    a_topk_ids: list[np.ndarray] = []
    a_topk_logprobs: list[np.ndarray] = []
    b_at_a: list[np.ndarray] = []
    b_topk_ids: list[np.ndarray] = []
    b_topk_logprobs: list[np.ndarray] = []
    a_at_b: list[np.ndarray] = []
    a_entropy: list[float] = []
    b_entropy: list[float] = []
    a_committed_logprob: list[float] = []
    b_committed_logprob: list[float] = []
    a_committed_rank: list[int] = []
    b_committed_rank: list[int] = []
    kl_a_b: list[float] = []

    for row_a, row_b, committed_id in zip(logits_a, logits_b, committed_ids, strict=True):
        log_a = torch.log_softmax(row_a.float(), dim=-1)
        log_b = torch.log_softmax(row_b.float(), dim=-1)
        prob_a = log_a.exp()
        prob_b = log_b.exp()
        top_a = torch.topk(log_a, stored_k)
        top_b = torch.topk(log_b, stored_k)

        a_topk_ids.append(top_a.indices.numpy().astype(np.int32))
        a_topk_logprobs.append(top_a.values.numpy().astype(np.float32))
        b_at_a.append(log_b[top_a.indices].numpy().astype(np.float32))
        b_topk_ids.append(top_b.indices.numpy().astype(np.int32))
        b_topk_logprobs.append(top_b.values.numpy().astype(np.float32))
        a_at_b.append(log_a[top_b.indices].numpy().astype(np.float32))
        a_entropy.append(float(-torch.sum(prob_a * log_a)))
        b_entropy.append(float(-torch.sum(prob_b * log_b)))
        kl_a_b.append(float(torch.sum(prob_a * (log_a - log_b))))

        if committed_id < 0:
            a_committed_logprob.append(float("nan"))
            b_committed_logprob.append(float("nan"))
            a_committed_rank.append(-1)
            b_committed_rank.append(-1)
        else:
            committed_a = log_a[committed_id]
            committed_b = log_b[committed_id]
            a_committed_logprob.append(float(committed_a))
            b_committed_logprob.append(float(committed_b))
            a_committed_rank.append(1 + int(torch.sum(log_a > committed_a)))
            b_committed_rank.append(1 + int(torch.sum(log_b > committed_b)))

    ids = np.array(committed_ids, dtype=np.int32)
    row_kind = np.full(len(ids), ROW_TOKEN, dtype=np.uint8)
    row_kind[ids == eos_token_id] = ROW_EOS
    row_kind[ids < 0] = ROW_CUT
    return ScoredRows(
        row_kind=row_kind,
        committed_ids=ids,
        a_topk_ids=np.stack(a_topk_ids),
        a_topk_logprobs=np.stack(a_topk_logprobs),
        b_logprobs_at_a_topk=np.stack(b_at_a),
        b_topk_ids=np.stack(b_topk_ids),
        b_topk_logprobs=np.stack(b_topk_logprobs),
        a_logprobs_at_b_topk=np.stack(a_at_b),
        a_entropy=np.array(a_entropy, dtype=np.float32),
        b_entropy=np.array(b_entropy, dtype=np.float32),
        a_committed_logprob=np.array(a_committed_logprob, dtype=np.float32),
        b_committed_logprob=np.array(b_committed_logprob, dtype=np.float32),
        a_committed_rank=np.array(a_committed_rank, dtype=np.int32),
        b_committed_rank=np.array(b_committed_rank, dtype=np.int32),
        kl_a_b=np.array(kl_a_b, dtype=np.float32),
    )


def score_decision_rows(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    prompt_length: int,
    committed_ids: list[int],
    eos_token_id: int,
    topk_store: int = TOPK_STORE,
) -> ScoredRows:
    """Select decision positions from full-sequence logits and score them."""
    if prompt_length < 1:
        raise ValueError("the encoded prompt must contain at least one token")
    start = prompt_length - 1
    stop = start + len(committed_ids)
    if stop > logits_a.shape[0] or stop > logits_b.shape[0]:
        raise ValueError(f"decision rows [{start}:{stop}] exceed the available sequence logits")
    return _score_logit_rows(
        logits_a[start:stop],
        logits_b[start:stop],
        committed_ids,
        eos_token_id=eos_token_id,
        topk_store=topk_store,
    )


def _forward_logits(model: Any, input_ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([input_ids], dtype=torch.long, device=model.device)).logits[0]
    return logits.cpu()


def _forward_last_logits(model: Any, input_ids: list[int]) -> torch.Tensor:
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_tensor, logits_to_keep=1).logits
    return logits.squeeze(1).float().cpu()


def score_rollout(decoder: Any, advisor: Any, tokenizer: Any, rollout: Rollout) -> ScoredRows:
    prompt_ids = tokenizer.encode(rollout.prompt)
    recorded_ids = [step.tokens_a[0] for step in rollout.steps]
    input_ids = prompt_ids + recorded_ids
    committed_ids = [*recorded_ids, tokenizer.eos_token_id if rollout.ended_with_eos else -1]
    logits_a = _forward_logits(decoder, input_ids)
    logits_b = _forward_logits(advisor, input_ids)
    return score_decision_rows(
        logits_a,
        logits_b,
        prompt_length=len(prompt_ids),
        committed_ids=committed_ids,
        eos_token_id=tokenizer.eos_token_id,
    )


def score_probe(
    decoder: Any,
    advisor: Any,
    tokenizer: Any,
    prompt: str,
    prefix: str,
    *,
    recorded_prefix_ids: list[int] | None,
) -> ScoredRows:
    prefix_ids = (
        tokenizer.encode(prefix, add_special_tokens=False) if recorded_prefix_ids is None else recorded_prefix_ids
    )
    input_ids = tokenizer.encode(prompt) + prefix_ids
    if not input_ids:
        raise ValueError("the encoded prompt and prefix are empty")
    logits_a = _forward_last_logits(decoder, input_ids)
    logits_b = _forward_last_logits(advisor, input_ids)
    return _score_logit_rows(
        logits_a,
        logits_b,
        [-1],
        eos_token_id=tokenizer.eos_token_id,
        topk_store=TOPK_STORE,
    )


def _arrays(scored: ScoredRows) -> dict[str, np.ndarray]:
    return {field: np.asarray(getattr(scored, field)) for field in scored.__dataclass_fields__}


def _validate_arrays(arrays: dict[str, np.ndarray], expected_rows: int) -> None:
    bad = {name: value.shape for name, value in arrays.items() if value.shape[0] != expected_rows}
    if bad:
        raise ValueError(f"stored arrays do not have {expected_rows} rows: {bad}")
    if arrays["a_topk_ids"].shape != arrays["a_topk_logprobs"].shape:
        raise ValueError("A top-k id and logprob shapes differ")
    if arrays["b_topk_ids"].shape != arrays["b_topk_logprobs"].shape:
        raise ValueError("B top-k id and logprob shapes differ")
    if arrays["b_logprobs_at_a_topk"].shape != arrays["a_topk_ids"].shape:
        raise ValueError("B-at-A cross-logprob shape differs from A top-k")
    if arrays["a_logprobs_at_b_topk"].shape != arrays["b_topk_ids"].shape:
        raise ValueError("A-at-B cross-logprob shape differs from B top-k")


def _cache_key(decoder_model: str, rollout: Rollout) -> str:
    decoder = os.path.basename(decoder_model.rstrip("/")).replace("/", "-")
    problem = rollout.problem_id.replace("/", "-")
    return f"{decoder}__{problem}__w{rollout.advisor_weight:g}__r{rollout.sample_rank}"


def _token_metadata(
    arrays: dict[str, np.ndarray], vocab: xtok_selection.Vocab, tokenizer: Any
) -> tuple[dict[str, str], dict[str, str]]:
    token_ids = set(arrays["committed_ids"][arrays["committed_ids"] >= 0].tolist())
    token_ids.update(arrays["a_topk_ids"].ravel().tolist())
    token_ids.update(arrays["b_topk_ids"].ravel().tolist())
    token_bytes_hex: dict[str, str] = {}
    labels: dict[str, str] = {}
    for token_id in sorted(token_ids):
        labels[str(token_id)] = token_label(token_id, vocab, tokenizer)
        if token_id == vocab.eos_id:
            continue
        piece = vocab.token_bytes[token_id] if token_id < len(vocab.token_bytes) else None
        if piece is not None:
            token_bytes_hex[str(token_id)] = piece.hex()
    return token_bytes_hex, labels


def save_scored(
    cache_dir: str,
    rollout: Rollout,
    scored: ScoredRows,
    *,
    decoder_model: str,
    advisor_model: str,
    tokenizer: Any,
    vocab: xtok_selection.Vocab,
) -> str:
    arrays = _arrays(scored)
    _validate_arrays(arrays, len(rollout.steps) + 1)
    token_bytes_hex, labels = _token_metadata(arrays, vocab, tokenizer)
    key = _cache_key(decoder_model, rollout)
    np.savez_compressed(os.path.join(cache_dir, key + ".npz"), **arrays)
    metadata = {
        "problem_id": rollout.problem_id,
        "completion_index": rollout.completion_index,
        "sample_rank": rollout.sample_rank,
        "advisor_weight": rollout.advisor_weight,
        "prompt": rollout.prompt,
        "chunks_hex": [step.chunk.hex() for step in rollout.steps],
        "passed": rollout.passed,
        "finish_reason": rollout.finish_reason,
        "ended_with_eos": rollout.ended_with_eos,
        "decoder_model": decoder_model,
        "decoder_revision": DECODER_REVISION,
        "advisor_model": advisor_model,
        "advisor_revision": ADVISOR_REVISION,
        "top_k_a": TOP_K_A,
        "top_k_b": TOP_K_B,
        "temperature": TEMPERATURE,
        "stored_top_k": TOPK_STORE,
        "eos_token_id": vocab.eos_id,
        "token_bytes_hex": token_bytes_hex,
        "token_labels": labels,
    }
    with open(os.path.join(cache_dir, key + ".json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return key


def _row_candidates(
    ids: np.ndarray, logprobs: np.ndarray, vocab: xtok_selection.Vocab, k: int
) -> dict[xtok_selection.Key, xtok_selection.Candidate]:
    entries = [
        {"token_id": int(token_id), "logit": float(logprob)}
        for token_id, logprob in zip(ids[:k], logprobs[:k], strict=True)
    ]
    return xtok_selection.candidates(vocab, entries)


def replay_miss_fraction(scored: ScoredRows, vocab: xtok_selection.Vocab) -> float:
    misses = 0
    committed = 0
    for row, token_id in enumerate(scored.committed_ids):
        if token_id < 0:
            continue
        a = _row_candidates(scored.a_topk_ids[row], scored.a_topk_logprobs[row], vocab, TOP_K_A)
        b = _row_candidates(scored.b_topk_ids[row], scored.b_topk_logprobs[row], vocab, TOP_K_B)
        key: xtok_selection.Key = xtok_selection.EOS_KEY if token_id == vocab.eos_id else vocab.token_bytes[token_id]
        committed += 1
        misses += key not in a and key not in b
    return misses / committed if committed else 0.0


def _clear_results(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for suffix in ("*.npz", "*.json"):
        for path in glob.glob(os.path.join(cache_dir, suffix)):
            os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug")
    parser.add_argument("--step-output")
    parser.add_argument("--prompts")
    parser.add_argument("--grades")
    parser.add_argument("--prefix", default=DEFAULT_EXECUTOR_PREFIX)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--problems", nargs="+", default=None)
    parser.add_argument("--samples", nargs="+", type=int, default=[0])
    parser.add_argument("--grade-filter", choices=("pass", "fail"))
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--cache-dir", default=None, help=f"cache dir; default ${CACHE_DIR_ENV_VAR}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoder-model")
    parser.add_argument("--advisor-model", default=ADVISOR_MODEL)
    args = parser.parse_args()

    explicit = any(value is not None for value in (args.step_output, args.prompts, args.grades))
    if args.slug is not None and explicit:
        parser.error("use --slug or explicit --step-output/--prompts/--grades, not both")
    if args.slug is None and not all(value is not None for value in (args.step_output, args.prompts, args.grades)):
        parser.error("pass --slug or all of --step-output, --prompts, and --grades")
    if args.slug is None and args.decoder_model is None:
        parser.error("--decoder-model is required with explicit input paths")

    if args.slug is not None:
        step_output, prompts, grades = resolve_step_paths(args.slug, prefix=args.prefix)
        if args.decoder_model is None:
            from experiments.downstream_scaling.models.delphi import DELPHI_HF_REPOS  # noqa: PLC0415

            try:
                args.decoder_model = DELPHI_HF_REPOS[args.slug]
            except KeyError:
                parser.error(f"unknown Delphi slug {args.slug!r}; known: {sorted(DELPHI_HF_REPOS)}")
    else:
        step_output, prompts, grades = args.step_output, args.prompts, args.grades

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cache_dir = resolve_cache_dir(args.cache_dir)
    rollouts = load_rollouts(
        step_output,
        prompts_output=prompts,
        grades_output=grades,
        advisor_weights=args.weights,
        sample_ranks=args.samples,
        problem_ids=args.problems,
        grade_filter=None if args.grade_filter is None else args.grade_filter == "pass",
        limit=args.num_rollouts,
        cache_dir=cache_dir,
    )
    if not rollouts:
        raise ValueError("no rollouts remain after selection and grade filtering")

    tokenizer, vocab, decoder, advisor = load_models(
        args.decoder_model,
        args.advisor_model,
        device=args.device,
    )
    _clear_results(cache_dir)
    for rollout in rollouts:
        verify_shared_tokenizer_path(rollout, vocab)
        scored = score_rollout(decoder, advisor, tokenizer, rollout)
        key = save_scored(
            cache_dir,
            rollout,
            scored,
            decoder_model=args.decoder_model,
            advisor_model=args.advisor_model,
            tokenizer=tokenizer,
            vocab=vocab,
        )
        miss_fraction = replay_miss_fraction(scored, vocab)
        log = logger.warning if miss_fraction > 0.05 else logger.info
        log("%s: replay miss fraction %.1f%%", key, 100.0 * miss_fraction)


if __name__ == "__main__":
    main()
