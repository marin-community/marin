#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic benchmark for tokenizer mechanism isolation.

This script builds a controlled synthetic source, then compares tokenizers that
change one mechanism at a time:

- fixed-semantic vs context-dependent value tokens
- unbounded mode-state vs periodic reset-bounded mode-state

Metrics are sequence-level only:

- clean whole-sequence loss (bits/sequence)
- prefix-corruption amplification deltas (bits/sequence)
- decoded event-space corruption amplification
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import fsspec
import numpy as np

from experiments.jpeg_tokenizer.base.eval import summarize_metric

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticConfig:
    num_modes: int
    num_values: int
    events_per_sequence: int
    mode_switch_prob: float
    value_copy_prob: float
    value_step_prob: float
    seed: int


@dataclass(frozen=True)
class DecodedSequence:
    joint_events: np.ndarray
    events_before_token: np.ndarray


@dataclass(frozen=True)
class TokenizerVariant:
    name: str
    vocab_size: int
    reset_interval: int | None


class NGramModel:
    """Fixed-order add-alpha n-gram LM."""

    def __init__(self, *, vocab_size: int, order: int, alpha: float) -> None:
        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
        if order <= 0:
            raise ValueError(f"order must be > 0, got {order}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.vocab_size = vocab_size
        self.order = order
        self.alpha = alpha
        self.bos_token = vocab_size
        self._totals: dict[tuple[int, ...], int] = {}
        self._counts: dict[tuple[int, ...], dict[int, int]] = {}

    def fit(self, sequences: Iterable[np.ndarray]) -> None:
        for sequence in sequences:
            context = [self.bos_token] * self.order
            for raw_token in sequence.tolist():
                token = int(raw_token)
                key = tuple(context)
                next_counts = self._counts.get(key)
                if next_counts is None:
                    next_counts = {}
                    self._counts[key] = next_counts
                    self._totals[key] = 0
                next_counts[token] = next_counts.get(token, 0) + 1
                self._totals[key] += 1
                context = [*context[1:], token]

    def loss_per_position(self, sequence: np.ndarray) -> np.ndarray:
        context = [self.bos_token] * self.order
        losses = np.zeros(len(sequence), dtype=np.float64)
        normalizer_floor = self.alpha * float(self.vocab_size)
        for idx, raw_token in enumerate(sequence.tolist()):
            token = int(raw_token)
            key = tuple(context)
            total = self._totals.get(key, 0)
            next_counts = self._counts.get(key)
            count = 0 if next_counts is None else next_counts.get(token, 0)
            prob = (count + self.alpha) / (float(total) + normalizer_floor)
            losses[idx] = -math.log(prob)
            context = [*context[1:], token]
        return losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train-sequences", type=int, default=3000)
    parser.add_argument("--num-eval-sequences", type=int, default=512)
    parser.add_argument("--events-per-sequence", type=int, default=512)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--mode-switch-prob", type=float, default=0.03)
    parser.add_argument("--value-copy-prob", type=float, default=0.75)
    parser.add_argument("--value-step-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ngram-order", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--perturb-fractions", default="0.5")
    parser.add_argument("--horizons", default="1,8,32,128")
    parser.add_argument("--log-every", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default="artifacts/jpeg_tokenizer/analysis/synthetic_tokenizer_mechanisms",
    )
    return parser.parse_args()


def _write_json(path: str, payload: object) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: str, text: str) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        handle.write(text)


def _summary(values_nats: np.ndarray) -> dict[str, object]:
    values_bits = values_nats / math.log(2.0)
    return {
        "nats_per_sequence": summarize_metric(values_nats.tolist()).to_dict(),
        "bits_per_sequence": summarize_metric(values_bits.tolist()).to_dict(),
    }


def _replacement_token_id(token: int, *, vocab_size: int) -> int:
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
    return (token + 1) % vocab_size


def _perturb_position(sequence_length: int, fraction: float) -> int:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return math.floor((sequence_length - 1) * fraction)


def _sample_mode_transitions(config: SyntheticConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    return rng.dirichlet(np.ones(config.num_values), size=config.num_modes).astype(np.float64)


def _sample_sequence(config: SyntheticConfig, mode_value_probs: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    mode = int(rng.integers(config.num_modes))
    value = int(rng.integers(config.num_values))
    joint = np.zeros(config.events_per_sequence, dtype=np.int32)
    for idx in range(config.events_per_sequence):
        if idx > 0 and float(rng.random()) < config.mode_switch_prob:
            mode = int(rng.integers(config.num_modes - 1))
            if mode >= (joint[idx - 1] // config.num_values):
                mode += 1

        if idx > 0 and float(rng.random()) < config.value_copy_prob:
            value = int(joint[idx - 1] % config.num_values)
        elif idx > 0 and float(rng.random()) < config.value_step_prob:
            step = -1 if float(rng.random()) < 0.5 else 1
            value = (int(joint[idx - 1] % config.num_values) + step) % config.num_values
        else:
            value = int(rng.choice(config.num_values, p=mode_value_probs[mode]))

        joint[idx] = mode * config.num_values + value
    return joint


def _generate_corpus(config: SyntheticConfig, *, num_sequences: int, seed_offset: int) -> list[np.ndarray]:
    mode_value_probs = _sample_mode_transitions(config)
    rng = np.random.default_rng(config.seed + seed_offset)
    return [_sample_sequence(config, mode_value_probs, rng=rng) for _ in range(num_sequences)]


def _encode_flat_joint(joint_events: np.ndarray) -> np.ndarray:
    return np.asarray(joint_events, dtype=np.int32)


def _encode_run_joint(joint_events: np.ndarray, *, num_modes: int, num_values: int) -> np.ndarray:
    tokens: list[int] = []
    prev_mode: int | None = None
    for joint in joint_events.tolist():
        mode = int(joint // num_values)
        if prev_mode is None or mode != prev_mode:
            tokens.append(mode)
            prev_mode = mode
        tokens.append(num_modes + int(joint))
    return np.asarray(tokens, dtype=np.int32)


def _encode_run_shared(
    joint_events: np.ndarray,
    *,
    num_modes: int,
    num_values: int,
    reset_interval: int | None,
) -> np.ndarray:
    tokens: list[int] = []
    prev_mode: int | None = None
    events_since_mode = 0
    for joint in joint_events.tolist():
        mode = int(joint // num_values)
        value = int(joint % num_values)
        needs_mode = prev_mode is None or mode != prev_mode
        if reset_interval is not None and events_since_mode >= reset_interval:
            needs_mode = True
        if needs_mode:
            tokens.append(mode)
            prev_mode = mode
            events_since_mode = 0
        tokens.append(num_modes + value)
        events_since_mode += 1
    return np.asarray(tokens, dtype=np.int32)


def _decode_flat_joint(tokens: np.ndarray) -> DecodedSequence:
    events = np.asarray(tokens, dtype=np.int32)
    events_before = np.arange(len(tokens), dtype=np.int32)
    return DecodedSequence(joint_events=events, events_before_token=events_before)


def _decode_run_joint(tokens: np.ndarray, *, num_modes: int, num_values: int) -> DecodedSequence:
    events: list[int] = []
    events_before = np.zeros(len(tokens), dtype=np.int32)
    for idx, raw_token in enumerate(tokens.tolist()):
        events_before[idx] = len(events)
        token = int(raw_token)
        if token < num_modes:
            continue
        joint = token - num_modes
        if joint < 0 or joint >= num_modes * num_values:
            joint = joint % (num_modes * num_values)
        events.append(joint)
    return DecodedSequence(joint_events=np.asarray(events, dtype=np.int32), events_before_token=events_before)


def _decode_run_shared(tokens: np.ndarray, *, num_modes: int, num_values: int) -> DecodedSequence:
    events: list[int] = []
    events_before = np.zeros(len(tokens), dtype=np.int32)
    current_mode = 0
    for idx, raw_token in enumerate(tokens.tolist()):
        events_before[idx] = len(events)
        token = int(raw_token)
        if token < num_modes:
            current_mode = token
            continue
        value = token - num_modes
        if value < 0 or value >= num_values:
            value = value % num_values
        events.append(current_mode * num_values + value)
    return DecodedSequence(joint_events=np.asarray(events, dtype=np.int32), events_before_token=events_before)


def _build_variants(config: SyntheticConfig, reset_interval: int) -> list[TokenizerVariant]:
    return [
        TokenizerVariant(
            name="flat_joint",
            vocab_size=config.num_modes * config.num_values,
            reset_interval=None,
        ),
        TokenizerVariant(
            name="run_joint",
            vocab_size=config.num_modes + (config.num_modes * config.num_values),
            reset_interval=None,
        ),
        TokenizerVariant(
            name="run_shared",
            vocab_size=config.num_modes + config.num_values,
            reset_interval=None,
        ),
        TokenizerVariant(
            name=f"run_shared_reset_{reset_interval}",
            vocab_size=config.num_modes + config.num_values,
            reset_interval=reset_interval,
        ),
    ]


def _encode_for_variant(variant: TokenizerVariant, joint_events: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    if variant.name == "flat_joint":
        return _encode_flat_joint(joint_events)
    if variant.name == "run_joint":
        return _encode_run_joint(joint_events, num_modes=config.num_modes, num_values=config.num_values)
    if variant.name.startswith("run_shared"):
        return _encode_run_shared(
            joint_events,
            num_modes=config.num_modes,
            num_values=config.num_values,
            reset_interval=variant.reset_interval,
        )
    raise ValueError(f"Unknown variant {variant.name}")


def _decode_for_variant(variant: TokenizerVariant, tokens: np.ndarray, config: SyntheticConfig) -> DecodedSequence:
    if variant.name == "flat_joint":
        return _decode_flat_joint(tokens)
    if variant.name == "run_joint":
        return _decode_run_joint(tokens, num_modes=config.num_modes, num_values=config.num_values)
    if variant.name.startswith("run_shared"):
        return _decode_run_shared(tokens, num_modes=config.num_modes, num_values=config.num_values)
    raise ValueError(f"Unknown variant {variant.name}")


def _evaluate_variant(
    variant: TokenizerVariant,
    config: SyntheticConfig,
    *,
    train_joint_sequences: list[np.ndarray],
    eval_joint_sequences: list[np.ndarray],
    ngram_order: int,
    alpha: float,
    perturb_fractions: list[float],
    horizons: list[int],
    log_every: int,
) -> dict[str, object]:
    train_token_sequences = [_encode_for_variant(variant, seq, config) for seq in train_joint_sequences]
    eval_token_sequences = [_encode_for_variant(variant, seq, config) for seq in eval_joint_sequences]

    lm = NGramModel(vocab_size=variant.vocab_size, order=ngram_order, alpha=alpha)
    lm.fit(train_token_sequences)

    token_lengths = np.asarray([len(seq) for seq in eval_token_sequences], dtype=np.float64)
    clean_sequence_losses: list[float] = []

    perturb_metrics: dict[float, dict[str, list[float]]] = {
        fraction: defaultdict(list) for fraction in perturb_fractions
    }
    semantic_metrics: dict[float, dict[str, list[float]]] = {
        fraction: defaultdict(list) for fraction in perturb_fractions
    }

    for idx, sequence in enumerate(eval_token_sequences):
        clean_loss = lm.loss_per_position(sequence)
        clean_sequence_losses.append(float(np.sum(clean_loss, dtype=np.float64)))
        clean_decoded = _decode_for_variant(variant, sequence, config)

        for fraction in perturb_fractions:
            perturb_pos = _perturb_position(len(sequence), fraction)
            perturbed = np.array(sequence, copy=True)
            perturbed[perturb_pos] = _replacement_token_id(int(perturbed[perturb_pos]), vocab_size=variant.vocab_size)

            perturbed_loss = lm.loss_per_position(perturbed)
            delta = perturbed_loss - clean_loss

            immediate_index = perturb_pos + 1
            immediate = 0.0 if immediate_index >= len(delta) else float(delta[immediate_index])
            perturb_metrics[fraction]["delta_total_nats"].append(float(np.sum(delta, dtype=np.float64)))
            perturb_metrics[fraction]["delta_immediate_nats"].append(immediate)

            for horizon in horizons:
                if horizon <= 0:
                    raise ValueError(f"horizon must be positive, got {horizon}")
                lo = immediate_index
                hi = min(len(delta), immediate_index + horizon)
                tail_lo = min(len(delta), immediate_index + 1)
                perturb_metrics[fraction][f"delta_h{horizon}_nats"].append(float(np.sum(delta[lo:hi], dtype=np.float64)))
                perturb_metrics[fraction][f"delta_tail_h{horizon}_nats"].append(
                    float(np.sum(delta[tail_lo:hi], dtype=np.float64))
                )

            perturbed_decoded = _decode_for_variant(variant, perturbed, config)
            start_event = int(clean_decoded.events_before_token[perturb_pos])
            clean_tail = clean_decoded.joint_events[start_event:]
            pert_tail = perturbed_decoded.joint_events[start_event:]
            aligned = min(len(clean_tail), len(pert_tail))
            if aligned <= 0:
                tail_hamming = 0.0
                tail_exact = 1.0
            else:
                tail_hamming = float(np.mean(clean_tail[:aligned] != pert_tail[:aligned]))
                tail_exact = 1.0 if len(clean_tail) == len(pert_tail) and np.array_equal(clean_tail, pert_tail) else 0.0
            length_change = abs(len(clean_tail) - len(pert_tail)) / max(1, len(clean_tail))
            semantic_metrics[fraction]["tail_event_hamming"].append(tail_hamming)
            semantic_metrics[fraction]["tail_length_change"].append(float(length_change))
            semantic_metrics[fraction]["tail_exact_match"].append(tail_exact)

        if (idx + 1) % log_every == 0:
            logger.info("Run %s processed %s/%s sequences", variant.name, idx + 1, len(eval_token_sequences))

    clean_nats = np.asarray(clean_sequence_losses, dtype=np.float64)
    result: dict[str, object] = {
        "name": variant.name,
        "vocab_size": variant.vocab_size,
        "reset_interval": variant.reset_interval,
        "ngram_order": ngram_order,
        "alpha": alpha,
        "num_train_sequences": len(train_token_sequences),
        "num_eval_sequences": len(eval_token_sequences),
        "tokens_per_sequence": summarize_metric(token_lengths.tolist()).to_dict(),
        "clean": _summary(clean_nats),
        "perturbation": {},
    }

    for fraction in perturb_fractions:
        metrics = perturb_metrics[fraction]
        semantic = semantic_metrics[fraction]
        fraction_result: dict[str, object] = {
            "delta_total": _summary(np.asarray(metrics["delta_total_nats"], dtype=np.float64)),
            "delta_immediate": _summary(np.asarray(metrics["delta_immediate_nats"], dtype=np.float64)),
            "semantic_tail_event_hamming": summarize_metric(semantic["tail_event_hamming"]).to_dict(),
            "semantic_tail_length_change": summarize_metric(semantic["tail_length_change"]).to_dict(),
            "semantic_tail_exact_match": summarize_metric(semantic["tail_exact_match"]).to_dict(),
        }
        for horizon in horizons:
            fraction_result[f"delta_h{horizon}"] = _summary(
                np.asarray(metrics[f"delta_h{horizon}_nats"], dtype=np.float64)
            )
            fraction_result[f"delta_tail_h{horizon}"] = _summary(
                np.asarray(metrics[f"delta_tail_h{horizon}_nats"], dtype=np.float64)
            )
        result["perturbation"][str(fraction)] = fraction_result
    return result


def _render_summary(results: list[dict[str, object]], *, perturb_fractions: list[float], horizons: list[int]) -> str:
    lines = [
        "# Synthetic Tokenizer Mechanism Benchmark",
        "",
        "All loss metrics are whole-sequence bits/sequence.",
        "",
        "## Clean Whole-Sequence Loss",
        "",
    ]
    for run in results:
        bits = run["clean"]["bits_per_sequence"]["mean"]  # type: ignore[index]
        toks = run["tokens_per_sequence"]["mean"]  # type: ignore[index]
        lines.append(f"- `{run['name']}`: mean bits/sequence = {bits:.2f}, mean tokens/sequence = {toks:.2f}")

    for fraction in perturb_fractions:
        lines.extend(["", f"## Prefix Corruption Fraction {fraction:.2f}", ""])
        lines.append("| Run | Delta Total | Delta Immediate | Semantic Tail Hamming | Tail Exact Match |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for run in results:
            metrics = run["perturbation"][str(fraction)]  # type: ignore[index]
            total_bits = metrics["delta_total"]["bits_per_sequence"]["mean"]  # type: ignore[index]
            immediate_bits = metrics["delta_immediate"]["bits_per_sequence"]["mean"]  # type: ignore[index]
            semantic_hamming = metrics["semantic_tail_event_hamming"]["mean"]  # type: ignore[index]
            tail_exact = metrics["semantic_tail_exact_match"]["mean"]  # type: ignore[index]
            lines.append(
                f"| `{run['name']}` | {total_bits:.2f} | {immediate_bits:.2f} | "
                f"{semantic_hamming:.3f} | {tail_exact:.3f} |"
            )

        lines.extend(["", "### Horizon Deltas", ""])
        lines.append("| Run | " + " | ".join([f"Delta h{h}" for h in horizons]) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(horizons)) + " |")
        for run in results:
            metrics = run["perturbation"][str(fraction)]  # type: ignore[index]
            cells = []
            for horizon in horizons:
                bits = metrics[f"delta_h{horizon}"]["bits_per_sequence"]["mean"]  # type: ignore[index]
                cells.append(f"{bits:.2f}")
            lines.append(f"| `{run['name']}` | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main(args: argparse.Namespace | None = None) -> None:
    parsed = parse_args() if args is None else args
    perturb_fractions = [float(v) for v in parsed.perturb_fractions.split(",") if v]
    horizons = [int(v) for v in parsed.horizons.split(",") if v]
    if not perturb_fractions:
        raise ValueError("At least one perturbation fraction is required")
    if not horizons:
        raise ValueError("At least one horizon is required")

    config = SyntheticConfig(
        num_modes=parsed.num_modes,
        num_values=parsed.num_values,
        events_per_sequence=parsed.events_per_sequence,
        mode_switch_prob=parsed.mode_switch_prob,
        value_copy_prob=parsed.value_copy_prob,
        value_step_prob=parsed.value_step_prob,
        seed=parsed.seed,
    )
    variants = _build_variants(config, parsed.reset_interval)
    train_joint = _generate_corpus(config, num_sequences=parsed.num_train_sequences, seed_offset=101)
    eval_joint = _generate_corpus(config, num_sequences=parsed.num_eval_sequences, seed_offset=202)

    results: list[dict[str, object]] = []
    for variant in variants:
        logger.info("Evaluating synthetic variant %s", variant.name)
        result = _evaluate_variant(
            variant,
            config,
            train_joint_sequences=train_joint,
            eval_joint_sequences=eval_joint,
            ngram_order=parsed.ngram_order,
            alpha=parsed.alpha,
            perturb_fractions=perturb_fractions,
            horizons=horizons,
            log_every=parsed.log_every,
        )
        results.append(result)

    payload = {
        "config": {
            "num_train_sequences": parsed.num_train_sequences,
            "num_eval_sequences": parsed.num_eval_sequences,
            "events_per_sequence": parsed.events_per_sequence,
            "num_modes": parsed.num_modes,
            "num_values": parsed.num_values,
            "mode_switch_prob": parsed.mode_switch_prob,
            "value_copy_prob": parsed.value_copy_prob,
            "value_step_prob": parsed.value_step_prob,
            "ngram_order": parsed.ngram_order,
            "alpha": parsed.alpha,
            "reset_interval": parsed.reset_interval,
            "seed": parsed.seed,
            "perturb_fractions": perturb_fractions,
            "horizons": horizons,
        },
        "runs": results,
    }
    out_prefix = parsed.output_dir.rstrip("/")
    _write_json(f"{out_prefix}/synthetic_mechanism_eval.json", payload)
    _write_text(
        f"{out_prefix}/summary.md",
        _render_summary(results, perturb_fractions=perturb_fractions, horizons=horizons),
    )
    logger.info("Wrote outputs to %s", out_prefix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
