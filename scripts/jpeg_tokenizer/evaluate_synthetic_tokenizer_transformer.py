#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic tokenizer benchmark with a tiny causal Transformer."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass

import flax.linen as nn
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from experiments.jpeg_tokenizer.base.eval import summarize_metric
from scripts.jpeg_tokenizer import evaluate_synthetic_tokenizer_mechanisms as synth

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    num_layers: int
    num_heads: int
    mlp_mult: int
    learning_rate: float
    steps: int
    batch_size: int
    seed: int


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_mult: int

    @nn.compact
    def __call__(self, x: jax.Array, *, mask: jax.Array) -> jax.Array:
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            deterministic=True,
        )(y, mask=mask)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_model * self.mlp_mult)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        return x + y


class TinyCausalTransformer(nn.Module):
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    mlp_mult: int

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input seq_len {seq_len} exceeds model max_seq_len {self.max_seq_len}")

        tok_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, name="token_embed")(tokens)
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.d_model),
        )
        x = tok_embed + pos_embed[:seq_len][None, :, :]
        mask = nn.attention.make_causal_mask(jnp.ones((batch_size, seq_len), dtype=jnp.bool_))
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_mult=self.mlp_mult,
            )(x, mask=mask)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train-sequences", type=int, default=3000)
    parser.add_argument("--num-eval-sequences", type=int, default=512)
    parser.add_argument("--events-per-sequence", type=int, default=512)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--mode-switch-prob", type=float, default=0.005)
    parser.add_argument("--value-copy-prob", type=float, default=0.75)
    parser.add_argument("--value-step-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--variants", default="run_joint,run_shared,run_shared_reset")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--perturb-fractions", default="0.5")
    parser.add_argument("--perturb-kind", choices=["any", "value_only", "mode_only"], default="any")
    parser.add_argument("--horizons", default="1,8,32,128")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="artifacts/jpeg_tokenizer/analysis/synthetic_tokenizer_transformer",
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


def _is_mode_token(variant: synth.TokenizerVariant, token: int, data_cfg: synth.SyntheticConfig) -> bool:
    if variant.name == "flat_joint":
        return False
    return token < data_cfg.num_modes


def _select_perturb_position(
    sequence: np.ndarray,
    *,
    fraction: float,
    variant: synth.TokenizerVariant,
    data_cfg: synth.SyntheticConfig,
    perturb_kind: str,
) -> int:
    base = synth._perturb_position(len(sequence), fraction)
    if perturb_kind == "any":
        return base

    want_mode = perturb_kind == "mode_only"

    def predicate(pos: int) -> bool:
        token = int(sequence[pos])
        return _is_mode_token(variant, token, data_cfg) == want_mode

    if predicate(base):
        return base

    for radius in range(1, len(sequence)):
        left = base - radius
        right = base + radius
        if left >= 0 and predicate(left):
            return left
        if right < len(sequence) and predicate(right):
            return right
    return base


def _pad_sequences(sequences: list[np.ndarray], *, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int32)
    max_len = int(lengths.max())
    padded = np.full((len(sequences), max_len), pad_id, dtype=np.int32)
    for idx, seq in enumerate(sequences):
        padded[idx, : len(seq)] = seq
    return padded, lengths


def _build_model_and_state(
    cfg: TransformerConfig,
    *,
    vocab_size: int,
    max_seq_len: int,
) -> tuple[TinyCausalTransformer, train_state.TrainState]:
    model = TinyCausalTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        mlp_mult=cfg.mlp_mult,
    )
    init_tokens = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(cfg.seed), init_tokens)
    tx = optax.adamw(learning_rate=cfg.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def _compute_per_position_nll(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = tokens[:, 1:]
    gathered = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -gathered


def _loss_mask_from_lengths(lengths: jax.Array, *, seq_len: int) -> jax.Array:
    positions = jnp.arange(seq_len - 1, dtype=jnp.int32)[None, :]
    return (positions < (lengths[:, None] - 1)).astype(jnp.float32)


def _train_variant(
    model: TinyCausalTransformer,
    state: train_state.TrainState,
    tokens: np.ndarray,
    lengths: np.ndarray,
    cfg: TransformerConfig,
    *,
    log_every: int,
) -> train_state.TrainState:
    @jax.jit
    def train_step(step_state: train_state.TrainState, batch_tokens: jax.Array, batch_lengths: jax.Array):
        seq_len = batch_tokens.shape[1]

        def loss_fn(params):
            logits = model.apply(params, batch_tokens)
            per_pos_nll = _compute_per_position_nll(logits, batch_tokens)
            mask = _loss_mask_from_lengths(batch_lengths, seq_len=seq_len)
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            loss = jnp.sum(per_pos_nll * mask) / denom
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(step_state.params)
        next_state = step_state.apply_gradients(grads=grads)
        return next_state, loss

    rng = np.random.default_rng(cfg.seed + 17)
    num_examples = tokens.shape[0]
    for step in range(cfg.steps):
        idx = rng.integers(0, num_examples, size=cfg.batch_size)
        batch_tokens = tokens[idx]
        batch_lengths = lengths[idx]
        state, loss = train_step(
            state,
            jnp.asarray(batch_tokens, dtype=jnp.int32),
            jnp.asarray(batch_lengths, dtype=jnp.int32),
        )
        if (step + 1) % log_every == 0:
            logger.info("train step %s/%s loss=%.4f", step + 1, cfg.steps, float(loss))
    return state


def _evaluate_losses(
    model: TinyCausalTransformer,
    params: dict[str, object],
    tokens: np.ndarray,
    lengths: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    @jax.jit
    def eval_step(batch_tokens: jax.Array) -> jax.Array:
        logits = model.apply(params, batch_tokens)
        return _compute_per_position_nll(logits, batch_tokens)

    per_seq_losses: list[np.ndarray] = []
    for start in range(0, tokens.shape[0], batch_size):
        end = min(tokens.shape[0], start + batch_size)
        batch_tokens = tokens[start:end]
        batch_lengths = lengths[start:end]
        per_pos = np.asarray(eval_step(jnp.asarray(batch_tokens, dtype=jnp.int32)), dtype=np.float64)
        mask = np.asarray(
            _loss_mask_from_lengths(jnp.asarray(batch_lengths), seq_len=batch_tokens.shape[1]),
            dtype=np.float64,
        )
        per_seq = np.sum(per_pos * mask, axis=1, dtype=np.float64)
        per_seq_losses.append(per_seq)
    return np.concatenate(per_seq_losses)


def _evaluate_variant(
    variant: synth.TokenizerVariant,
    data_cfg: synth.SyntheticConfig,
    train_cfg: TransformerConfig,
    *,
    train_joint_sequences: list[np.ndarray],
    eval_joint_sequences: list[np.ndarray],
    perturb_fractions: list[float],
    perturb_kind: str,
    horizons: list[int],
    log_every: int,
) -> dict[str, object]:
    train_seqs = [synth._encode_for_variant(variant, seq, data_cfg) for seq in train_joint_sequences]
    eval_seqs = [synth._encode_for_variant(variant, seq, data_cfg) for seq in eval_joint_sequences]

    model_vocab_size = variant.vocab_size + 1
    pad_id = variant.vocab_size
    train_tokens, train_lengths = _pad_sequences(train_seqs, pad_id=pad_id)
    eval_tokens, eval_lengths = _pad_sequences(eval_seqs, pad_id=pad_id)

    model, state = _build_model_and_state(
        train_cfg,
        vocab_size=model_vocab_size,
        max_seq_len=train_tokens.shape[1],
    )
    state = _train_variant(model, state, train_tokens, train_lengths, train_cfg, log_every=log_every)
    clean_nats = _evaluate_losses(model, state.params, eval_tokens, eval_lengths, batch_size=train_cfg.batch_size)

    @jax.jit
    def one_seq_loss(tokens_row: jax.Array) -> jax.Array:
        logits = model.apply(state.params, tokens_row[None, :])
        return _compute_per_position_nll(logits, tokens_row[None, :])[0]

    perturbation_metrics: dict[float, dict[str, list[float]]] = {fraction: {} for fraction in perturb_fractions}
    semantic_metrics: dict[float, dict[str, list[float]]] = {fraction: {} for fraction in perturb_fractions}
    for fraction in perturb_fractions:
        perturbation_metrics[fraction] = {
            "delta_total_nats": [],
            "delta_immediate_nats": [],
        }
        semantic_metrics[fraction] = {
            "tail_event_hamming": [],
            "tail_length_change": [],
            "tail_exact_match": [],
        }
        for horizon in horizons:
            perturbation_metrics[fraction][f"delta_h{horizon}_nats"] = []
            perturbation_metrics[fraction][f"delta_tail_h{horizon}_nats"] = []

    for idx, seq in enumerate(eval_seqs):
        clean_per_pos = np.asarray(one_seq_loss(jnp.asarray(seq, dtype=jnp.int32)), dtype=np.float64)
        clean_decoded = synth._decode_for_variant(variant, seq, data_cfg)

        for fraction in perturb_fractions:
            perturb_pos = _select_perturb_position(
                seq,
                fraction=fraction,
                variant=variant,
                data_cfg=data_cfg,
                perturb_kind=perturb_kind,
            )
            pert = np.array(seq, copy=True)
            pert[perturb_pos] = synth._replacement_token_id(int(pert[perturb_pos]), vocab_size=variant.vocab_size)
            pert_per_pos = np.asarray(one_seq_loss(jnp.asarray(pert, dtype=jnp.int32)), dtype=np.float64)
            delta = pert_per_pos - clean_per_pos

            immediate_index = min(perturb_pos, len(delta) - 1)
            perturbation_metrics[fraction]["delta_total_nats"].append(float(np.sum(delta, dtype=np.float64)))
            perturbation_metrics[fraction]["delta_immediate_nats"].append(float(delta[immediate_index]))

            for horizon in horizons:
                lo = immediate_index
                hi = min(len(delta), immediate_index + horizon)
                tail_lo = min(len(delta), immediate_index + 1)
                perturbation_metrics[fraction][f"delta_h{horizon}_nats"].append(
                    float(np.sum(delta[lo:hi], dtype=np.float64))
                )
                perturbation_metrics[fraction][f"delta_tail_h{horizon}_nats"].append(
                    float(np.sum(delta[tail_lo:hi], dtype=np.float64))
                )

            pert_decoded = synth._decode_for_variant(variant, pert, data_cfg)
            start_event = int(clean_decoded.events_before_token[perturb_pos])
            clean_tail = clean_decoded.joint_events[start_event:]
            pert_tail = pert_decoded.joint_events[start_event:]
            aligned = min(len(clean_tail), len(pert_tail))
            if aligned <= 0:
                tail_hamming = 0.0
                tail_exact = 1.0
            else:
                tail_hamming = float(np.mean(clean_tail[:aligned] != pert_tail[:aligned]))
                tail_exact = 1.0 if len(clean_tail) == len(pert_tail) and np.array_equal(clean_tail, pert_tail) else 0.0
            tail_length_change = abs(len(clean_tail) - len(pert_tail)) / max(1, len(clean_tail))
            semantic_metrics[fraction]["tail_event_hamming"].append(tail_hamming)
            semantic_metrics[fraction]["tail_length_change"].append(float(tail_length_change))
            semantic_metrics[fraction]["tail_exact_match"].append(tail_exact)

        if (idx + 1) % log_every == 0:
            logger.info("eval perturb %s %s/%s", variant.name, idx + 1, len(eval_seqs))

    result: dict[str, object] = {
        "name": variant.name,
        "vocab_size": variant.vocab_size,
        "reset_interval": variant.reset_interval,
        "tokens_per_sequence": summarize_metric([float(len(s)) for s in eval_seqs]).to_dict(),
        "clean": _summary(clean_nats),
        "perturbation": {},
    }
    for fraction in perturb_fractions:
        pm = perturbation_metrics[fraction]
        sm = semantic_metrics[fraction]
        fraction_result: dict[str, object] = {
            "delta_total": _summary(np.asarray(pm["delta_total_nats"], dtype=np.float64)),
            "delta_immediate": _summary(np.asarray(pm["delta_immediate_nats"], dtype=np.float64)),
            "semantic_tail_event_hamming": summarize_metric(sm["tail_event_hamming"]).to_dict(),
            "semantic_tail_length_change": summarize_metric(sm["tail_length_change"]).to_dict(),
            "semantic_tail_exact_match": summarize_metric(sm["tail_exact_match"]).to_dict(),
        }
        for horizon in horizons:
            fraction_result[f"delta_h{horizon}"] = _summary(np.asarray(pm[f"delta_h{horizon}_nats"], dtype=np.float64))
            fraction_result[f"delta_tail_h{horizon}"] = _summary(
                np.asarray(pm[f"delta_tail_h{horizon}_nats"], dtype=np.float64)
            )
        result["perturbation"][str(fraction)] = fraction_result
    return result


def _render_summary(results: list[dict[str, object]], *, perturb_fractions: list[float], horizons: list[int]) -> str:
    lines = [
        "# Synthetic Tokenizer Transformer Benchmark",
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
        lines.append("| Run | Delta Total | Delta Immediate | Semantic Tail Hamming |")
        lines.append("| --- | ---: | ---: | ---: |")
        for run in results:
            metrics = run["perturbation"][str(fraction)]  # type: ignore[index]
            total_bits = metrics["delta_total"]["bits_per_sequence"]["mean"]  # type: ignore[index]
            immediate_bits = metrics["delta_immediate"]["bits_per_sequence"]["mean"]  # type: ignore[index]
            semantic_hamming = metrics["semantic_tail_event_hamming"]["mean"]  # type: ignore[index]
            lines.append(f"| `{run['name']}` | {total_bits:.2f} | {immediate_bits:.2f} | {semantic_hamming:.3f} |")

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


def _select_variants(
    all_variants: list[synth.TokenizerVariant],
    variant_names: list[str],
    reset_interval: int,
) -> list[synth.TokenizerVariant]:
    selected: list[synth.TokenizerVariant] = []
    for name in variant_names:
        stripped = name.strip()
        if stripped == "run_shared_reset":
            target_name = f"run_shared_reset_{reset_interval}"
        else:
            target_name = stripped
        matched = next((variant for variant in all_variants if variant.name == target_name), None)
        if matched is None:
            raise ValueError(f"Unknown variant {stripped!r}; available {[v.name for v in all_variants]}")
        selected.append(matched)
    return selected


def main(args: argparse.Namespace | None = None) -> None:
    parsed = parse_args() if args is None else args
    perturb_fractions = [float(v) for v in parsed.perturb_fractions.split(",") if v]
    horizons = [int(v) for v in parsed.horizons.split(",") if v]
    if not perturb_fractions:
        raise ValueError("At least one perturb fraction is required")
    if not horizons:
        raise ValueError("At least one horizon is required")

    data_cfg = synth.SyntheticConfig(
        num_modes=parsed.num_modes,
        num_values=parsed.num_values,
        events_per_sequence=parsed.events_per_sequence,
        mode_switch_prob=parsed.mode_switch_prob,
        value_copy_prob=parsed.value_copy_prob,
        value_step_prob=parsed.value_step_prob,
        seed=parsed.seed,
    )
    train_cfg = TransformerConfig(
        d_model=parsed.d_model,
        num_layers=parsed.num_layers,
        num_heads=parsed.num_heads,
        mlp_mult=parsed.mlp_mult,
        learning_rate=parsed.learning_rate,
        steps=parsed.steps,
        batch_size=parsed.batch_size,
        seed=parsed.seed,
    )

    all_variants = synth._build_variants(data_cfg, parsed.reset_interval)
    variants = _select_variants(all_variants, parsed.variants.split(","), parsed.reset_interval)
    train_joint = synth._generate_corpus(data_cfg, num_sequences=parsed.num_train_sequences, seed_offset=101)
    eval_joint = synth._generate_corpus(data_cfg, num_sequences=parsed.num_eval_sequences, seed_offset=202)

    results: list[dict[str, object]] = []
    for variant in variants:
        logger.info("Evaluating transformer synthetic variant %s", variant.name)
        results.append(
            _evaluate_variant(
                variant,
                data_cfg,
                train_cfg,
                train_joint_sequences=train_joint,
                eval_joint_sequences=eval_joint,
                perturb_fractions=perturb_fractions,
                perturb_kind=parsed.perturb_kind,
                horizons=horizons,
                log_every=parsed.log_every,
            )
        )

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
            "seed": parsed.seed,
            "reset_interval": parsed.reset_interval,
            "variants": [v.name for v in variants],
            "steps": parsed.steps,
            "batch_size": parsed.batch_size,
            "learning_rate": parsed.learning_rate,
            "d_model": parsed.d_model,
            "num_layers": parsed.num_layers,
            "num_heads": parsed.num_heads,
            "mlp_mult": parsed.mlp_mult,
            "perturb_fractions": perturb_fractions,
            "perturb_kind": parsed.perturb_kind,
            "horizons": horizons,
        },
        "runs": results,
    }
    out_prefix = parsed.output_dir.rstrip("/")
    _write_json(f"{out_prefix}/synthetic_transformer_eval.json", payload)
    _write_text(
        f"{out_prefix}/summary.md",
        _render_summary(results, perturb_fractions=perturb_fractions, horizons=horizons),
    )
    logger.info("Wrote outputs to %s", out_prefix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
