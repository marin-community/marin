# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

r"""Generate Snowball tokens, retain serving routes, and replay them in training.

Run from the repository root with four Explicit CPU devices:
  JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python lib/levanter/benchmarks/snowball_rollout_replay.py \
    --backend cpu --architecture mixed --output report.json

Architecture "nope" (default) retains the one-layer full-causal NoPE control;
"mixed" uses the model's two-layer schedule: half-RoPE/windowed, then NoPE/full
causal. CPU reference fixtures use B4/S8/P3+5/page4/D8, with mixed window4.
They report observed mismatches and do not qualify native zero-KL arithmetic.

Parent-only native recipe: four local TPUv4 devices, --backend tpu and either
architecture. B4/S256/P127+129 sampled/page128/E2/top1/H128/I32/Q2/KV1/D128/V23,
BF16 KV, fixed selectors, highest precision, and mixed window128/rotary64.
Only prompt tokens are supplied: categorical cached generation retains every
serving route, including the final sampled input's unsampled output. Unchanged
weights replay those routes through public training on both physical page maps.
Native acceptance requires zero bit mismatches for full-vocabulary predictions,
sampled-token logprobs and all output rows on both maps; execution alone is not
acceptance. No checkpoints, downloads, optimizer steps or RL updates are used.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from enum import StrEnum
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.snowball_rollout_replay import generate_with_route_captures, replay_generated_routes
from levanter.models.snowball import SnowballLMHeadModel, long_attention_layer_mask
from snowball_zero_kl import (
    MODEL_SEED,
    TOKEN_SEED,
    BenchmarkArgs,
    _array_metrics,
    _fingerprint,
    _logit_metrics,
    _metadata,
    _selector_metadata,
    make_fixture,
    qualified_config,
)

BASE_OID = "1b3d5e5cb02c9306657f8aae17b76979c7b4bbec"
SAMPLING_SEED = 44017


class Architecture(StrEnum):
    NOPE = "nope"
    MIXED = "mixed"


@eqx.filter_jit
def _sampled_log_probs(logits, tokens):
    return jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tokens[..., None], axis=-1)[..., 0]


def _prediction_metrics(serving: np.ndarray, teacher: np.ndarray, tokens: np.ndarray) -> dict:
    metrics = _logit_metrics(serving, teacher)
    metrics["serving_kl_to_replay"] = metrics.pop("teacher_kl")
    metrics["serving_weighted_cross_entropy"] = metrics.pop("teacher_weighted_cross_entropy")
    serving_log_probs = np.asarray(_sampled_log_probs(jnp.asarray(serving), jnp.asarray(tokens)))
    teacher_log_probs = np.asarray(_sampled_log_probs(jnp.asarray(teacher), jnp.asarray(tokens)))
    metrics["sampled_token_log_probs_fp32"] = _array_metrics(serving_log_probs, teacher_log_probs)
    return metrics


def run(report: dict, *, backend: str, source_sha: str | None, architecture: Architecture = Architecture.NOPE) -> None:
    """Populate a bounded report; the CLI retains it even after an execution failure."""
    if backend != jax.default_backend():
        raise RuntimeError(f"requested {backend} but runtime is {jax.default_backend()}")
    if backend == "tpu" and (
        jax.device_count() != 4
        or jax.local_device_count() != 4
        or jax.process_count() != 1
        or any("tpu v4" not in device.device_kind.lower() for device in jax.devices())
    ):
        raise RuntimeError("native qualification requires exactly four local TPUv4 devices in one process")
    total_length, prompt_length, page_size = (256, 127, 128) if backend == "tpu" else (8, 3, 4)
    config = qualified_config(implementation="fixed_schedule" if backend == "tpu" else "native", backend=backend)
    config = dataclasses.replace(
        config,
        num_layers=2 if architecture == Architecture.MIXED else 1,
        sliding_window=page_size,
    )
    if backend == "cpu":
        # Reference arithmetic with top-k routing; native selectors stay unchanged.
        config = dataclasses.replace(
            config,
            max_seq_len=total_length,
            hidden_dim=16,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            num_experts=4,
            num_experts_per_token=2,
        )
    fixture = make_fixture(sequence_length=total_length, page_size=page_size)
    mesh = compact_grug_mesh(expert_axis_size=1)
    metadata = _metadata(BenchmarkArgs(source_sha=source_sha), backend=backend, mesh=mesh)
    layer_schedule = [
        {
            "layer": layer,
            "attention": "nope_full_causal" if is_long else "half_rope_windowed_causal",
            "window": None if is_long else config.sliding_window,
            "rotary_dim": 0 if is_long else config.inferred_head_dim // 2,
            "unrotated_dim": config.inferred_head_dim if is_long else config.inferred_head_dim // 2,
            "qk_mult": config.qk_mult,
            "attention_scale": config.inferred_head_dim**-0.5,
            "q_layout": ["batch", "position", "query_head", "head_dim"],
            "kv_layout": ["batch", "position", "kv_head", "head_dim"],
            "cache_layout": ["physical_page", "page_offset", "interleaved_key_value_heads", "head_dim"],
        }
        for layer, is_long in enumerate(np.asarray(long_attention_layer_mask(config.num_layers)))
    ]
    metadata.update(
        architecture=architecture.value,
        layer_schedule=layer_schedule,
        page_size=page_size,
        attention_gate="unmodified initialized weights; 2*sigmoid(0)=1 (active attention)",
        schema="snowball_generated_route_training_replay.v1",
        base_oid=BASE_OID,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        api_sha256=hashlib.sha256(
            (Path(__file__).parents[1] / "src/levanter/inference/snowball_rollout_replay.py").read_bytes()
        ).hexdigest(),
        fixture=(
            f"{'TPUv4-fixed' if backend == 'tpu' else 'CPU-reference'}"
            f"/B4/S{total_length}/{architecture.value}/{config.num_layers}layers"
            f"/E{config.num_experts}/top{config.num_experts_per_token}"
            f"/H{config.hidden_dim}/I{config.intermediate_dim}"
            f"/Q{config.num_heads}/KV{config.num_kv_heads}/D{config.inferred_head_dim}/V{config.vocab_size}"
        ),
        sampling_seed=SAMPLING_SEED,
        prompt_seed=TOKEN_SEED,
        model_seed=MODEL_SEED,
        prompt_length=prompt_length,
        generated_tokens=total_length - prompt_length,
        sampling_rule="temperature-one categorical; fold_in(sampling_key, continuation_index)",
        predictor_positions=list(range(prompt_length - 1, total_length - 1)),
        extra_unsampled_output_position=total_length - 1,
        row_order="batch-major teacher tokens; prefill then decoded sampled tokens within each slot",
        timing="one synchronized hosted generation and replay per map, includes compilation; no throughput claim",
    )
    metadata.pop("warmup")
    metadata.pop("measured")
    report.update(
        metadata=metadata,
        config=dataclasses.asdict(config),
        selectors=_selector_metadata(config),
        page_maps=[page_map.tolist() for page_map in fixture.page_maps],
    )
    results, teachers = [], []
    with (
        jax.set_mesh(mesh),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}),
        jax.default_matmul_precision("highest"),
    ):
        report["metadata"]["default_matmul_precision"] = str(jax.config.jax_default_matmul_precision)
        report["metadata"]["mesh_axis_types"] = [str(axis) for axis in mesh.axis_types]
        model = SnowballLMHeadModel.init(hax.Axis("vocab", config.vocab_size), config, key=jax.random.key(MODEL_SEED))
        parameter_digest = _fingerprint(model)
        report["parameter_sha256_before"] = parameter_digest
        report["attention_gates"] = [
            {
                "layer": layer,
                "parameter_sha256": _fingerprint(block.attn.attn_gate),
                "nonzero_weights": int(np.count_nonzero(np.asarray(block.attn.attn_gate))),
            }
            for layer, block in enumerate(model.transformer.blocks)
        ]
        prompts = hax.named(jnp.asarray(np.asarray(fixture.tokens.array)[:, :prompt_length]), ("batch", "position"))
        report["prompt_sha256"] = _fingerprint(prompts)
        for index, page_map in enumerate(fixture.page_maps):
            row = {"page_map_index": index, "phase": "generation", "status": "running"}
            report["runs"].append(row)
            started = time.perf_counter()
            generated = generate_with_route_captures(
                model,
                prompts,
                sampling_key=jax.random.key(SAMPLING_SEED),
                generated_tokens=total_length - prompt_length,
                page_map=page_map,
                page_size=page_size,
                cache_dtype=jnp.bfloat16,
            )
            row.update(
                phase="teacher_replay",
                generation_time=time.perf_counter() - started,
                parameter_sha256=_fingerprint(model),
                generated_token_ids=generated.sampled_token_ids.tolist(),
                token_sha256=_fingerprint(generated.tokens),
                route_sha256=_fingerprint(generated.teacher_capture),
                serving_logits_sha256=_fingerprint(generated.serving_logits),
                capture_provenance={
                    "producer": "SnowballLMHeadModel.decode_with_route_captures",
                    "teacher_recapture": False,
                    "serving_calls": len(generated.serving_captures),
                    "captured_token_rows_per_layer": total_length * 4,
                    "layers": config.num_layers,
                    "per_layer_route_sha256": [
                        _fingerprint(capture) for capture in generated.teacher_capture.route_captures
                    ],
                    "top_k": config.num_experts_per_token,
                    "call_route_sha256": [
                        _fingerprint(capture.route_captures) for capture in generated.serving_captures
                    ],
                    "binding": "returned serving token/position/validity metadata checked against exact generated sequence",
                },
            )
            started = time.perf_counter()
            teacher = replay_generated_routes(model, generated)
            row["teacher_replay_time"] = time.perf_counter() - started
            # All accuracy slicing is on completed host arrays, including Explicit meshes.
            if not np.isfinite(generated.serving_logits).all() or not np.isfinite(teacher).all():
                raise FloatingPointError("non-finite serving or teacher logits")
            predictors = generated.predictor_positions
            row.update(
                phase="complete",
                status="success",
                predictions=_prediction_metrics(
                    generated.prediction_logits, teacher[:, predictors], generated.sampled_token_ids
                ),
                all_output_rows=_logit_metrics(generated.serving_logits, teacher),
                extra_unsampled_next_row=_logit_metrics(generated.unsampled_next_logits, teacher[:, -1]),
                teacher_logits_sha256=_fingerprint(teacher),
                parameter_sha256_after_replay=_fingerprint(model),
            )
            results.append(generated)
            teachers.append(teacher)
        report["parameter_sha256_after"] = _fingerprint(model)
    if report["parameter_sha256_after"] != parameter_digest:
        raise AssertionError("generation/replay changed parameter bytes")
    left, right = results
    report["cross_page_maps"] = {
        "sampled_ids_equal": bool(np.array_equal(left.sampled_token_ids, right.sampled_token_ids)),
        "tokens_equal": bool(np.array_equal(np.asarray(left.tokens.array), np.asarray(right.tokens.array))),
        "routes_equal": _fingerprint(left.teacher_capture) == _fingerprint(right.teacher_capture),
        "serving_logits": _logit_metrics(left.serving_logits, right.serving_logits),
        "replay_logits": _logit_metrics(teachers[0], teachers[1]),
    }
    if not all(row["predictions"]["finite"] and row["all_output_rows"]["finite"] for row in report["runs"]):
        raise FloatingPointError("non-finite serving or teacher logits")
    report["prediction_parity"] = (
        "exact" if all(row["predictions"]["bitwise_fp32"] for row in report["runs"]) else "different"
    )
    report["native_qualification"] = {
        "status": "observed" if backend == "tpu" else "unexecuted",
        "reason": (
            "bounded fixed-schedule generation/replay executed"
            if backend == "tpu"
            else "CPU reference controls do not qualify TPU arithmetic"
        ),
    }
    report["status"] = "success"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--architecture", type=Architecture, choices=tuple(Architecture), default=Architecture.NOPE)
    parser.add_argument("--source-sha", help="Exact frozen source commit for a snapshot without Git metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {"status": "running", "runs": [], "native_qualification": {"status": "unexecuted"}}
    try:
        run(report, backend=args.backend, source_sha=args.source_sha, architecture=args.architecture)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        if report["runs"] and report["runs"][-1]["status"] == "running":
            report["runs"][-1]["status"] = "failed"
        raise
    finally:
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(output)
        else:
            args.output.write_text(output)


if __name__ == "__main__":
    main()
