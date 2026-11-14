# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate Snowball tokens, retain serving routes, and replay them in training.

CPU control (four Explicit devices):
  JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python snowball_rollout_replay.py --backend cpu --output report.json

Parent-only native recipe: four local TPUv4 devices, --backend tpu. The fixed
B4/S256/NoPE fixture uses prompt127 +129 sampled tokens, page128 and BF16 KV.
No checkpoints, data downloads, optimizer steps or production serving are used.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.snowball_rollout_replay import generate_with_route_captures, replay_generated_routes
from levanter.models.snowball import SnowballLMHeadModel
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

BASE_OID = "077b269560e5545c559c67fa921ac897c5efdbc9"
SAMPLING_SEED = 44017


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


def run(report: dict, *, backend: str, source_sha: str | None) -> None:
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
    if backend == "cpu":
        # Smaller reference fixture, with multiple layers and top-k for row-order coverage.
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
            num_layers=2,
        )
    fixture = make_fixture(sequence_length=total_length, page_size=page_size)
    mesh = compact_grug_mesh(expert_axis_size=1)
    metadata = _metadata(BenchmarkArgs(source_sha=source_sha), backend=backend, mesh=mesh)
    metadata.update(
        schema="snowball_generated_route_training_replay.v1",
        base_oid=BASE_OID,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        api_sha256=hashlib.sha256(
            (Path(__file__).parents[1] / "src/levanter/inference/snowball_rollout_replay.py").read_bytes()
        ).hexdigest(),
        fixture=(
            "B4/S256/oneNoPElayer/E2/top1/H128/I32/Q2/KV1/D128/V23"
            if backend == "tpu"
            else "CPU-reference-B4/S8/two-layers/E4/top2/H16/I16/Q2/KV1/D8/V23"
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
    parser.add_argument("--source-sha", help="Exact frozen source commit for a snapshot without Git metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {"status": "running", "runs": [], "native_qualification": {"status": "unexecuted"}}
    try:
        run(report, backend=args.backend, source_sha=args.source_sha)
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
