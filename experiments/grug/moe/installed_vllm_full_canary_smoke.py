# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full GrugMoE canary export plus installed-vLLM correctness and serving smoke."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROMPT_IDS = [1, 42, 128, 2048, 17, 3072, 5, 63]
_DEFAULT_MODEL_SIZE = "full-canary"
_MODEL_SIZE_TO_PARITY_CONFIG = {
    "full-canary": "canary",
    "small-diagnostic": "small-diagnostic",
}
_REFERENCE_JSON_NAME = "installed_vllm_reference.json"
_DEFAULT_MAX_LOGPROB_DELTA = 5.0
_DEFAULT_ROUTING_MARGIN_TOLERANCE = 1e-2
_ROUTING_MISMATCH_PRINT_LIMIT = 20
_ROUTING_FULL_DETAILS_TOKEN_LAYER_LIMIT = 64
_ROUTING_DEBUG_VECTOR_LIMIT = 16
_ROUTING_DEBUG_ENV = "GRUGMOE_ROUTING_DEBUG"
_ROUTING_DEBUG_LAYER_ENV = "GRUGMOE_ROUTING_DEBUG_LAYER"
_ROUTING_DEBUG_TOKEN_POSITION_ENV = "GRUGMOE_ROUTING_DEBUG_TOKEN_POSITION"
_ROUTING_DEBUG_VECTOR_LIMIT_ENV = "GRUGMOE_ROUTING_DEBUG_VECTOR_LIMIT"


def _direct_url(package: str) -> str:
    direct_url = md.distribution(package).read_text("direct_url.json")
    return direct_url.strip() if direct_url else ""


def _print_runtime_header() -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")

    print("remote_cwd=" + os.getcwd())
    try:
        marin_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        marin_sha = f"unavailable:{exc!r}"
    print("marin_sha=" + marin_sha)
    for package in ("vllm", "tpu-inference"):
        print(f"{package}_direct_url=" + _direct_url(package))
    print("vllm_version=" + md.version("vllm"))
    print("tpu-inference_version=" + md.version("tpu-inference"))
    print("jax_version=" + md.version("jax"))
    print("libtpu_version=" + md.version("libtpu"))
    print("grugmoe_spec=" + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe")))


def _directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _shard_count(path: Path) -> int:
    return len(list(path.glob("*.safetensors")))


def _artifact_dir(output_dir: Path) -> Path:
    return output_dir / "grugmoe-inference"


def _reference_path(output_dir: Path) -> Path:
    return output_dir / _REFERENCE_JSON_NAME


def _artifact_vocab_size(artifact_dir: Path) -> int:
    with (artifact_dir / "config.json").open() as f:
        config = json.load(f)
    vocab_size = int(config["vocab_size"])
    if vocab_size <= 0:
        raise AssertionError(f"invalid artifact vocab_size={vocab_size}")
    return vocab_size


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["VLLM_TARGET_DEVICE"] = "tpu"
    env["MODEL_IMPL_TYPE"] = "auto"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _configure_vllm_env() -> None:
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    # Marin's generic vLLM helper defaults MODEL_IMPL_TYPE to "vllm".
    # GrugMoE is exposed by tpu-inference's JAX model registry, so force auto.
    os.environ["MODEL_IMPL_TYPE"] = "auto"
    # Keep vLLM's API server and EngineCore in one process so libtpu is claimed once.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _configure_routing_debug_env(
    *,
    enabled: bool,
    token_position: int,
    layer: int,
    vector_limit: int,
) -> None:
    keys = (
        _ROUTING_DEBUG_ENV,
        _ROUTING_DEBUG_LAYER_ENV,
        _ROUTING_DEBUG_TOKEN_POSITION_ENV,
        _ROUTING_DEBUG_VECTOR_LIMIT_ENV,
    )
    if not enabled:
        for key in keys:
            os.environ.pop(key, None)
        return
    os.environ[_ROUTING_DEBUG_ENV] = "1"
    os.environ[_ROUTING_DEBUG_LAYER_ENV] = str(layer)
    os.environ[_ROUTING_DEBUG_TOKEN_POSITION_ENV] = str(token_position)
    os.environ[_ROUTING_DEBUG_VECTOR_LIMIT_ENV] = str(vector_limit)
    print(
        "routing_debug_env="
        + json.dumps(
            {
                _ROUTING_DEBUG_ENV: os.environ[_ROUTING_DEBUG_ENV],
                _ROUTING_DEBUG_LAYER_ENV: os.environ[_ROUTING_DEBUG_LAYER_ENV],
                _ROUTING_DEBUG_TOKEN_POSITION_ENV: os.environ[_ROUTING_DEBUG_TOKEN_POSITION_ENV],
                _ROUTING_DEBUG_VECTOR_LIMIT_ENV: os.environ[_ROUTING_DEBUG_VECTOR_LIMIT_ENV],
            },
            sort_keys=True,
        )
    )


def _server_base_url(server_url: str) -> str:
    return server_url.removesuffix("/v1")


def export_artifact(
    output_dir: Path,
    *,
    model_size: str,
    max_shard_size: int,
    generation_tokens: int,
    routing_debug: bool,
    routing_debug_token_position: int,
    routing_debug_layer: int,
    routing_debug_vector_limit: int,
) -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")

    import tpu_inference.models.jax.grugmoe as tpu_grugmoe

    from experiments.grug.moe import vllm_tpu_parity as parity

    config_name = _MODEL_SIZE_TO_PARITY_CONFIG[model_size]
    print("artifact_generation_process=started")
    print("artifact_model_size=" + model_size)
    print("artifact_parity_config=" + config_name)
    parity.check_realistic_training_state_roundtrip(
        tpu_grugmoe,
        config_name=config_name,
        output_dir=output_dir,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
        reference_output_path=_reference_path(output_dir),
        routing_debug_token_position=routing_debug_token_position if routing_debug else None,
        routing_debug_layer=routing_debug_layer if routing_debug else None,
        routing_debug_vector_limit=routing_debug_vector_limit,
    )
    print("artifact_generation_process=completed")


def _extract_logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def _entry_for_token(entry: Any, token_id: int, position: int) -> Any:
    if not isinstance(entry, dict):
        raise AssertionError(f"prompt_logprobs[{position}] is not a dict: {entry!r}")
    if token_id in entry:
        return entry[token_id]
    if str(token_id) in entry:
        return entry[str(token_id)]
    for key, value in entry.items():
        try:
            if int(key) == token_id:
                return value
        except (TypeError, ValueError):
            continue
    raise AssertionError(
        f"prompt_logprobs[{position}] is missing selected token {token_id}; " f"available={list(entry)[:10]}"
    )


def _selected_prompt_logprobs(prompt_logprobs: Any, token_ids: list[int]) -> np.ndarray:
    if prompt_logprobs is None:
        raise AssertionError("installed vLLM did not return prompt_logprobs")
    if len(prompt_logprobs) != len(token_ids):
        raise AssertionError(f"prompt_logprobs length {len(prompt_logprobs)} != token_ids length {len(token_ids)}")

    selected = []
    for position in range(1, len(token_ids)):
        entry = prompt_logprobs[position]
        if entry is None:
            raise AssertionError(f"prompt_logprobs[{position}] is unexpectedly None")
        selected.append(_extract_logprob_value(_entry_for_token(entry, token_ids[position], position)))
    return np.asarray(selected, dtype=np.float64)


def _assert_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))[:10].tolist()
        raise AssertionError(f"{name} contains non-finite values at {bad}")


def _compare_logprobs(
    *,
    reference: dict[str, Any],
    vllm_continuation: np.ndarray,
    generated_token_ids: list[int],
    max_logprob_delta: float,
) -> None:
    score_token_ids = [int(token_id) for token_id in reference["score_token_ids"]]
    ref_continuation = np.asarray(reference["levanter_continuation_logprobs"], dtype=np.float64)
    positions = [int(position) for position in reference["logprob_token_positions"]]
    if len(positions) != len(ref_continuation):
        raise AssertionError(
            "reference continuation logprob metadata is inconsistent: "
            f"positions={len(positions)} logprobs={len(ref_continuation)}"
        )
    if len(generated_token_ids) != len(ref_continuation):
        raise AssertionError(
            "vLLM continuation generated-token metadata is inconsistent: "
            f"generated={len(generated_token_ids)} logprobs={len(ref_continuation)}"
        )
    if any(position <= 0 or position >= len(score_token_ids) for position in positions):
        raise AssertionError(f"reference logprob token positions are out of range: {positions}")

    vllm_continuation = np.asarray(vllm_continuation, dtype=np.float64)
    if vllm_continuation.shape != ref_continuation.shape:
        raise AssertionError(f"vLLM continuation logprobs shape {vllm_continuation.shape} != {ref_continuation.shape}")
    _assert_finite("levanter continuation logprobs", ref_continuation)
    _assert_finite("vLLM continuation logprobs", vllm_continuation)
    deltas = np.abs(vllm_continuation - ref_continuation)
    summary = {
        "count": len(ref_continuation),
        "max_abs_delta": float(np.max(deltas)) if len(deltas) else 0.0,
        "mean_abs_delta": float(np.mean(deltas)) if len(deltas) else 0.0,
        "max_allowed_abs_delta": float(max_logprob_delta),
        "tokens": [
            {
                "token_position": int(position),
                "token_id": int(score_token_ids[position]),
                "vllm_generated_token_id": int(generated_token_ids[idx]),
                "levanter_logprob": float(ref_continuation[idx]),
                "vllm_logprob": float(vllm_continuation[idx]),
                "abs_delta": float(deltas[idx]),
            }
            for idx, position in enumerate(positions)
        ],
    }
    print("logprob_summary=" + json.dumps(summary, sort_keys=True))
    if summary["max_abs_delta"] > max_logprob_delta:
        raise AssertionError(f"selected-token logprob delta {summary['max_abs_delta']} exceeds {max_logprob_delta}")


def _selected_output_logprob(logprobs: Any, token_id: int, label: str) -> float:
    if logprobs is None:
        raise AssertionError(f"{label} did not return completion logprobs")
    if len(logprobs) != 1:
        raise AssertionError(f"{label} expected one completion logprob entry, got {len(logprobs)}")
    return _extract_logprob_value(_entry_for_token(logprobs[0], token_id, 0))


def _score_continuation_logprobs(
    llm: Any,
    sampling_params_type: Any,
    reference: dict[str, Any],
    *,
    num_logprobs: int,
) -> tuple[np.ndarray, list[int]]:
    prompt_ids = [int(token_id) for token_id in reference["prompt_ids"]]
    continuation_ids = [int(token_id) for token_id in reference["continuation_ids"]]
    vllm_logprobs = []
    generated_token_ids = []
    for token_offset, target_token_id in enumerate(continuation_ids):
        prefix_token_ids = prompt_ids + continuation_ids[:token_offset]
        outputs = llm.generate(
            [prefix_token_ids],
            sampling_params_type(
                max_tokens=1,
                temperature=0.0,
                logprobs=num_logprobs,
            ),
            use_tqdm=False,
        )
        if len(outputs) != 1 or not outputs[0].outputs:
            raise AssertionError(f"expected one vLLM logprob output for target {target_token_id}, got {outputs!r}")
        completion = outputs[0].outputs[0]
        if len(completion.token_ids) != 1:
            raise AssertionError(
                f"expected one generated token for target {target_token_id}, got {completion.token_ids}"
            )
        generated_token_ids.append(int(completion.token_ids[0]))
        vllm_logprobs.append(
            _selected_output_logprob(
                completion.logprobs,
                target_token_id,
                f"continuation_position={token_offset} target={target_token_id}",
            )
        )
    return np.asarray(vllm_logprobs, dtype=np.float64), generated_token_ids


def _compare_routing(
    *,
    reference: dict[str, Any],
    routed_experts: Any,
    routing_margin_tolerance: float,
) -> None:
    if routed_experts is None:
        raise AssertionError("installed vLLM did not return routed_experts")
    expected = np.asarray(reference["levanter_routed_experts"], dtype=np.int64)
    actual = np.asarray(routed_experts, dtype=np.int64)
    margins = np.asarray(reference["levanter_router_margin"], dtype=np.float64)
    score_token_ids = [int(token_id) for token_id in reference["score_token_ids"]]
    if actual.shape != expected.shape:
        raise AssertionError(f"routed_experts shape {actual.shape} != reference shape {expected.shape}")
    if margins.shape != expected.shape[:2]:
        raise AssertionError(f"router margins shape {margins.shape} != token/layer shape {expected.shape[:2]}")
    _assert_finite("reference router margins", margins)

    top_k = int(expected.shape[2])
    token_layers = int(expected.shape[0] * expected.shape[1])
    mismatch_print_limit = (
        token_layers if token_layers <= _ROUTING_FULL_DETAILS_TOKEN_LAYER_LIMIT else _ROUTING_MISMATCH_PRINT_LIMIT
    )
    ordered_matches = np.all(actual == expected, axis=-1)
    top1_matches = actual[..., 0] == expected[..., 0]
    overlap_counts = np.zeros(expected.shape[:2], dtype=np.int64)
    mismatch_locations: list[dict[str, Any]] = []
    suspicious_locations: list[dict[str, Any]] = []
    suspicious_mismatch_count = 0
    low_margin_boundary_mismatches = 0

    for token_idx in range(expected.shape[0]):
        for layer_idx in range(expected.shape[1]):
            expected_set = set(int(value) for value in expected[token_idx, layer_idx])
            actual_set = set(int(value) for value in actual[token_idx, layer_idx])
            overlap = len(expected_set & actual_set)
            overlap_counts[token_idx, layer_idx] = overlap
            margin = float(margins[token_idx, layer_idx])
            if not bool(ordered_matches[token_idx, layer_idx]):
                item = {
                    "token_position": int(token_idx),
                    "token_id": int(score_token_ids[token_idx]),
                    "layer": int(layer_idx),
                    "levanter": expected[token_idx, layer_idx].astype(int).tolist(),
                    "vllm": actual[token_idx, layer_idx].astype(int).tolist(),
                    "unordered_overlap": int(overlap),
                    "router_margin": margin,
                    "top1_match": bool(top1_matches[token_idx, layer_idx]),
                }
                if len(mismatch_locations) < mismatch_print_limit:
                    mismatch_locations.append(item)
                if overlap < top_k and margin > routing_margin_tolerance:
                    suspicious_mismatch_count += 1
                    if len(suspicious_locations) < mismatch_print_limit:
                        suspicious_locations.append(item)
                elif overlap < top_k:
                    low_margin_boundary_mismatches += 1

    unordered_full_matches = overlap_counts == top_k
    summary = {
        "token_layer_count": token_layers,
        "top_k": top_k,
        "ordered_topk_match_count": int(np.sum(ordered_matches)),
        "ordered_topk_match_rate": float(np.sum(ordered_matches) / token_layers),
        "unordered_full_match_count": int(np.sum(unordered_full_matches)),
        "unordered_full_match_rate": float(np.sum(unordered_full_matches) / token_layers),
        "mean_unordered_overlap": float(np.sum(overlap_counts) / (token_layers * top_k)),
        "top1_match_count": int(np.sum(top1_matches)),
        "top1_match_rate": float(np.sum(top1_matches) / token_layers),
        "mismatch_count": int(token_layers - np.sum(ordered_matches)),
        "low_margin_boundary_mismatch_count": int(low_margin_boundary_mismatches),
        "suspicious_mismatch_count": int(suspicious_mismatch_count),
        "routing_margin_tolerance": float(routing_margin_tolerance),
        "routing_mismatch_print_limit": int(mismatch_print_limit),
    }
    print("routing_summary=" + json.dumps(summary, sort_keys=True))
    print("routing_mismatch_locations=" + json.dumps(mismatch_locations, sort_keys=True))
    if suspicious_mismatch_count:
        print("routing_suspicious_mismatches=" + json.dumps(suspicious_locations, sort_keys=True))
        raise AssertionError(
            f"{suspicious_mismatch_count} routed-expert mismatches had boundary margin > " f"{routing_margin_tolerance}"
        )


def _print_reported_routed_experts_debug(
    *,
    reference: dict[str, Any],
    routed_experts: Any,
    token_position: int,
    layer: int,
) -> None:
    if routed_experts is None:
        print("vllm_reported_routed_experts_debug=null")
        return
    actual = np.asarray(routed_experts, dtype=np.int64)
    score_token_ids = [int(token_id) for token_id in reference["score_token_ids"]]
    payload: dict[str, Any] = {
        "source": "vLLM CompletionOutput.routed_experts",
        "shape": list(actual.shape),
        "token_position": int(token_position),
        "layer": int(layer),
    }
    if 0 <= token_position < actual.shape[0]:
        payload["token_id"] = score_token_ids[token_position]
        payload["token_routes_all_layers"] = actual[token_position].astype(int).tolist()
        if 0 <= layer < actual.shape[1]:
            payload["reported_topk_expert_ids"] = actual[token_position, layer].astype(int).tolist()
    levanter_debug = reference.get("levanter_routing_debug")
    if isinstance(levanter_debug, dict):
        payload["levanter_topk_expert_ids"] = levanter_debug.get("topk_expert_ids")
        payload["levanter_router_margin"] = levanter_debug.get("router_margin")
    print("vllm_reported_routed_experts_debug=" + json.dumps(payload, sort_keys=True))


def score_artifact(
    artifact_dir: Path,
    reference_path: Path,
    *,
    max_logprob_delta: float,
    routing_margin_tolerance: float,
    routing_debug: bool,
    routing_debug_token_position: int,
    routing_debug_layer: int,
    routing_debug_vector_limit: int,
) -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")
    _configure_vllm_env()
    _configure_routing_debug_env(
        enabled=routing_debug,
        token_position=routing_debug_token_position,
        layer=routing_debug_layer,
        vector_limit=routing_debug_vector_limit,
    )

    from vllm import LLM, SamplingParams

    print("score_artifact_dir=" + str(artifact_dir))
    print("score_reference_json=" + str(reference_path))
    for package in ("vllm", "tpu-inference"):
        print(f"score_{package}_direct_url=" + _direct_url(package))
    print("score_grugmoe_spec=" + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe")))

    with reference_path.open() as f:
        reference = json.load(f)
    model_config = reference.get("model_config")
    if isinstance(model_config, dict):
        shape_keys = (
            "vocab_size",
            "hidden_dim",
            "intermediate_dim",
            "shared_expert_intermediate_dim",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "num_experts",
            "num_experts_per_token",
            "max_seq_len",
            "sliding_window",
        )
        shape = {key: model_config.get(key) for key in shape_keys}
        print("score_model_shape=" + json.dumps(shape, sort_keys=True))
        print("score_model_config=" + json.dumps(model_config, sort_keys=True))
    score_token_ids = [int(token_id) for token_id in reference["score_token_ids"]]
    vocab_size = _artifact_vocab_size(artifact_dir)
    max_model_len = max(16, len(score_token_ids) + 1)
    print("score_full_vocab_logprobs=" + str(vocab_size))
    llm = LLM(
        model=str(artifact_dir),
        tokenizer=str(artifact_dir),
        runner="generate",
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        max_logprobs=vocab_size,
        enable_return_routed_experts=True,
    )
    print("score_llm_initialized=True")
    routing_outputs = llm.generate(
        [score_token_ids],
        SamplingParams(max_tokens=1, temperature=0.0),
        use_tqdm=False,
    )
    if len(routing_outputs) != 1 or not routing_outputs[0].outputs:
        raise AssertionError(f"expected one vLLM score output, got {routing_outputs!r}")
    output = routing_outputs[0]
    completion = output.outputs[0]
    print("score_prompt_token_ids=" + repr(list(output.prompt_token_ids or [])))
    print("score_generated_token_ids=" + repr(list(completion.token_ids)))
    vllm_continuation_logprobs, generated_by_prefix = _score_continuation_logprobs(
        llm,
        SamplingParams,
        reference,
        num_logprobs=vocab_size,
    )
    print("score_continuation_generated_token_ids=" + repr(generated_by_prefix))
    _compare_logprobs(
        reference=reference,
        vllm_continuation=vllm_continuation_logprobs,
        generated_token_ids=generated_by_prefix,
        max_logprob_delta=max_logprob_delta,
    )
    if routing_debug:
        _print_reported_routed_experts_debug(
            reference=reference,
            routed_experts=completion.routed_experts,
            token_position=routing_debug_token_position,
            layer=routing_debug_layer,
        )
    _compare_routing(
        reference=reference,
        routed_experts=completion.routed_experts,
        routing_margin_tolerance=routing_margin_tolerance,
    )
    print("installed_path_result=works:full_canary_logprobs_and_routing")


def serve_artifact(
    artifact_dir: Path,
    *,
    model_size: str,
    generation_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
) -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")
    _configure_vllm_env()

    import requests
    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_server import VllmEnvironment

    print("serve_artifact_dir=" + str(artifact_dir))
    print("serve_model_size=" + model_size)
    for package in ("vllm", "tpu-inference"):
        print(f"serve_{package}_direct_url=" + _direct_url(package))
    print("serve_grugmoe_spec=" + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe")))

    model = ModelConfig(
        name=f"grugmoe-{model_size}",
        path=str(artifact_dir),
        engine_kwargs={
            "max_model_len": 16,
            "max_num_batched_tokens": 16,
        },
    )
    extra_args = [
        "--runner",
        "generate",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--skip-tokenizer-init",
        "--tokens-only",
    ]

    with VllmEnvironment(
        model=model,
        port=server_port,
        timeout_seconds=server_timeout_seconds,
        extra_args=extra_args,
    ) as env:
        print("llm_initialized=True")
        print("vllm_server_initialized=True")
        print("vllm_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""))
        generate_url = _server_base_url(env.server_url) + "/inference/v1/generate"
        response = requests.post(
            generate_url,
            json={
                "model": env.model_id,
                "token_ids": _PROMPT_IDS,
                "sampling_params": {
                    "max_tokens": generation_tokens,
                    "temperature": 0.0,
                },
                "stream": False,
            },
            timeout=300,
        )
        print("vllm_generate_status_code=" + str(response.status_code))
        if not response.ok:
            print("vllm_generate_response_text=" + response.text[:4000])
            print("vllm_server_logs_tail_begin")
            print(env.logs_tail(max_lines=400))
            print("vllm_server_logs_tail_end")
            response.raise_for_status()
        payload = response.json()
        print("vllm_server_logs_tail_begin")
        print(env.logs_tail(max_lines=200))
        print("vllm_server_logs_tail_end")

    generated_ids = payload["choices"][0]["token_ids"]
    print("installed_full_canary_generate_payload=" + repr(payload))
    print("installed_full_canary_generated=" + repr([(_PROMPT_IDS, generated_ids, "")]))
    print("installed_path_result=works:full_canary_generate")


def _run_phase_subprocess(
    phase: str,
    *,
    output_dir: Path,
    model_size: str,
    artifact_dir: Path | None = None,
    reference_path: Path | None = None,
    max_shard_size: int,
    generation_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
    max_logprob_delta: float,
    routing_margin_tolerance: float,
    routing_debug: bool,
    routing_debug_token_position: int,
    routing_debug_layer: int,
    routing_debug_vector_limit: int,
) -> None:
    args = [
        sys.executable,
        "-m",
        "experiments.grug.moe.installed_vllm_full_canary_smoke",
        "--phase",
        phase,
        "--output-dir",
        str(output_dir),
        "--model-size",
        model_size,
        "--max-shard-size",
        str(max_shard_size),
        "--generation-tokens",
        str(generation_tokens),
        "--server-port",
        str(server_port),
        "--server-timeout-seconds",
        str(server_timeout_seconds),
        "--max-logprob-delta",
        str(max_logprob_delta),
        "--routing-margin-tolerance",
        str(routing_margin_tolerance),
    ]
    if routing_debug:
        args.extend(
            [
                "--routing-debug",
                "--routing-debug-token-position",
                str(routing_debug_token_position),
                "--routing-debug-layer",
                str(routing_debug_layer),
                "--routing-debug-vector-limit",
                str(routing_debug_vector_limit),
            ]
        )
    if artifact_dir is not None:
        args.extend(["--artifact-dir", str(artifact_dir)])
    if reference_path is not None:
        args.extend(["--reference-path", str(reference_path)])
    subprocess.run(args, check=True, env=_subprocess_env())


def run(
    output_dir: Path,
    *,
    model_size: str,
    max_shard_size: int,
    generation_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
    max_logprob_delta: float,
    routing_margin_tolerance: float,
    routing_debug: bool,
    routing_debug_token_position: int,
    routing_debug_layer: int,
    routing_debug_vector_limit: int,
) -> None:
    _print_runtime_header()
    print("installed_model_size=" + model_size)
    print(
        "installed_routing_debug="
        + json.dumps(
            {
                "enabled": bool(routing_debug),
                "token_position": int(routing_debug_token_position),
                "layer": int(routing_debug_layer),
                "vector_limit": int(routing_debug_vector_limit),
            },
            sort_keys=True,
        )
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    _run_phase_subprocess(
        "export",
        output_dir=output_dir,
        model_size=model_size,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
        server_port=server_port,
        server_timeout_seconds=server_timeout_seconds,
        max_logprob_delta=max_logprob_delta,
        routing_margin_tolerance=routing_margin_tolerance,
        routing_debug=routing_debug,
        routing_debug_token_position=routing_debug_token_position,
        routing_debug_layer=routing_debug_layer,
        routing_debug_vector_limit=routing_debug_vector_limit,
    )

    artifact_dir = _artifact_dir(output_dir)
    reference_path = _reference_path(output_dir)
    print("installed_artifact_dir=" + str(artifact_dir))
    print("installed_reference_json=" + str(reference_path))
    print("full_canary_artifact_dir=" + str(artifact_dir))
    print("full_canary_reference_json=" + str(reference_path))
    print("full_canary_artifact_bytes=" + str(_directory_size_bytes(artifact_dir)))
    print("full_canary_shard_count=" + str(_shard_count(artifact_dir)))

    _run_phase_subprocess(
        "serve",
        output_dir=output_dir,
        model_size=model_size,
        artifact_dir=artifact_dir,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
        server_port=server_port,
        server_timeout_seconds=server_timeout_seconds,
        max_logprob_delta=max_logprob_delta,
        routing_margin_tolerance=routing_margin_tolerance,
        routing_debug=routing_debug,
        routing_debug_token_position=routing_debug_token_position,
        routing_debug_layer=routing_debug_layer,
        routing_debug_vector_limit=routing_debug_vector_limit,
    )
    _run_phase_subprocess(
        "score",
        output_dir=output_dir,
        model_size=model_size,
        artifact_dir=artifact_dir,
        reference_path=reference_path,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
        server_port=server_port,
        server_timeout_seconds=server_timeout_seconds,
        max_logprob_delta=max_logprob_delta,
        routing_margin_tolerance=routing_margin_tolerance,
        routing_debug=routing_debug,
        routing_debug_token_position=routing_debug_token_position,
        routing_debug_layer=routing_debug_layer,
        routing_debug_vector_limit=routing_debug_vector_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "export", "score", "serve"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/grugmoe-installed-vllm-full-canary"),
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(_MODEL_SIZE_TO_PARITY_CONFIG),
        default=_DEFAULT_MODEL_SIZE,
        help="Installed-path validation profile to export, serve, and score.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Artifact directory for --phase score/serve. Defaults to OUTPUT_DIR/grugmoe-inference.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help=f"Reference JSON for --phase score. Defaults to OUTPUT_DIR/{_REFERENCE_JSON_NAME}.",
    )
    parser.add_argument("--max-shard-size", type=int, default=268435456)
    parser.add_argument("--generation-tokens", type=int, default=3)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-logprob-delta", type=float, default=_DEFAULT_MAX_LOGPROB_DELTA)
    parser.add_argument("--routing-margin-tolerance", type=float, default=_DEFAULT_ROUTING_MARGIN_TOLERANCE)
    parser.add_argument(
        "--routing-debug",
        action="store_true",
        help="Emit Levanter, tpu-inference, and final-output routing debug records for one token/layer.",
    )
    parser.add_argument("--routing-debug-token-position", type=int, default=0)
    parser.add_argument("--routing-debug-layer", type=int, default=0)
    parser.add_argument("--routing-debug-vector-limit", type=int, default=_ROUTING_DEBUG_VECTOR_LIMIT)
    args = parser.parse_args()

    if args.phase == "export":
        _print_runtime_header()
        export_artifact(
            args.output_dir,
            model_size=args.model_size,
            max_shard_size=args.max_shard_size,
            generation_tokens=args.generation_tokens,
            routing_debug=args.routing_debug,
            routing_debug_token_position=args.routing_debug_token_position,
            routing_debug_layer=args.routing_debug_layer,
            routing_debug_vector_limit=args.routing_debug_vector_limit,
        )
    elif args.phase == "score":
        _print_runtime_header()
        score_artifact(
            args.artifact_dir or _artifact_dir(args.output_dir),
            args.reference_path or _reference_path(args.output_dir),
            max_logprob_delta=args.max_logprob_delta,
            routing_margin_tolerance=args.routing_margin_tolerance,
            routing_debug=args.routing_debug,
            routing_debug_token_position=args.routing_debug_token_position,
            routing_debug_layer=args.routing_debug_layer,
            routing_debug_vector_limit=args.routing_debug_vector_limit,
        )
    elif args.phase == "serve":
        _print_runtime_header()
        serve_artifact(
            args.artifact_dir or _artifact_dir(args.output_dir),
            model_size=args.model_size,
            generation_tokens=args.generation_tokens,
            server_port=args.server_port,
            server_timeout_seconds=args.server_timeout_seconds,
        )
    else:
        run(
            args.output_dir,
            model_size=args.model_size,
            max_shard_size=args.max_shard_size,
            generation_tokens=args.generation_tokens,
            server_port=args.server_port,
            server_timeout_seconds=args.server_timeout_seconds,
            max_logprob_delta=args.max_logprob_delta,
            routing_margin_tolerance=args.routing_margin_tolerance,
            routing_debug=args.routing_debug,
            routing_debug_token_position=args.routing_debug_token_position,
            routing_debug_layer=args.routing_debug_layer,
            routing_debug_vector_limit=args.routing_debug_vector_limit,
        )


if __name__ == "__main__":
    main()
