#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hard-fail live probe for Marin RL vLLM TPU weight hot-swap."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

LOGGER = logging.getLogger("rl_real_weight_update_probe")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<no package metadata>"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _to_numpy(value: Any) -> np.ndarray:
    import jax

    if hasattr(value, "value"):
        value = value.value
    value = jax.device_get(value)
    return np.asarray(value)


def _array_summary(value: Any) -> dict[str, Any]:
    array = _to_numpy(value)
    numeric = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "mean": float(numeric.mean()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "first_values": numeric.reshape(-1)[:8].tolist(),
    }


def _find_state_owner(driver_worker: Any) -> tuple[str, Any]:
    candidates = [
        ("driver_worker", driver_worker),
        ("driver_worker.model_runner", getattr(driver_worker, "model_runner", None)),
        ("driver_worker.worker", getattr(driver_worker, "worker", None)),
    ]
    worker = getattr(driver_worker, "worker", None)
    if worker is not None:
        candidates.append(("driver_worker.worker.model_runner", getattr(worker, "model_runner", None)))

    for name, candidate in candidates:
        if candidate is not None and hasattr(candidate, "state"):
            return name, candidate

    attrs = sorted(attr for attr in dir(driver_worker) if not attr.startswith("_"))[:80]
    raise RuntimeError(f"Could not locate a state owner under driver_worker; public attrs={attrs}")


def _nearby_keys(state: Mapping[str, Any], key: str) -> list[str]:
    key_parts = [part for part in key.split(".") if part not in {"model", "vllm_model"}]
    scored = []
    for candidate in state.keys():
        score = sum(part in candidate for part in key_parts)
        if score:
            scored.append((score, candidate))
    return [candidate for _score, candidate in sorted(scored, reverse=True)[:30]]


def _generate(ctx: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    from marin.rl.decoding import DecodingConfig, DecodingStrategy

    output = ctx.batch_completions(
        [prompt],
        n=1,
        decoding=DecodingConfig(
            strategy=DecodingStrategy.GREEDY,
            temperature=0.0,
            max_output_tokens=max_tokens,
            ignore_eos=True,
        ),
    )[0]
    choice = output.choices[0]
    return {
        "text": choice.message.content,
        "response_token_ids": list(getattr(choice, "response_token_ids", [])),
        "prompt_token_ids": list(getattr(choice, "prompt_token_ids", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--canonical-model", default=None)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prompt", default="What is 2+2? Answer with one short phrase.")
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--source-key", default="model.norm")
    parser.add_argument("--source-state-key", default="model.norm.weight")
    parser.add_argument("--replacement-value", type=float, default=0.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "tpu")
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from marin.rl.environments.inference_ctx.vllm import (
        InferenceMode,
        VLLMEngineConfig,
        VLLMFallbackSamplingConfig,
        vLLMInferenceContext,
        vLLMInferenceContextConfig,
    )
    from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS

    canonical_model = args.canonical_model or args.model
    LOGGER.info(
        "package versions: %s",
        {
            "marin": _package_version("marin"),
            "vllm": _package_version("vllm"),
            "tpu-inference": _package_version("tpu-inference"),
            "jax": _package_version("jax"),
            "torch": _package_version("torch"),
        },
    )
    LOGGER.info("creating vLLM inference context for model=%s canonical=%s", args.model, canonical_model)

    ctx = vLLMInferenceContext(
        vLLMInferenceContextConfig(
            engine=VLLMEngineConfig(
                model_name=args.model,
                canonical_model_name=canonical_model,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                mode=InferenceMode.SYNC,
            ),
            fallback_sampling=VLLMFallbackSamplingConfig(top_k=None),
        )
    )

    driver_worker = ctx.llm.llm_engine.model_executor.driver_worker
    state_owner_name, state_owner = _find_state_owner(driver_worker)
    state = state_owner.state
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected mapping-backed vLLM target state, got {type(state).__name__}")

    target_key, target_sharding = MODEL_MAPPINGS[canonical_model][args.source_key]
    LOGGER.info(
        "using mapped source=%s state_key=%s target=%s sharding=%s owner=%s state_keys=%d",
        args.source_key,
        args.source_state_key,
        target_key,
        target_sharding,
        state_owner_name,
        len(state),
    )
    if target_key not in state:
        nearby = _nearby_keys(state, target_key)
        print(
            json.dumps(
                {
                    "status": "missing_target_key",
                    "target_key": target_key,
                    "nearby_keys": nearby,
                    "first_state_keys": list(state.keys())[:30],
                },
                indent=2,
                default=_json_default,
            )
        )
        return 3

    before_weight = _array_summary(state[target_key])
    before_generation = _generate(ctx, args.prompt, args.max_output_tokens)

    replacement = np.full(before_weight["shape"], args.replacement_value, dtype=np.float32)
    LOGGER.info("calling reload_model with one-tensor update; replacement summary=%s", _array_summary(replacement))
    try:
        ctx.reload_model(None, {args.source_state_key: replacement})
    except Exception:
        traceback.print_exc()
        return 4

    state = state_owner.state
    after_weight = _array_summary(state[target_key])
    after_generation = _generate(ctx, args.prompt, args.max_output_tokens)
    changed = before_weight["sha256"] != after_weight["sha256"]
    reached_value = (
        after_weight["shape"] == before_weight["shape"]
        and abs(after_weight["mean"] - args.replacement_value) < 1e-3
        and abs(after_weight["min"] - args.replacement_value) < 1e-3
        and abs(after_weight["max"] - args.replacement_value) < 1e-3
    )
    generation_changed = before_generation["response_token_ids"] != after_generation["response_token_ids"]

    result = {
        "status": "passed" if changed and reached_value else "failed",
        "model": args.model,
        "canonical_model": canonical_model,
        "source_key": args.source_key,
        "source_state_key": args.source_state_key,
        "target_key": target_key,
        "target_sharding": target_sharding,
        "state_owner": state_owner_name,
        "replacement_value": args.replacement_value,
        "before_weight": before_weight,
        "after_weight": after_weight,
        "weight_hash_changed": changed,
        "weight_reached_requested_value": reached_value,
        "before_generation": before_generation,
        "after_generation": after_generation,
        "generation_changed": generation_changed,
        "config": asdict(args) if hasattr(args, "__dataclass_fields__") else vars(args),
    }
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if changed and reached_value else 5


if __name__ == "__main__":
    sys.exit(main())
