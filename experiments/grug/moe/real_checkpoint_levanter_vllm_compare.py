# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter/JAX GrugMoE real-checkpoint smoke compared with a vLLM TPU result.

This diagnostic restores the same trained GrugMoE checkpoint used by the vLLM
smoke, tokenizes the same prompt, runs deterministic full-forward greedy decode
for a small number of tokens, and writes a structured comparison against the
persisted vLLM result.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import equinox as eqx
import fsspec
import haliax
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.model import MoEMLP
from experiments.grug.moe.real_checkpoint_vllm_smoke import (
    _MARIN_GIT_SHA_ENV,
    DEFAULT_DISK,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RAM,
    DEFAULT_REGION,
    DEFAULT_TPU_TYPE,
    EUROPE_WEST4_GCS_PREFIX,
    LLAMA31_TOKENIZER_PATH,
    PROMPT,
    REAL_CHECKPOINT_PATH,
    _exists,
    _git_sha,
    _load_legacy_split_expert_checkpoint,
    _rm_tree,
    _write_json,
    join_path,
    real_checkpoint_model_config,
    require_local_or_europe_west4,
    runtime_snapshot,
)
from experiments.grug.moe.real_checkpoint_vllm_smoke import (
    SmokeConfig as VllmSmokeConfig,
)

VLLM_REFERENCE_RESULT_PATH = (
    "gs://marin-eu-west4/tmp/ttl=14d/grugmoe-real-checkpoint-vllm-smoke/" "final-codex-20260620-fresh/result.json"
)
OUTPUT_ROOT = "gs://marin-eu-west4/tmp/ttl=14d/grugmoe-real-checkpoint-levanter-vllm-compare"
CACHE_ROOT = "gs://marin-eu-west4/compilation-cache/grugmoe-real-checkpoint-levanter-vllm-compare"
DEFAULT_DECODE_SEQ_LEN = 128
_DEPENDENCY_GROUPS = ["eval", "tpu"]
_DEFAULT_OUTPUT_SHARP_EDGE = (
    "The validated vLLM JSON stores text and token counts, but not completion token IDs or logprobs. "
    "Continuation token IDs are therefore derived with the same tokenizer when possible, and vLLM logprob "
    "comparison is marked unavailable."
)
_SPLASH_PADDING_SHARP_EDGE = (
    "Levanter/JAX decode pads the model input to 128 tokens so the production TPU splash attention kernel can run; "
    "only logits at the real prompt/generated prefix positions are used for the comparison."
)


@dataclass(frozen=True)
class CompareConfig:
    checkpoint_path: str
    tokenizer_path: str
    vllm_result_path: str
    output_dir: str
    cache_dir: str
    prompt: str
    max_new_tokens: int
    decode_seq_len: int
    overwrite: bool
    fail_on_mismatch: bool

    @property
    def result_path(self) -> str:
        return join_path(self.output_dir, "result.json")


def validate_locality(config: CompareConfig) -> dict[str, str]:
    paths = {
        "checkpoint_path": config.checkpoint_path,
        "tokenizer_path": config.tokenizer_path,
        "vllm_result_path": config.vllm_result_path,
        "output_dir": config.output_dir,
        "cache_dir": config.cache_dir,
        "result_path": config.result_path,
    }
    for label, path in paths.items():
        require_local_or_europe_west4(label, path)
    return paths


def _read_json(path: str) -> dict[str, Any]:
    require_local_or_europe_west4("json_path", path)
    with fsspec.open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _configure_runtime_environment(config: CompareConfig) -> None:
    require_local_or_europe_west4("cache_dir", config.cache_dir)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", config.cache_dir)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _runtime_snapshot() -> dict[str, Any]:
    snapshot = runtime_snapshot(include_jax_devices=False)
    try:
        grugmoe_spec = repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe"))
    except ModuleNotFoundError as exc:
        grugmoe_spec = f"unavailable:{exc!r}"
    snapshot.update(
        {
            "grugmoe_spec": grugmoe_spec,
            "jax_process_index": jax.process_index(),
            "jax_process_count": jax.process_count(),
            "jax_local_device_count": jax.local_device_count(),
            "jax_devices": [str(device) for device in jax.devices()],
        }
    )
    return snapshot


def _restore_config(config: CompareConfig) -> VllmSmokeConfig:
    return VllmSmokeConfig(
        phase="export",
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
        output_dir=config.output_dir,
        cache_dir=config.cache_dir,
        prompt=config.prompt,
        expected_output=None,
        calibrate_output=False,
        overwrite=config.overwrite,
        max_shard_size=0,
        max_model_len=0,
        max_tokens=config.max_new_tokens,
        server_port=0,
        server_timeout_seconds=0,
        vllm_dtype="bfloat16",
    )


def _executable_mlp_from_legacy_split(split_mlp: Any) -> MoEMLP:
    w_gate_up = jnp.concatenate([split_mlp.w_gate, split_mlp.w_up], axis=-1)
    expert_mlp = MoEExpertMlp(
        w_gate_up=w_gate_up,
        w_down=split_mlp.w_down,
        implementation=split_mlp.cfg.moe_implementation,
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
    )
    return MoEMLP(
        router=split_mlp.router,
        router_bias=split_mlp.router_bias,
        expert_mlp=expert_mlp,
        cfg=split_mlp.cfg,
    )


def executable_model_from_legacy_split(model: Any) -> Any:
    """Convert a legacy split-expert checkpoint template into a callable model."""
    for layer_index, block in enumerate(model.blocks):
        if hasattr(block.mlp, "expert_mlp"):
            continue
        executable_mlp = _executable_mlp_from_legacy_split(block.mlp)
        model = eqx.tree_at(lambda m, i=layer_index: m.blocks[i].mlp, model, executable_mlp)
    return model


def _last_position_logits_batch(model: Any, token_ids: jax.Array) -> jax.Array:
    return model.logits(token_ids)[:, -1, :].astype(jnp.float32)


def _position_logits_batch(model: Any, token_ids: jax.Array, position: jax.Array) -> jax.Array:
    return model.logits(token_ids)[:, position, :].astype(jnp.float32)


def _selected_logprob(logits: np.ndarray, token_id: int) -> float:
    logits_f64 = logits.astype(np.float64)
    max_logit = np.max(logits_f64)
    log_z = max_logit + np.log(np.exp(logits_f64 - max_logit).sum())
    return float(logits_f64[token_id] - log_z)


def _mesh_batch_axis_size(mesh: Any) -> int:
    size = 1
    for axis_name in ("replica_dcn", "data", "expert"):
        size *= int(mesh.shape.get(axis_name, 1))
    return size


def _decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def greedy_decode(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    batch_size: int = 1,
    decode_seq_len: int | None = None,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    if decode_seq_len is not None and len(prompt_ids) + max_new_tokens > decode_seq_len:
        raise ValueError(
            f"prompt length {len(prompt_ids)} + max_new_tokens {max_new_tokens} exceeds "
            f"decode_seq_len {decode_seq_len}"
        )

    token_ids_array: np.ndarray | None = None
    if decode_seq_len is None:
        token_ids = jnp.asarray([prompt_ids] * batch_size, dtype=jnp.int32)
    else:
        token_ids_array = np.full((batch_size, decode_seq_len), pad_token_id, dtype=np.int32)
        token_ids_array[:, : len(prompt_ids)] = np.asarray(prompt_ids, dtype=np.int32)
        token_ids = jnp.asarray(token_ids_array, dtype=jnp.int32)
    generated_ids: list[int] = []
    generated_token_texts: list[str] = []
    selected_token_logprobs: list[float] = []
    steps: list[dict[str, Any]] = []
    last_logits = jax.jit(_last_position_logits_batch)
    position_logits = jax.jit(_position_logits_batch)
    started = time.time()

    for step_index in range(max_new_tokens):
        if decode_seq_len is None:
            step_logits_batch = last_logits(model, token_ids)
        else:
            if token_ids_array is None:
                raise AssertionError("token_ids_array must be initialized for fixed-length decode")
            position = jnp.asarray(len(prompt_ids) + step_index - 1, dtype=jnp.int32)
            step_logits_batch = position_logits(model, jnp.asarray(token_ids_array, dtype=jnp.int32), position)
        step_logits = np.asarray(jax.device_get(step_logits_batch))[0]
        selected_token_id = int(np.argmax(step_logits, axis=-1))
        selected_logprob = _selected_logprob(step_logits, selected_token_id)
        selected_text = _decode_one(tokenizer, selected_token_id)

        generated_ids.append(selected_token_id)
        generated_token_texts.append(selected_text)
        selected_token_logprobs.append(selected_logprob)
        steps.append(
            {
                "generated_token_index": step_index,
                "token_id": selected_token_id,
                "token_text": selected_text,
                "selected_token_logprob": selected_logprob,
            }
        )
        if decode_seq_len is None:
            next_token = jnp.full((batch_size, 1), selected_token_id, dtype=token_ids.dtype)
            token_ids = jnp.concatenate([token_ids, next_token], axis=1)
        else:
            if token_ids_array is None:
                raise AssertionError("token_ids_array must be initialized for fixed-length decode")
            token_ids_array[:, len(prompt_ids) + step_index] = selected_token_id

    score_len = len(prompt_ids) + len(generated_ids)
    if decode_seq_len is None:
        score_token_ids = np.asarray(token_ids[:, :score_len]).tolist()[0]
    else:
        if token_ids_array is None:
            raise AssertionError("token_ids_array must be initialized for fixed-length decode")
        score_token_ids = token_ids_array[0, :score_len].tolist()
    return {
        "prompt_token_ids": [int(token_id) for token_id in prompt_ids],
        "prompt_token_count": len(prompt_ids),
        "decode_batch_size": batch_size,
        "decode_seq_len": decode_seq_len,
        "pad_token_id": pad_token_id if decode_seq_len is not None else None,
        "generated_token_ids": generated_ids,
        "generated_token_texts": generated_token_texts,
        "decoded_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "selected_token_logprobs": selected_token_logprobs,
        "steps": steps,
        "elapsed_seconds": time.time() - started,
        "score_token_ids": score_token_ids,
    }


def _extract_vllm_reference(payload: dict[str, Any]) -> dict[str, Any]:
    serve_result = payload.get("serve_result")
    if not isinstance(serve_result, dict):
        raise ValueError("vLLM reference result is missing serve_result")
    raw_response = serve_result.get("raw_response")
    if not isinstance(raw_response, dict):
        raw_response = {}
    choices = raw_response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    raw_token_ids = choice.get("token_ids")
    if not isinstance(raw_token_ids, list):
        raw_token_ids = None
    prompt_token_ids = choice.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list):
        prompt_token_ids = None
    logprobs = choice.get("logprobs")

    return {
        "prompt": payload.get("prompt"),
        "completion": serve_result.get("completion", choice.get("text", "")),
        "passed": serve_result.get("passed"),
        "expected_output": serve_result.get("expected_output"),
        "calibrated_expected_output": serve_result.get("calibrated_expected_output"),
        "raw_completion_token_ids": raw_token_ids,
        "raw_prompt_token_ids": prompt_token_ids,
        "raw_logprobs": logprobs,
        "usage": usage,
        "prompt_token_count": usage.get("prompt_tokens"),
        "completion_token_count": usage.get("completion_tokens"),
        "model": raw_response.get("model"),
        "system_fingerprint": raw_response.get("system_fingerprint"),
    }


def _tokenizer_encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return [int(token_id) for token_id in ids]


def choose_prompt_tokenization(
    tokenizer: Any,
    prompt: str,
    *,
    expected_prompt_token_count: int | None,
) -> tuple[list[int], dict[str, Any]]:
    candidates = [
        {
            "add_special_tokens": True,
            "token_ids": _tokenizer_encode(tokenizer, prompt, add_special_tokens=True),
        },
        {
            "add_special_tokens": False,
            "token_ids": _tokenizer_encode(tokenizer, prompt, add_special_tokens=False),
        },
    ]
    selected = candidates[0]
    if expected_prompt_token_count is not None:
        for candidate in candidates:
            if len(candidate["token_ids"]) == expected_prompt_token_count:
                selected = candidate
                break
    report = {
        "expected_prompt_token_count": expected_prompt_token_count,
        "selected_add_special_tokens": selected["add_special_tokens"],
        "selected_prompt_token_count": len(selected["token_ids"]),
        "selected_prompt_token_ids": selected["token_ids"],
        "candidates": [
            {
                "add_special_tokens": candidate["add_special_tokens"],
                "prompt_token_count": len(candidate["token_ids"]),
                "prompt_token_ids": candidate["token_ids"],
            }
            for candidate in candidates
        ],
        "matches_expected_count": (
            expected_prompt_token_count is None or len(selected["token_ids"]) == expected_prompt_token_count
        ),
    }
    return selected["token_ids"], report


def derive_vllm_continuation_token_ids(
    tokenizer: Any,
    *,
    prompt: str,
    prompt_ids: list[int],
    prompt_add_special_tokens: bool,
    completion: str,
    raw_completion_token_ids: list[int] | None,
) -> tuple[list[int] | None, str | None, dict[str, Any]]:
    if raw_completion_token_ids is not None:
        return [int(token_id) for token_id in raw_completion_token_ids], "raw_response", {}

    if completion is None:
        return None, None, {"reason": "missing vLLM completion text"}

    full_ids = _tokenizer_encode(tokenizer, prompt + completion, add_special_tokens=prompt_add_special_tokens)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return full_ids[len(prompt_ids) :], "derived_prompt_plus_completion_suffix", {}

    completion_only_ids = _tokenizer_encode(tokenizer, completion, add_special_tokens=False)
    return (
        completion_only_ids,
        "derived_completion_only",
        {
            "reason": "prompt+completion tokenization was not prefixed by prompt tokenization",
            "prompt_plus_completion_token_ids": full_ids,
        },
    )


def _first_mismatch(left: list[Any], right: list[Any]) -> int | None:
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
        if left_item != right_item:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _token_texts(tokenizer: Any, token_ids: list[int] | None) -> list[str] | None:
    if token_ids is None:
        return None
    return [_decode_one(tokenizer, token_id) for token_id in token_ids]


def compare_levanter_to_vllm(
    *,
    tokenizer: Any,
    prompt: str,
    tokenization: dict[str, Any],
    levanter_result: dict[str, Any],
    vllm_reference: dict[str, Any],
    vllm_continuation_token_ids: list[int] | None,
    vllm_token_ids_source: str | None,
) -> dict[str, Any]:
    levanter_ids = levanter_result["generated_token_ids"]
    vllm_completion = str(vllm_reference.get("completion", ""))
    levanter_text = str(levanter_result["decoded_text"])
    vllm_token_texts = _token_texts(tokenizer, vllm_continuation_token_ids)
    prompt_text_match = vllm_reference.get("prompt") == prompt
    prompt_token_count_match = bool(tokenization["matches_expected_count"])
    text_match = levanter_text == vllm_completion
    token_ids_match = vllm_continuation_token_ids is not None and levanter_ids == list(vllm_continuation_token_ids)

    divergence = None
    token_mismatch = None
    if vllm_continuation_token_ids is not None:
        token_mismatch = _first_mismatch(levanter_ids, list(vllm_continuation_token_ids))
    if token_mismatch is not None:
        levanter_token_id = levanter_ids[token_mismatch] if token_mismatch < len(levanter_ids) else None
        vllm_token_id = (
            vllm_continuation_token_ids[token_mismatch] if token_mismatch < len(vllm_continuation_token_ids) else None
        )
        levanter_token_text = (
            levanter_result["generated_token_texts"][token_mismatch]
            if token_mismatch < len(levanter_result["generated_token_texts"])
            else None
        )
        vllm_token_text = (
            vllm_token_texts[token_mismatch] if vllm_token_texts and token_mismatch < len(vllm_token_texts) else None
        )
        levanter_logprob = (
            levanter_result["selected_token_logprobs"][token_mismatch]
            if token_mismatch < len(levanter_result["selected_token_logprobs"])
            else None
        )
        divergence = {
            "generated_token_index": token_mismatch,
            "levanter_token_id": levanter_token_id,
            "levanter_token_text": levanter_token_text,
            "vllm_token_id": vllm_token_id,
            "vllm_token_text": vllm_token_text,
            "levanter_selected_token_logprob": levanter_logprob,
            "vllm_selected_token_logprob": None,
            "selected_token_logprob_delta": None,
            "likely_cause_or_next_debugging_target": (
                "vLLM token IDs were not stored in the reference JSON; rerun the vLLM reference with token IDs "
                "and logprobs if a derived-token mismatch needs confirmation."
                if vllm_token_ids_source and vllm_token_ids_source.startswith("derived")
                else "Check checkpoint restore/layout, tokenizer settings, and greedy decode parity."
            ),
        }
    elif not text_match:
        divergence = {
            "generated_token_index": None,
            "levanter_token_id": None,
            "levanter_token_text": None,
            "vllm_token_id": None,
            "vllm_token_text": None,
            "levanter_selected_token_logprob": None,
            "vllm_selected_token_logprob": None,
            "selected_token_logprob_delta": None,
            "text_first_difference": _first_text_difference(levanter_text, vllm_completion),
            "likely_cause_or_next_debugging_target": (
                "The reference JSON lacks authoritative vLLM token IDs/logprobs. Capture them from vLLM, then "
                "compare the first divergent token against Levanter logits."
            ),
        }
    elif not prompt_token_count_match:
        divergence = {
            "generated_token_index": None,
            "likely_cause_or_next_debugging_target": (
                "Prompt token count differs; check tokenizer add_special_tokens settings."
            ),
        }

    return {
        "prompt_text_match": prompt_text_match,
        "prompt_token_count_match": prompt_token_count_match,
        "text_match": text_match,
        "token_ids_match": token_ids_match,
        "vllm_token_ids_source": vllm_token_ids_source,
        "selected_token_logprobs_comparable": False,
        "vllm_logprobs_available": vllm_reference.get("raw_logprobs") is not None,
        "passed": prompt_text_match and prompt_token_count_match and text_match and token_ids_match,
        "divergence": divergence,
    }


def _first_text_difference(left: str, right: str) -> dict[str, Any] | None:
    for index, (left_char, right_char) in enumerate(zip(left, right, strict=False)):
        if left_char != right_char:
            return {"char_index": index, "levanter_char": left_char, "vllm_char": right_char}
    if len(left) != len(right):
        return {
            "char_index": min(len(left), len(right)),
            "levanter_char": left[min(len(left), len(right)) :] or None,
            "vllm_char": right[min(len(left), len(right)) :] or None,
        }
    return None


def run_compare(config: CompareConfig) -> dict[str, Any]:
    _configure_runtime_environment(config)
    locality_paths = validate_locality(config)
    if config.output_dir.startswith(EUROPE_WEST4_GCS_PREFIX) or os.path.exists(config.output_dir):
        if _exists(config.output_dir):
            if not config.overwrite:
                raise FileExistsError(f"{config.output_dir} already exists; pass --overwrite to regenerate it")
            _rm_tree(config.output_dir)

    _require_reference_inputs(config)
    vllm_payload = _read_json(config.vllm_result_path)
    vllm_reference = _extract_vllm_reference(vllm_payload)
    expected_prompt_tokens = vllm_reference["prompt_token_count"]
    if expected_prompt_tokens is not None:
        expected_prompt_tokens = int(expected_prompt_tokens)

    print("grugmoe_levanter_vllm_compare_preflight=" + json.dumps(locality_paths, sort_keys=True), flush=True)
    print("grugmoe_levanter_vllm_reference=" + json.dumps(vllm_reference, sort_keys=True), flush=True)

    tokenizer = load_tokenizer(config.tokenizer_path)
    prompt_ids, tokenization = choose_prompt_tokenization(
        tokenizer,
        config.prompt,
        expected_prompt_token_count=expected_prompt_tokens,
    )
    vllm_continuation_token_ids, vllm_token_ids_source, vllm_derivation_note = derive_vllm_continuation_token_ids(
        tokenizer,
        prompt=config.prompt,
        prompt_ids=prompt_ids,
        prompt_add_special_tokens=bool(tokenization["selected_add_special_tokens"]),
        completion=str(vllm_reference.get("completion", "")),
        raw_completion_token_ids=vllm_reference.get("raw_completion_token_ids"),
    )

    model_cfg = real_checkpoint_model_config()
    mesh = compact_grug_mesh()
    decode_batch_size = _mesh_batch_axis_size(mesh)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(_restore_config(config), model_cfg)
        model = executable_model_from_legacy_split(loaded_model)
        levanter_result = greedy_decode(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=config.max_new_tokens,
            batch_size=decode_batch_size,
            decode_seq_len=config.decode_seq_len,
            pad_token_id=pad_token_id,
        )
    total_elapsed = time.time() - started

    comparison = compare_levanter_to_vllm(
        tokenizer=tokenizer,
        prompt=config.prompt,
        tokenization=tokenization,
        levanter_result=levanter_result,
        vllm_reference=vllm_reference,
        vllm_continuation_token_ids=vllm_continuation_token_ids,
        vllm_token_ids_source=vllm_token_ids_source,
    )

    result = {
        "checkpoint_path": config.checkpoint_path,
        "tokenizer_path": config.tokenizer_path,
        "vllm_result_path": config.vllm_result_path,
        "output_dir": config.output_dir,
        "result_path": config.result_path,
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "decode_seq_len": config.decode_seq_len,
        "locality_paths": locality_paths,
        "runtime": _runtime_snapshot(),
        "model_config": dataclasses.asdict(model_cfg),
        "tokenization": tokenization,
        "levanter_result": levanter_result,
        "vllm_reference": {
            **vllm_reference,
            "continuation_token_ids": vllm_continuation_token_ids,
            "continuation_token_ids_source": vllm_token_ids_source,
            "continuation_token_texts": _token_texts(tokenizer, vllm_continuation_token_ids),
            "continuation_token_derivation_note": vllm_derivation_note,
            "selected_token_logprobs": None,
        },
        "comparison": comparison,
        "elapsed_seconds": total_elapsed,
        "sharp_edges": [_DEFAULT_OUTPUT_SHARP_EDGE, _SPLASH_PADDING_SHARP_EDGE],
    }
    _write_json(config.result_path, result)
    print("grugmoe_levanter_vllm_compare=" + json.dumps(result, sort_keys=True), flush=True)
    if config.fail_on_mismatch and not comparison["passed"]:
        raise AssertionError("Levanter/JAX output did not match vLLM reference; see result JSON for divergence")
    return result


def _require_reference_inputs(config: CompareConfig) -> None:
    for label, path in (
        ("checkpoint metadata", join_path(config.checkpoint_path, "metadata.json")),
        ("tokenizer.json", join_path(config.tokenizer_path, "tokenizer.json")),
        ("vLLM result", config.vllm_result_path),
    ):
        require_local_or_europe_west4(label, path)
        if not _exists(path):
            raise FileNotFoundError(f"{label} not found at {path}")


def submit_compare(config: CompareConfig, *, tpu_type: str, region: str, ram: str, disk: str, job_name: str) -> None:
    from fray import current_client  # noqa: PLC0415
    from fray.cluster import ResourceConfig  # noqa: PLC0415
    from fray.types import Entrypoint, JobRequest, create_environment  # noqa: PLC0415
    from marin.training.run_environment import env_vars_for_dependency_groups  # noqa: PLC0415

    if region != DEFAULT_REGION:
        raise ValueError(f"This smoke is pinned to {DEFAULT_REGION}; got {region!r}")
    resources = ResourceConfig.with_tpu(tpu_type, regions=[region], ram=ram, disk=disk)
    env_vars = env_vars_for_dependency_groups(
        resources,
        _DEPENDENCY_GROUPS,
        {
            _MARIN_GIT_SHA_ENV: _git_sha(),
            "JAX_COMPILATION_CACHE_DIR": config.cache_dir,
            "PYTHONUNBUFFERED": "1",
        },
    )
    request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(run_compare, args=(config,)),
        resources=resources,
        environment=create_environment(
            extras=_DEPENDENCY_GROUPS,
            env_vars=env_vars,
        ),
        max_retries_failure=0,
    )
    print(
        "submitting_grugmoe_levanter_vllm_compare="
        + json.dumps(
            {
                "job_name": job_name,
                "tpu_type": tpu_type,
                "region": region,
                "ram": ram,
                "disk": disk,
                "vllm_result_path": config.vllm_result_path,
                "output_dir": config.output_dir,
                "result_path": config.result_path,
                "cache_dir": config.cache_dir,
                "marin_sha": env_vars.get(_MARIN_GIT_SHA_ENV),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    job = current_client().submit(request)
    print("submitted_job_id=" + str(job.job_id), flush=True)
    job.wait(raise_on_failure=True)


def _default_output_dir() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return join_path(OUTPUT_ROOT, stamp)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default=REAL_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer-path", default=LLAMA31_TOKENIZER_PATH)
    parser.add_argument("--vllm-result-path", default=VLLM_REFERENCE_RESULT_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=CACHE_ROOT)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--decode-seq-len", type=int, default=DEFAULT_DECODE_SEQ_LEN)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--ram", default=DEFAULT_RAM)
    parser.add_argument("--disk", default=DEFAULT_DISK)
    parser.add_argument("--job-name", default="grugmoe-real-checkpoint-levanter-vllm-compare")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in this process instead of submitting a TPU job.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.region != DEFAULT_REGION:
        raise ValueError(f"This smoke is pinned to {DEFAULT_REGION}; got {args.region!r}")
    config = CompareConfig(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        vllm_result_path=args.vllm_result_path,
        output_dir=args.output_dir or _default_output_dir(),
        cache_dir=args.cache_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        decode_seq_len=args.decode_seq_len,
        overwrite=args.overwrite,
        fail_on_mismatch=args.fail_on_mismatch,
    )
    if args.local:
        run_compare(config)
    else:
        submit_compare(
            config,
            tpu_type=args.tpu_type,
            region=args.region,
            ram=args.ram,
            disk=args.disk,
            job_name=args.job_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
