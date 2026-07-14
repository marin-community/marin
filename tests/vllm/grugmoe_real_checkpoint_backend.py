# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend subprocess entrypoints for the GrugMoE real-checkpoint e2e.

Heavy TPU/JAX/vLLM imports stay inside explicitly-run backend phases so pytest
collection remains cheap.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata as md
import importlib.util
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rigging.filesystem import StoragePath

EUROPE_WEST4_GCS_PREFIX = "gs://marin-eu-west4/"
REGION = "europe-west4"
TPU_TYPE = "v6e-4"
CHECKPOINT_PATH = "gs://marin-eu-west4/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980"
TOKENIZER_PATH = "gs://marin-eu-west4/tokenizers/meta-llama/Meta-Llama-3.1-8B/hf-hub-0.36.0"
OUTPUT_ROOT = "gs://marin-eu-west4/tmp/ttl=14d/grugmoe-real-checkpoint-e2e"
CACHE_ROOT = "gs://marin-eu-west4/compilation-cache/grugmoe-real-checkpoint-e2e"
PROMPT = (
    "Answer with digits only. No words. No punctuation. What is the Answer to the Ultimate Question of Life, "
    "the Universe, and Everything?"
)
EXPECTED_CONTINUATION = " The Ultimate. The Ultimate. The Ultimate"
MAX_MODEL_LEN = 128
MAX_NUM_BATCHED_TOKENS = 128
MAX_NEW_TOKENS = 8
LEVANTER_PROMPT_ADD_SPECIAL_TOKENS = True
EXPECTED_PROMPT_TOKEN_COUNT = 29
DECODE_SEQ_LEN = 128
SERVER_TIMEOUT_SECONDS = 1800
SERVED_MODEL_NAME = "grugmoe-real-checkpoint-e2e"
VLLM_DTYPE = "bfloat16"
MAX_SHARD_SIZE = 256 * 1024 * 1024
_REAL_CHECKPOINT_HIDDEN_DIM = 512


@dataclass(frozen=True)
class E2EPaths:
    output_dir: str
    cache_dir: str
    artifact_dir: str
    export_result_path: str
    vllm_result_path: str
    levanter_result_path: str
    summary_result_path: str


@dataclass(frozen=True)
class StagedArtifact:
    vllm_model_path: str
    staging: dict[str, Any]


def _join_path(base: str, *parts: str) -> str:
    parsed = urlparse(base)
    if parsed.scheme in {"", "file"}:
        return os.path.join(base, *parts)
    return posixpath.join(base.rstrip("/"), *parts)


def _fs_path(path: str):
    import fsspec  # noqa: PLC0415

    return fsspec.core.url_to_fs(path)


def _exists(path: str) -> bool:
    return StoragePath(path).exists()


def _remove_tree(path: str) -> None:
    target = StoragePath(path)
    if target.exists():
        target.rmtree()


def _write_json(path: str, payload: dict[str, Any]) -> None:
    parent = path.rsplit("/", 1)[0]
    StoragePath(parent).mkdirs()
    StoragePath(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: str) -> dict[str, Any]:
    _require_europe_west4_path("json_path", path)
    payload = json.loads(StoragePath(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _require_europe_west4_path(label: str, path: str) -> None:
    parsed = urlparse(path)
    if parsed.scheme == "gs":
        if path.startswith(EUROPE_WEST4_GCS_PREFIX):
            return
        raise ValueError(f"{label} must be under {EUROPE_WEST4_GCS_PREFIX}, got {path!r}")
    if parsed.scheme in {"", "file"}:
        return
    raise ValueError(f"{label} must be a local path or {EUROPE_WEST4_GCS_PREFIX} path, got {path!r}")


def _require_file(label: str, path: str) -> None:
    _require_europe_west4_path(label, path)
    if not _exists(path):
        raise FileNotFoundError(f"{label} not found at {path}")


def _require_constants_are_europe_west4(paths: E2EPaths | None = None) -> None:
    if REGION != "europe-west4":
        raise ValueError(f"GrugMoE e2e region must be europe-west4, got {REGION!r}")
    for label, path in {
        "checkpoint_path": CHECKPOINT_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "output_root": OUTPUT_ROOT,
        "cache_root": CACHE_ROOT,
        **(
            {
                "output_dir": paths.output_dir,
                "cache_dir": paths.cache_dir,
                "artifact_dir": paths.artifact_dir,
                "export_result_path": paths.export_result_path,
                "vllm_result_path": paths.vllm_result_path,
                "levanter_result_path": paths.levanter_result_path,
                "summary_result_path": paths.summary_result_path,
            }
            if paths is not None
            else {}
        ),
    }.items():
        _require_europe_west4_path(label, path)


def _metadata_text(path: str, *, timeout_seconds: float = 1.0) -> str | None:
    request = Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except (OSError, URLError, TimeoutError):
        return None


def _runtime_region() -> str | None:
    for env_name in ("MARIN_TPU_REGION", "TPU_REGION", "GOOGLE_CLOUD_REGION", "CLOUD_ML_REGION"):
        value = os.environ.get(env_name)
        if value:
            return value

    zone = _metadata_text("instance/zone")
    if zone:
        zone_name = zone.rsplit("/", 1)[-1]
        if "-" in zone_name:
            return zone_name.rsplit("-", 1)[0]
    return None


def _require_runtime_region() -> None:
    region = _runtime_region()
    if region != REGION:
        raise RuntimeError(
            f"GrugMoE real-checkpoint e2e must run in {REGION}; detected {region!r}. "
            "Run it on a v6e-4 TPU VM in europe-west4 or set MARIN_TPU_REGION for explicit validation."
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        return f"unavailable:{exc!r}"


def _direct_url(package: str) -> str:
    try:
        direct_url = md.distribution(package).read_text("direct_url.json")
    except md.PackageNotFoundError:
        return "not-installed"
    return direct_url.strip() if direct_url else ""


def _version(package: str) -> str:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not-installed"


def _runtime_snapshot(*, include_jax_devices: bool = False, include_grugmoe_spec: bool = False) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "marin_sha": os.environ.get("MARIN_GIT_SHA") or _git_sha(),
        "region": _runtime_region(),
        "tpu_type": TPU_TYPE,
        "packages": {
            package: {"version": _version(package), "direct_url": _direct_url(package)}
            for package in ("marin-core", "vllm", "tpu-inference", "jax", "libtpu")
        },
    }
    if include_grugmoe_spec:
        try:
            snapshot["grugmoe_spec"] = repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe"))
        except ModuleNotFoundError as exc:
            snapshot["grugmoe_spec"] = f"unavailable:{exc!r}"
    if include_jax_devices:
        import jax  # noqa: PLC0415

        snapshot.update(
            {
                "jax_process_index": jax.process_index(),
                "jax_process_count": jax.process_count(),
                "jax_local_device_count": jax.local_device_count(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    return snapshot


def _real_checkpoint_model_config():
    from experiments.grug.moe.model import GrugModelConfig  # noqa: PLC0415

    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=_REAL_CHECKPOINT_HIDDEN_DIM,
        intermediate_dim=256,
        shared_expert_intermediate_dim=512,
        num_experts=256,
        num_experts_per_token=4,
        num_layers=6,
        num_heads=4,
        num_kv_heads=1,
        max_seq_len=4096,
        sliding_window=2048,
        initializer_std=0.5 / (_REAL_CHECKPOINT_HIDDEN_DIM**0.5),
        qk_mult=1.3,
        router_z_loss_coef=0.0,
    )


# This adapter exists only because the pinned real checkpoint predates the
# current fused GrugMoE expert layout. Keep it backend-local unless another
# production export path needs to support legacy split-expert checkpoints.
def _legacy_split_expert_inference_state_dict(model: Any, cfg: Any, prefix: str | None = None) -> dict[str, Any]:
    from experiments.grug.moe.model import (  # noqa: PLC0415
        _linear_inference_tensor,
        _with_state_dict_prefix,
    )

    tensors: dict[str, Any] = {
        "model.embed_tokens.weight": model.token_embed,
        "model.embed_norm.weight": model.embed_norm.weight,
        "model.embed_gated_norm.down_proj.weight": _linear_inference_tensor(model.embed_gated_norm.w_down),
        "model.embed_gated_norm.up_proj.weight": _linear_inference_tensor(model.embed_gated_norm.w_up),
        "model.norm.weight": model.final_norm.weight,
        "model.final_gated_norm.down_proj.weight": _linear_inference_tensor(model.final_gated_norm.w_down),
        "model.final_gated_norm.up_proj.weight": _linear_inference_tensor(model.final_gated_norm.w_up),
        "lm_head.weight": _linear_inference_tensor(model.output_proj),
    }
    for layer_index, block in enumerate(model.blocks):
        layer_prefix = f"model.layers.{layer_index}"
        tensors.update(
            {
                f"{layer_prefix}.input_layernorm.weight": block.rms_attn.weight,
                f"{layer_prefix}.attn_gated_norm.down_proj.weight": _linear_inference_tensor(
                    block.attn_gated_norm.w_down
                ),
                f"{layer_prefix}.attn_gated_norm.up_proj.weight": _linear_inference_tensor(block.attn_gated_norm.w_up),
                f"{layer_prefix}.self_attn.q_proj.weight": _linear_inference_tensor(block.attn.w_q),
                f"{layer_prefix}.self_attn.k_proj.weight": _linear_inference_tensor(block.attn.w_k),
                f"{layer_prefix}.self_attn.v_proj.weight": _linear_inference_tensor(block.attn.w_v),
                f"{layer_prefix}.self_attn.o_proj.weight": _linear_inference_tensor(block.attn.w_o),
                f"{layer_prefix}.self_attn.attn_gate.weight": _linear_inference_tensor(block.attn.attn_gate),
                f"{layer_prefix}.post_attention_layernorm.weight": block.rms_mlp.weight,
                f"{layer_prefix}.mlp_gated_norm.down_proj.weight": _linear_inference_tensor(block.mlp_gated_norm.w_down),
                f"{layer_prefix}.mlp_gated_norm.up_proj.weight": _linear_inference_tensor(block.mlp_gated_norm.w_up),
                f"{layer_prefix}.mlp.router.weight": _linear_inference_tensor(block.mlp.router),
                f"{layer_prefix}.mlp.router.bias": block.mlp.router_bias,
                f"{layer_prefix}.mlp.experts.gate_proj.weight": _linear_inference_tensor(block.mlp.w_gate),
                f"{layer_prefix}.mlp.experts.up_proj.weight": _linear_inference_tensor(block.mlp.w_up),
                f"{layer_prefix}.mlp.experts.down_proj.weight": _linear_inference_tensor(block.mlp.w_down),
            }
        )
        if block.shared is not None:
            tensors.update(
                {
                    f"{layer_prefix}.shared_expert.gate_proj.weight": _linear_inference_tensor(block.shared.w_gate),
                    f"{layer_prefix}.shared_expert.up_proj.weight": _linear_inference_tensor(block.shared.w_up),
                    f"{layer_prefix}.shared_expert.down_proj.weight": _linear_inference_tensor(block.shared.w_down),
                }
            )
    return {_with_state_dict_prefix(prefix, name): value for name, value in tensors.items()}


def _load_legacy_split_expert_checkpoint(checkpoint_path: str, model_cfg: Any):
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from haliax import Axis  # noqa: PLC0415
    from levanter.checkpoint import latest_checkpoint_path, load_checkpoint  # noqa: PLC0415
    from levanter.utils.jax_utils import is_inexact_arrayish  # noqa: PLC0415

    class LegacySplitMoEMLP(eqx.Module):
        router: jax.Array
        router_bias: jax.Array
        w_gate: jax.Array
        w_up: jax.Array
        w_down: jax.Array
        cfg: Any = eqx.field(static=True)

    def legacy_split_expert_template(cfg: Any, vocab: Axis, *, key: jax.Array):
        model = cfg.build(vocab, key=key)
        for layer_index in range(cfg.num_layers):
            expert = model.blocks[layer_index].mlp.expert_mlp
            gate, up = jnp.split(expert.w_gate_up, [cfg.intermediate_dim], axis=-1)
            original_mlp = model.blocks[layer_index].mlp
            split_mlp = LegacySplitMoEMLP(
                router=original_mlp.router,
                router_bias=original_mlp.router_bias,
                w_gate=gate,
                w_up=up,
                w_down=expert.w_down,
                cfg=original_mlp.cfg,
            )
            model = eqx.tree_at(lambda m, i=layer_index: m.blocks[i].mlp, model, split_mlp)
        return model

    vocab = Axis("vocab", model_cfg.vocab_size)
    key = jax.random.PRNGKey(0)
    model = eqx.filter_eval_shape(legacy_split_expert_template, model_cfg, vocab, key=key)
    trainable, non_trainable = eqx.partition(model, is_inexact_arrayish)
    latest_checkpoint = latest_checkpoint_path(checkpoint_path)
    print("loading_grugmoe_real_checkpoint=" + str(latest_checkpoint), flush=True)
    trainable = load_checkpoint(trainable, latest_checkpoint, subpath="params")
    if trainable is None:
        raise RuntimeError(f"Failed to load trainable params from {latest_checkpoint}")
    return eqx.combine(trainable, non_trainable)


def _export_backend(args: argparse.Namespace) -> None:
    _require_constants_are_europe_west4(
        E2EPaths(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
            export_result_path=args.result_path,
            vllm_result_path=_join_path(args.output_dir, "vllm-result.json"),
            levanter_result_path=_join_path(args.output_dir, "levanter-result.json"),
            summary_result_path=_join_path(args.output_dir, "result.json"),
        )
    )
    _require_file("checkpoint metadata", _join_path(args.checkpoint_path, "metadata.json"))
    _require_file("tokenizer.json", _join_path(args.tokenizer_path, "tokenizer.json"))
    if _exists(args.artifact_dir):
        _remove_tree(args.artifact_dir)

    import equinox as eqx  # noqa: PLC0415
    import haliax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    from haliax import Axis  # noqa: PLC0415
    from haliax.partitioning import set_mesh  # noqa: PLC0415
    from levanter.compat.hf_checkpoints import load_tokenizer  # noqa: PLC0415
    from levanter.grug.sharding import compact_grug_mesh  # noqa: PLC0415

    class LegacySplitExpertExportModel(eqx.Module):
        model: Any
        config: Any = eqx.field(static=True)

        @property
        def Vocab(self) -> Axis:
            return Axis("vocab", self.config.vocab_size)

        def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
            return _legacy_split_expert_inference_state_dict(self.model, self.config, prefix=prefix)

    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.cache_dir)
    model_cfg = _real_checkpoint_model_config()
    mesh = compact_grug_mesh()
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(args.checkpoint_path, model_cfg)
        tokenizer = load_tokenizer(args.tokenizer_path)
        converter = model_cfg.hf_checkpoint_converter().replaced(tokenizer=tokenizer)
        converter.save_pretrained(
            LegacySplitExpertExportModel(loaded_model, model_cfg),
            args.artifact_dir,
            save_tokenizer=True,
            max_shard_size=MAX_SHARD_SIZE,
        )
    _require_file("exported config.json", _join_path(args.artifact_dir, "config.json"))
    _require_file("exported tokenizer.json", _join_path(args.artifact_dir, "tokenizer.json"))
    result = {
        "phase": "export",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "result_path": args.result_path,
        "model_config": dataclasses.asdict(model_cfg),
        "runtime": _runtime_snapshot(include_jax_devices=True, include_grugmoe_spec=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_real_checkpoint_export_result=" + json.dumps(result, sort_keys=True), flush=True)


def _local_filesystem_path(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme == "file":
        return parsed.path
    if parsed.scheme == "":
        return path
    raise ValueError(f"{path!r} is not a local filesystem path")


def _copy_tree_to_local(source_dir: str, local_dir: str) -> int:
    _require_europe_west4_path("artifact_dir", source_dir)
    fs, source_path = _fs_path(source_dir)
    if not fs.exists(source_path):
        raise FileNotFoundError(f"artifact_dir not found at {source_dir}")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    copied = 0
    for source_file in fs.find(source_path):
        rel_path = posixpath.relpath(source_file, source_path)
        local_file = os.path.join(local_dir, *rel_path.split("/"))
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with fs.open(source_file, "rb") as src, open(local_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"artifact_dir contained no files: {source_dir}")
    return copied


def _stage_artifact_for_vllm(artifact_dir: str) -> StagedArtifact:
    _require_europe_west4_path("artifact_dir", artifact_dir)
    parsed = urlparse(artifact_dir)
    if parsed.scheme in {"", "file"}:
        local_path = _local_filesystem_path(artifact_dir)
        return StagedArtifact(
            vllm_model_path=local_path,
            staging={
                "staged": False,
                "source_artifact_dir": artifact_dir,
                "vllm_model_path": local_path,
                "copied_files": None,
            },
        )

    local_root = tempfile.mkdtemp(prefix="grugmoe-real-checkpoint-vllm-artifact-")
    local_path = os.path.join(local_root, "artifact")
    copied_files = _copy_tree_to_local(artifact_dir, local_path)
    _require_file("staged artifact config.json", _join_path(local_path, "config.json"))
    _require_file("staged artifact tokenizer.json", _join_path(local_path, "tokenizer.json"))
    return StagedArtifact(
        vllm_model_path=local_path,
        staging={
            "staged": True,
            "source_artifact_dir": artifact_dir,
            "vllm_model_path": local_path,
            "copied_files": copied_files,
        },
    )


def _completion_payload(env: Any) -> dict[str, Any]:
    import requests  # noqa: PLC0415

    if env.model_id is None:
        raise RuntimeError("Expected vLLM server to expose a model id.")
    response = requests.post(
        f"{env.server_url}/completions",
        json={
            "model": env.model_id,
            "prompt": PROMPT,
            "temperature": 0.0,
            "max_tokens": MAX_NEW_TOKENS,
        },
        timeout=300,
    )
    print("vllm_completions_status_code=" + str(response.status_code), flush=True)
    if not response.ok:
        print("vllm_completions_response_text=" + response.text[:4000], flush=True)
        print("vllm_server_logs_tail_begin", flush=True)
        print(env.logs_tail(max_lines=400), flush=True)
        print("vllm_server_logs_tail_end", flush=True)
        response.raise_for_status()
    return response.json()


def _vllm_backend(args: argparse.Namespace) -> None:
    _require_constants_are_europe_west4(
        E2EPaths(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
            export_result_path=_join_path(args.output_dir, "export-result.json"),
            vllm_result_path=args.result_path,
            levanter_result_path=_join_path(args.output_dir, "levanter-result.json"),
            summary_result_path=_join_path(args.output_dir, "result.json"),
        )
    )
    _require_file("artifact config.json", _join_path(args.artifact_dir, "config.json"))
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["MODEL_IMPL_TYPE"] = "auto"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", args.cache_dir)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    from marin.evaluation.evaluators.evaluator import ModelConfig  # noqa: PLC0415
    from marin.inference.vllm_server import VllmEnvironment  # noqa: PLC0415

    staged_artifact = _stage_artifact_for_vllm(args.artifact_dir)
    model = ModelConfig(
        name=SERVED_MODEL_NAME,
        path=staged_artifact.vllm_model_path,
        engine_kwargs={
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        },
    )
    extra_args = [
        "--runner",
        "generate",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        VLLM_DTYPE,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--enforce-eager",
        "--max-num-seqs",
        "1",
    ]
    started = time.time()
    with VllmEnvironment(model=model, timeout_seconds=SERVER_TIMEOUT_SECONDS, extra_args=extra_args) as env:
        print("vllm_server_initialized=True", flush=True)
        print("vllm_server_url=" + env.server_url, flush=True)
        print("vllm_model_path=" + staged_artifact.vllm_model_path, flush=True)
        print("vllm_artifact_staging=" + json.dumps(staged_artifact.staging, sort_keys=True), flush=True)
        print("vllm_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""), flush=True)
        payload = _completion_payload(env)
        logs_tail = env.logs_tail(max_lines=120)
        model_id = env.model_id
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"expected exactly one completion choice, got {payload!r}")
    completion = str(choices[0].get("text", ""))
    result = {
        "phase": "vllm",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "artifact_dir": args.artifact_dir,
        "result_path": args.result_path,
        "prompt": PROMPT,
        "completion": completion,
        "expected_continuation": EXPECTED_CONTINUATION,
        "passed": completion == EXPECTED_CONTINUATION,
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_model_id": model_id,
        "vllm_model_path": staged_artifact.vllm_model_path,
        "artifact_staging": staged_artifact.staging,
        "vllm_engine_kwargs": model.engine_kwargs,
        "vllm_args": extra_args,
        "raw_response": payload,
        "vllm_logs_tail": logs_tail,
        "runtime": _runtime_snapshot(include_grugmoe_spec=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_real_checkpoint_vllm_result=" + json.dumps(result, sort_keys=True), flush=True)
    if completion != EXPECTED_CONTINUATION:
        raise AssertionError(f"vLLM completion {completion!r} != expected {EXPECTED_CONTINUATION!r}")


def _tokenizer_encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return [int(token_id) for token_id in ids]


def _levanter_prompt_token_ids(tokenizer: Any) -> tuple[list[int], dict[str, Any]]:
    token_ids = _tokenizer_encode(
        tokenizer,
        PROMPT,
        add_special_tokens=LEVANTER_PROMPT_ADD_SPECIAL_TOKENS,
    )
    report = {
        "add_special_tokens": LEVANTER_PROMPT_ADD_SPECIAL_TOKENS,
        "expected_prompt_token_count": EXPECTED_PROMPT_TOKEN_COUNT,
        "prompt_token_count": len(token_ids),
        "prompt_token_ids": token_ids,
        "matches_expected_count": len(token_ids) == EXPECTED_PROMPT_TOKEN_COUNT,
    }
    if len(token_ids) != EXPECTED_PROMPT_TOKEN_COUNT:
        raise ValueError(
            f"Levanter prompt token count {len(token_ids)} != expected {EXPECTED_PROMPT_TOKEN_COUNT}; "
            f"tokenization={report!r}"
        )
    return token_ids, report


def _decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _mesh_batch_axis_size(mesh: Any) -> int:
    size = 1
    for axis_name in ("replica_dcn", "data", "expert"):
        size *= int(mesh.shape.get(axis_name, 1))
    return size


def _selected_logprob(logits: Any, token_id: int) -> float:
    import numpy as np  # noqa: PLC0415

    logits_f64 = logits.astype(np.float64)
    max_logit = np.max(logits_f64)
    log_z = max_logit + np.log(np.exp(logits_f64 - max_logit).sum())
    return float(logits_f64[token_id] - log_z)


def _executable_model_from_legacy_split(model: Any) -> Any:
    import equinox as eqx  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from levanter.grug.grug_moe import MoEExpertMlp  # noqa: PLC0415
    from levanter.utils.activation import ActivationFunctionEnum  # noqa: PLC0415

    from experiments.grug.moe.model import MoEMLP  # noqa: PLC0415

    def executable_mlp_from_legacy_split(split_mlp: Any) -> MoEMLP:
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

    for layer_index, block in enumerate(model.blocks):
        if hasattr(block.mlp, "expert_mlp"):
            continue
        executable_mlp = executable_mlp_from_legacy_split(block.mlp)
        model = eqx.tree_at(lambda m, i=layer_index: m.blocks[i].mlp, model, executable_mlp)
    return model


def _greedy_decode(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    batch_size: int,
    decode_seq_len: int,
    pad_token_id: int,
) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if len(prompt_ids) + max_new_tokens > decode_seq_len:
        raise ValueError(
            f"prompt length {len(prompt_ids)} + max_new_tokens {max_new_tokens} exceeds decode_seq_len {decode_seq_len}"
        )

    def position_logits_batch(the_model: Any, token_ids: Any, position: Any) -> Any:
        return the_model.logits(token_ids)[:, position, :].astype(jnp.float32)

    token_ids_array = np.full((batch_size, decode_seq_len), pad_token_id, dtype=np.int32)
    token_ids_array[:, : len(prompt_ids)] = np.asarray(prompt_ids, dtype=np.int32)
    generated_ids: list[int] = []
    generated_token_texts: list[str] = []
    selected_token_logprobs: list[float] = []
    steps: list[dict[str, Any]] = []
    position_logits = jax.jit(position_logits_batch)
    started = time.time()

    for step_index in range(max_new_tokens):
        position = jnp.asarray(len(prompt_ids) + step_index - 1, dtype=jnp.int32)
        step_logits_batch = position_logits(model, jnp.asarray(token_ids_array, dtype=jnp.int32), position)
        step_logits = np.asarray(jax.device_get(step_logits_batch))[0]
        selected_token_id = int(np.argmax(step_logits, axis=-1))
        selected_text = _decode_one(tokenizer, selected_token_id)
        selected_logprob = _selected_logprob(step_logits, selected_token_id)
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
        token_ids_array[:, len(prompt_ids) + step_index] = selected_token_id

    return {
        "prompt_token_ids": [int(token_id) for token_id in prompt_ids],
        "prompt_token_count": len(prompt_ids),
        "decode_batch_size": batch_size,
        "decode_seq_len": decode_seq_len,
        "pad_token_id": pad_token_id,
        "generated_token_ids": generated_ids,
        "generated_token_texts": generated_token_texts,
        "completion": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "selected_token_logprobs": selected_token_logprobs,
        "steps": steps,
        "elapsed_seconds": time.time() - started,
    }


def _levanter_backend(args: argparse.Namespace) -> None:
    _require_constants_are_europe_west4(
        E2EPaths(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
            export_result_path=_join_path(args.output_dir, "export-result.json"),
            vllm_result_path=_join_path(args.output_dir, "vllm-result.json"),
            levanter_result_path=args.result_path,
            summary_result_path=_join_path(args.output_dir, "result.json"),
        )
    )
    _require_file("checkpoint metadata", _join_path(args.checkpoint_path, "metadata.json"))
    _require_file("tokenizer.json", _join_path(args.tokenizer_path, "tokenizer.json"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.cache_dir)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    import haliax  # noqa: PLC0415
    from haliax.partitioning import set_mesh  # noqa: PLC0415
    from levanter.compat.hf_checkpoints import load_tokenizer  # noqa: PLC0415
    from levanter.grug.sharding import compact_grug_mesh  # noqa: PLC0415

    tokenizer = load_tokenizer(args.tokenizer_path)
    prompt_ids, tokenization = _levanter_prompt_token_ids(tokenizer)
    model_cfg = _real_checkpoint_model_config()
    mesh = compact_grug_mesh()
    decode_batch_size = _mesh_batch_axis_size(mesh)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(args.checkpoint_path, model_cfg)
        model = _executable_model_from_legacy_split(loaded_model)
        decode_result = _greedy_decode(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=decode_batch_size,
            decode_seq_len=DECODE_SEQ_LEN,
            pad_token_id=pad_token_id,
        )
    completion = str(decode_result["completion"])
    result = {
        "phase": "levanter",
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "result_path": args.result_path,
        "prompt": PROMPT,
        "completion": completion,
        "expected_continuation": EXPECTED_CONTINUATION,
        "passed": completion == EXPECTED_CONTINUATION,
        "tokenization": tokenization,
        "decode_result": decode_result,
        "runtime": _runtime_snapshot(include_jax_devices=True, include_grugmoe_spec=True),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.result_path, result)
    print("grugmoe_real_checkpoint_levanter_result=" + json.dumps(result, sort_keys=True), flush=True)
    if completion != EXPECTED_CONTINUATION:
        raise AssertionError(f"Levanter/JAX completion {completion!r} != expected {EXPECTED_CONTINUATION!r}")


def _parse_backend_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal GrugMoE real-checkpoint e2e backend")
    parser.add_argument("--backend", choices=("export", "vllm", "levanter"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-path", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_backend_args(sys.argv[1:] if argv is None else argv)
    _require_runtime_region()
    match args.backend:
        case "export":
            _export_backend(args)
        case "vllm":
            _vllm_backend(args)
        case "levanter":
            _levanter_backend(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
