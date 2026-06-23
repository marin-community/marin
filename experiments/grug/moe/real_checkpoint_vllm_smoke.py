# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GrugMoE real-checkpoint smoke for the TPU vLLM path.

Default mode submits a v6e-4 job in europe-west4. Use ``--local`` only from a
TPU VM that is already in the intended region.

This smoke verifies inference plumbing for a pretrained checkpoint. If greedy
decoding does not return the semantic answer ``42``, keep the calibrated
deterministic continuation as the assertion; this is not an instruction-
following benchmark.
"""

from __future__ import annotations

import dataclasses
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
from urllib.parse import urlparse

import click
import equinox as eqx
import haliax
import jax
import jax.numpy as jnp
import requests
from haliax import Axis
from haliax.partitioning import set_mesh
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.jax_utils import is_inexact_arrayish
from marin.evaluation.evaluators.evaluator import ModelConfig

from experiments.grug.moe.model import (
    GrugModelConfig,
    _linear_inference_tensor,
    _with_state_dict_prefix,
    canonical_grugmoe_tensor_names,
)
from experiments.grug.moe.real_checkpoint_runtime import (
    DEFAULT_DISK,
    DEFAULT_RAM,
    DEFAULT_REGION,
    DEFAULT_TPU_TYPE,
    MARIN_GIT_SHA_ENV,
    default_output_dir,
    exists,
    fs_path,
    git_sha,
    join_path,
    remove_tree,
    require_file,
    require_local_or_europe_west4,
    runtime_snapshot,
    submit_tpu_job,
    write_json,
)

REAL_CHECKPOINT_PATH = "gs://marin-eu-west4/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980"
LLAMA31_TOKENIZER_PATH = "gs://marin-eu-west4/tokenizers/meta-llama/Meta-Llama-3.1-8B/hf-hub-0.36.0"
OUTPUT_ROOT = "gs://marin-eu-west4/tmp/ttl=14d/grugmoe-real-checkpoint-vllm-smoke"
CACHE_ROOT = "gs://marin-eu-west4/compilation-cache/grugmoe-real-checkpoint-vllm-smoke"
PROMPT = (
    "Answer with digits only. No words. No punctuation. What is the Answer to the Ultimate Question of Life, "
    "the Universe, and Everything?"
)
DEFAULT_EXPECTED_OUTPUT = " The Ultimate. The Ultimate. The Ultimate"
DEFAULT_MAX_SHARD_SIZE = 256 * 1024 * 1024
DEFAULT_MAX_MODEL_LEN = 128
DEFAULT_MAX_TOKENS = 8
DEFAULT_SERVER_TIMEOUT_SECONDS = 1800
DEFAULT_VLLM_DTYPE = "bfloat16"
_DEPENDENCY_GROUPS = ["eval", "tpu", "vllm"]
_REAL_CHECKPOINT_HIDDEN_DIM = 512
_VLLM_REGISTRY_PRELOAD_MODULE = "experiments.grug.moe.vllm_registry"
_PRELOAD_MODULES_ENV = "MARIN_VLLM_PRELOAD_MODULES"


@dataclass(frozen=True)
class SmokeConfig:
    phase: str
    checkpoint_path: str
    tokenizer_path: str
    output_dir: str
    cache_dir: str
    prompt: str
    expected_output: str | None
    calibrate_output: bool
    overwrite: bool
    max_shard_size: int
    max_model_len: int
    max_tokens: int
    server_port: int
    server_timeout_seconds: int
    vllm_dtype: str

    @property
    def artifact_dir(self) -> str:
        return join_path(self.output_dir, "artifact")

    @property
    def result_path(self) -> str:
        return join_path(self.output_dir, "result.json")


@dataclass(frozen=True)
class StagedArtifact:
    vllm_model_path: str
    staging: dict[str, Any]


def validate_locality(config: SmokeConfig) -> dict[str, str]:
    paths = {
        "checkpoint_path": config.checkpoint_path,
        "tokenizer_path": config.tokenizer_path,
        "output_dir": config.output_dir,
        "cache_dir": config.cache_dir,
        "artifact_dir": config.artifact_dir,
        "vllm_model_path": config.artifact_dir,
        "result_path": config.result_path,
    }
    for label, path in paths.items():
        require_local_or_europe_west4(label, path)
    return paths


def _local_filesystem_path(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme == "file":
        return parsed.path
    if parsed.scheme == "":
        return path
    raise ValueError(f"{path!r} is not a local filesystem path")


def _copy_tree_to_local(source_dir: str, local_dir: str) -> int:
    require_local_or_europe_west4("artifact_dir", source_dir)
    fs, source_path = fs_path(source_dir)
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


def stage_artifact_for_vllm(artifact_dir: str) -> StagedArtifact:
    """Return a local vLLM model path, staging GCS artifacts when needed."""
    require_local_or_europe_west4("artifact_dir", artifact_dir)
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
    require_file("staged artifact config.json", join_path(local_path, "config.json"))
    require_file("staged artifact tokenizer.json", join_path(local_path, "tokenizer.json"))
    return StagedArtifact(
        vllm_model_path=local_path,
        staging={
            "staged": True,
            "source_artifact_dir": artifact_dir,
            "vllm_model_path": local_path,
            "copied_files": copied_files,
        },
    )


def real_checkpoint_model_config():
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


def prepare_inputs(config: SmokeConfig) -> dict[str, Any]:
    locality_paths = validate_locality(config)
    require_file("checkpoint metadata", join_path(config.checkpoint_path, "metadata.json"))
    require_file("tokenizer.json", join_path(config.tokenizer_path, "tokenizer.json"))
    return {
        "locality_paths": locality_paths,
        "model_config": dataclasses.asdict(real_checkpoint_model_config()),
        "runtime": runtime_snapshot(include_jax_devices=config.phase == "export", include_grugmoe_spec=True),
    }


class _LegacySplitMoEExpertMlp(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    implementation: Any = eqx.field(static=True)
    activation: ActivationFunctionEnum = eqx.field(static=True)
    capacity_factor: float = eqx.field(static=True)


class _LegacySplitMoEMLP(eqx.Module):
    router: jax.Array
    router_bias: jax.Array
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)


class _LegacySplitExpertExportModel(eqx.Module):
    model: Any
    config: GrugModelConfig = eqx.field(static=True)

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        return legacy_split_expert_inference_state_dict(self.model, self.config, prefix=prefix)


def _split_expert_gate_up(expert_mlp: Any, intermediate_dim: int) -> tuple[jax.Array, jax.Array]:
    if hasattr(expert_mlp, "w_gate_up"):
        return jnp.split(expert_mlp.w_gate_up, [intermediate_dim], axis=-1)
    return expert_mlp.w_gate, expert_mlp.w_up


def _expert_weights(mlp: Any) -> Any:
    return mlp.expert_mlp if hasattr(mlp, "expert_mlp") else mlp


def legacy_split_expert_inference_state_dict(
    model: Any,
    cfg: GrugModelConfig,
    prefix: str | None = None,
) -> dict[str, jax.Array]:
    tensors: dict[str, jax.Array] = {
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
        expert_weights = _expert_weights(block.mlp)
        gate, up = _split_expert_gate_up(expert_weights, cfg.intermediate_dim)
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
                f"{layer_prefix}.mlp.experts.gate_proj.weight": _linear_inference_tensor(gate),
                f"{layer_prefix}.mlp.experts.up_proj.weight": _linear_inference_tensor(up),
                f"{layer_prefix}.mlp.experts.down_proj.weight": _linear_inference_tensor(expert_weights.w_down),
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

    if set(tensors) != canonical_grugmoe_tensor_names(cfg):
        raise ValueError("Legacy GrugMoE export tensor names do not match the canonical schema")

    return {_with_state_dict_prefix(prefix, name): value for name, value in tensors.items()}


def _legacy_split_expert_template(model_cfg: GrugModelConfig, vocab: Axis, *, key: jax.Array):
    model = model_cfg.build(vocab, key=key)
    for layer_index in range(model_cfg.num_layers):
        expert = model.blocks[layer_index].mlp.expert_mlp
        gate, up = jnp.split(expert.w_gate_up, [model_cfg.intermediate_dim], axis=-1)
        split_expert = _LegacySplitMoEExpertMlp(
            w_gate=gate,
            w_up=up,
            w_down=expert.w_down,
            implementation=expert.implementation,
            activation=ActivationFunctionEnum.silu,
            capacity_factor=expert.capacity_factor,
        )
        original_mlp = model.blocks[layer_index].mlp
        split_mlp = _LegacySplitMoEMLP(
            router=original_mlp.router,
            router_bias=original_mlp.router_bias,
            w_gate=split_expert.w_gate,
            w_up=split_expert.w_up,
            w_down=split_expert.w_down,
            cfg=original_mlp.cfg,
        )
        model = eqx.tree_at(lambda m, i=layer_index: m.blocks[i].mlp, model, split_mlp)
    return model


def _load_legacy_split_expert_checkpoint(config: SmokeConfig, model_cfg: GrugModelConfig):
    vocab = Axis("vocab", model_cfg.vocab_size)
    key = jax.random.PRNGKey(0)
    model = eqx.filter_eval_shape(_legacy_split_expert_template, model_cfg, vocab, key=key)
    trainable, non_trainable = eqx.partition(model, is_inexact_arrayish)
    checkpoint_path = latest_checkpoint_path(config.checkpoint_path)
    print("loading_grugmoe_real_checkpoint=" + str(checkpoint_path), flush=True)
    trainable = load_checkpoint(trainable, checkpoint_path, subpath="params")
    if trainable is None:
        raise RuntimeError(f"Failed to load trainable params from {checkpoint_path}")
    return eqx.combine(trainable, non_trainable)


def export_artifact(config: SmokeConfig) -> None:
    if exists(config.output_dir):
        if not config.overwrite:
            raise FileExistsError(f"{config.output_dir} already exists; pass --overwrite to regenerate it")
        remove_tree(config.output_dir)

    model_cfg = real_checkpoint_model_config()
    mesh = compact_grug_mesh()
    with ExitStack() as stack:
        stack.enter_context(set_mesh(mesh))
        stack.enter_context(haliax.axis_mapping({}))
        loaded_model = _load_legacy_split_expert_checkpoint(config, model_cfg)
        tokenizer = load_tokenizer(config.tokenizer_path)
        converter = model_cfg.hf_checkpoint_converter().replaced(tokenizer=tokenizer)
        converter.save_pretrained(
            _LegacySplitExpertExportModel(loaded_model, model_cfg),
            config.artifact_dir,
            save_tokenizer=True,
            max_shard_size=config.max_shard_size,
        )
    require_file("exported config.json", join_path(config.artifact_dir, "config.json"))
    require_file("exported tokenizer.json", join_path(config.artifact_dir, "tokenizer.json"))


def configure_runtime_environment(config: SmokeConfig) -> None:
    require_local_or_europe_west4("cache_dir", config.cache_dir)
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["MODEL_IMPL_TYPE"] = "auto"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", config.cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", config.cache_dir)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    preload_modules = [
        module.strip() for module in os.environ.get(_PRELOAD_MODULES_ENV, "").split(",") if module.strip()
    ]
    if _VLLM_REGISTRY_PRELOAD_MODULE not in preload_modules:
        preload_modules.append(_VLLM_REGISTRY_PRELOAD_MODULE)
    os.environ[_PRELOAD_MODULES_ENV] = ",".join(preload_modules)


def _completion_payload(env: Any, config: SmokeConfig) -> dict[str, Any]:
    if env.model_id is None:
        raise RuntimeError("Expected vLLM server to expose a model id.")
    response = requests.post(
        f"{env.server_url}/completions",
        json={
            "model": env.model_id,
            "prompt": config.prompt,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
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


def serve_artifact(config: SmokeConfig) -> dict[str, Any]:
    require_file("artifact config.json", join_path(config.artifact_dir, "config.json"))
    configure_runtime_environment(config)
    staged_artifact = stage_artifact_for_vllm(config.artifact_dir)

    # VllmEnvironment is serving-phase local; load it after TPU/vLLM env setup.
    from marin.inference.vllm_server import VllmEnvironment  # noqa: PLC0415

    model = ModelConfig(
        name="grugmoe-real-checkpoint-vllm-smoke",
        path=staged_artifact.vllm_model_path,
        engine_kwargs={
            "max_model_len": config.max_model_len,
            "max_num_batched_tokens": config.max_model_len,
        },
    )
    extra_args = [
        "--runner",
        "generate",
        "--dtype",
        config.vllm_dtype,
        "--enforce-eager",
        "--max-num-seqs",
        "1",
    ]

    started = time.time()
    with VllmEnvironment(
        model=model,
        port=config.server_port,
        timeout_seconds=config.server_timeout_seconds,
        extra_args=extra_args,
    ) as env:
        print("vllm_server_initialized=True", flush=True)
        print("vllm_server_url=" + env.server_url, flush=True)
        print("vllm_model_path=" + staged_artifact.vllm_model_path, flush=True)
        print("vllm_artifact_staging=" + json.dumps(staged_artifact.staging, sort_keys=True), flush=True)
        print("vllm_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""), flush=True)
        payload = _completion_payload(env, config)
        logs_tail = env.logs_tail(max_lines=120)
    elapsed = time.time() - started

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"expected exactly one completion choice, got {payload!r}")
    completion = str(choices[0].get("text", ""))
    expected = None if config.calibrate_output else config.expected_output
    calibrated_expected = "42" if completion == "42" else completion
    if expected is None and not config.calibrate_output:
        raise AssertionError(
            "No deterministic output is configured yet. Run once with --calibrate-output, then record the observed "
            "completion in DEFAULT_EXPECTED_OUTPUT or pass --expected-output."
        )
    if expected is not None and completion != expected:
        raise AssertionError(f"completion {completion!r} != expected {expected!r}")

    return {
        "completion": completion,
        "expected_output": expected,
        "configured_expected_output": config.expected_output,
        "calibrated_expected_output": calibrated_expected,
        "calibration_mode": config.calibrate_output,
        "passed": config.calibrate_output or (expected is not None and completion == expected),
        "vllm_model_path": staged_artifact.vllm_model_path,
        "artifact_staging": staged_artifact.staging,
        "elapsed_seconds": elapsed,
        "raw_response": payload,
        "vllm_logs_tail": logs_tail,
    }


def _run_isolated_export(config: SmokeConfig) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.grug.moe.real_checkpoint_vllm_smoke",
        "--local",
        "--phase",
        "export",
        "--checkpoint-path",
        config.checkpoint_path,
        "--tokenizer-path",
        config.tokenizer_path,
        "--output-dir",
        config.output_dir,
        "--cache-dir",
        config.cache_dir,
        "--max-shard-size",
        str(config.max_shard_size),
        "--max-model-len",
        str(config.max_model_len),
        "--max-tokens",
        str(config.max_tokens),
        "--server-port",
        str(config.server_port),
        "--server-timeout-seconds",
        str(config.server_timeout_seconds),
        "--vllm-dtype",
        config.vllm_dtype,
    ]
    if config.overwrite:
        command.append("--overwrite")
    print("grugmoe_real_checkpoint_isolated_export_command=" + json.dumps(command), flush=True)
    subprocess.run(command, check=True)


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    configure_runtime_environment(config)
    preparation = prepare_inputs(config)
    print("grugmoe_real_checkpoint_preflight=" + json.dumps(preparation, sort_keys=True), flush=True)

    if config.phase == "all":
        _run_isolated_export(config)
        print("grugmoe_real_checkpoint_exported_artifact=" + config.artifact_dir, flush=True)
    elif config.phase == "export":
        export_artifact(config)
        print("grugmoe_real_checkpoint_exported_artifact=" + config.artifact_dir, flush=True)

    serve_result: dict[str, Any] | None = None
    if config.phase in {"all", "serve"}:
        serve_result = serve_artifact(config)

    result = {
        "checkpoint_path": config.checkpoint_path,
        "tokenizer_path": config.tokenizer_path,
        "output_dir": config.output_dir,
        "artifact_dir": config.artifact_dir,
        "vllm_model_path": serve_result["vllm_model_path"] if serve_result is not None else config.artifact_dir,
        "prompt": config.prompt,
        "phase": config.phase,
        "locality_paths": preparation["locality_paths"],
        "runtime": preparation["runtime"],
        "serve_result": serve_result,
        "note": (
            "Inference-path smoke only; a calibrated continuation that is not '42' should not be interpreted as an "
            "instruction-following failure."
        ),
    }
    write_json(config.result_path, result)
    print("grugmoe_real_checkpoint_vllm_smoke=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def submit_smoke(config: SmokeConfig, *, tpu_type: str, region: str, ram: str, disk: str, job_name: str) -> None:
    submit_tpu_job(
        config=config,
        entrypoint=run_smoke,
        dependency_groups=_DEPENDENCY_GROUPS,
        base_env_vars={
            MARIN_GIT_SHA_ENV: git_sha(),
            "MODEL_IMPL_TYPE": "auto",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "JAX_COMPILATION_CACHE_DIR": config.cache_dir,
            "VLLM_XLA_CACHE_PATH": config.cache_dir,
            "PYTHONUNBUFFERED": "1",
        },
        tpu_type=tpu_type,
        region=region,
        ram=ram,
        disk=disk,
        job_name=job_name,
        summary_label="submitting_grugmoe_real_checkpoint_smoke",
        summary_fields={
            "output_dir": config.output_dir,
            "artifact_dir": config.artifact_dir,
            "cache_dir": config.cache_dir,
        },
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--local", is_flag=True, help="Run in this process instead of submitting a TPU job.")
@click.option("--phase", type=click.Choice(("all", "preflight", "export", "serve")), default="all", show_default=True)
@click.option("--checkpoint-path", default=REAL_CHECKPOINT_PATH, show_default=True)
@click.option("--tokenizer-path", default=LLAMA31_TOKENIZER_PATH, show_default=True)
@click.option("--output-dir", default=None)
@click.option("--cache-dir", default=CACHE_ROOT, show_default=True)
@click.option("--prompt", default=PROMPT, show_default=True)
@click.option("--expected-output", default=DEFAULT_EXPECTED_OUTPUT, show_default=True)
@click.option(
    "--calibrate-output",
    is_flag=True,
    help="Record the observed deterministic completion instead of asserting a preconfigured one.",
)
@click.option("--overwrite", is_flag=True, help="Remove an existing output dir before export.")
@click.option("--max-shard-size", type=int, default=DEFAULT_MAX_SHARD_SIZE, show_default=True)
@click.option("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN, show_default=True)
@click.option("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, show_default=True)
@click.option("--server-port", type=int, default=8000, show_default=True)
@click.option("--server-timeout-seconds", type=int, default=DEFAULT_SERVER_TIMEOUT_SECONDS, show_default=True)
@click.option("--vllm-dtype", type=click.Choice(("bfloat16", "float32")), default=DEFAULT_VLLM_DTYPE, show_default=True)
@click.option("--tpu-type", default=DEFAULT_TPU_TYPE, show_default=True)
@click.option("--region", default=DEFAULT_REGION, show_default=True)
@click.option("--ram", default=DEFAULT_RAM, show_default=True)
@click.option("--disk", default=DEFAULT_DISK, show_default=True)
@click.option("--job-name", default="grugmoe-real-checkpoint-vllm-smoke", show_default=True)
def _main_command(
    *,
    local: bool,
    phase: str,
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str | None,
    cache_dir: str,
    prompt: str,
    expected_output: str | None,
    calibrate_output: bool,
    overwrite: bool,
    max_shard_size: int,
    max_model_len: int,
    max_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
    vllm_dtype: str,
    tpu_type: str,
    region: str,
    ram: str,
    disk: str,
    job_name: str,
) -> None:
    config = SmokeConfig(
        phase=phase,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir or default_output_dir(OUTPUT_ROOT),
        cache_dir=cache_dir,
        prompt=prompt,
        expected_output=expected_output,
        calibrate_output=calibrate_output,
        overwrite=overwrite,
        max_shard_size=max_shard_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        server_port=server_port,
        server_timeout_seconds=server_timeout_seconds,
        vllm_dtype=vllm_dtype,
    )
    validate_locality(config)
    if local:
        run_smoke(config)
        return
    submit_smoke(
        config,
        tpu_type=tpu_type,
        region=region,
        ram=ram,
        disk=disk,
        job_name=job_name,
    )


def main(argv: list[str] | None = None) -> int:
    _main_command.main(args=argv, standalone_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
