# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower evaluation model policy into exact remote-inference configuration."""

import json
from collections.abc import Mapping
from pathlib import Path

from fray.types import ResourceConfig, create_environment
from huggingface_hub import hf_hub_download
from iris.cluster.setup_scripts import default_setup_script
from rigging.filesystem import StoragePath

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ServeBackend, has_vllm_option, serve_config_vllm_args
from marin.inference.config import (
    BrokerConfig,
    IrisConfig,
    LevanterEngineConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)

ENDPOINT_READY_TIMEOUT_SECONDS = 2400
EVAL_SERVE_MAX_NUM_BATCHED_TOKENS = 512
DEFAULT_SERVE_CPU = 8.0
DEFAULT_SERVE_MEMORY = "64g"
DEFAULT_SERVE_DISK = "100g"
_QUIET_VLLM_ARGS = ("--uvicorn-log-level", "warning")


def _auto_serve_overrides_from_config(
    model: str,
    config: dict,
    max_model_len: int | None,
    existing_extra_args: tuple[str, ...],
) -> tuple[tuple[str, ...], int | None]:
    """Derive portable text-eval vLLM flags without overriding explicit choices."""
    serialized_config = json.dumps(config).lower()
    architectures = " ".join(config.get("architectures") or ()).lower()
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config

    derived: tuple[tuple[str, str], ...] = ()
    if (
        "gated_delta_net" in serialized_config
        or "linear_attn" in serialized_config
        or "qwen3next" in architectures
        or "qwen3_5" in architectures
        or "qwen3.5" in model.lower()
        or "qwen3-next" in model.lower()
    ):
        derived += (("--gdn-prefill-backend", "triton"),)
    if config.get("vision_config") or "forconditionalgeneration" in architectures:
        derived += (("--limit-mm-per-prompt", '{"image":0,"video":0}'),)
    if "qwen" in model.lower() and (
        "thinking" in model.lower() or "qwen3.5" in model.lower() or "qwen3-next" in model.lower()
    ):
        derived += (("--reasoning-parser", "qwen3"),)

    merged = list(existing_extra_args)
    for option, value in derived:
        if not has_vllm_option(existing_extra_args, option):
            merged.extend((option, value))

    native_max_model_len = text_config.get("max_position_embeddings") or config.get("max_position_embeddings")
    if isinstance(native_max_model_len, int | float) and max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))
    return tuple(merged), max_model_len


def auto_serve_overrides(
    model: str,
    max_model_len: int | None,
    existing_extra_args: tuple[str, ...] = (),
    *,
    revision: str | None = None,
) -> tuple[tuple[str, ...], int | None]:
    """Inspect a model's config.json and fill portable vLLM defaults."""
    if "://" in model:
        config_path = StoragePath(model) / "config.json"
    elif Path(model).is_dir():
        config_path = StoragePath(str(Path(model) / "config.json"))
    else:
        config_path = StoragePath(hf_hub_download(model, "config.json", revision=revision))
    config = json.loads(config_path.read_text())
    return _auto_serve_overrides_from_config(model, config, max_model_len, existing_extra_args)


def inference_config_for_model(
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    *,
    env_vars: Mapping[str, str],
    capability_origin: str | None = None,
    api_model: str | None = None,
    instances: int = 1,
    broker: BrokerConfig | None = None,
) -> RemoteInferenceConfig:
    """Lower one model and selected accelerator into the inference subsystem's exact configs."""
    serve = model.serve
    extra_args = serve_config_vllm_args(serve)
    max_model_len = serve.max_model_len
    if serve.backend is ServeBackend.VLLM and serve.auto_overrides:
        extra_args, max_model_len = auto_serve_overrides(
            model.location,
            max_model_len,
            extra_args,
            revision=model.revision,
        )

    hint = model.resource_hint
    cpu = hint.cpu or DEFAULT_SERVE_CPU
    memory = hint.memory or DEFAULT_SERVE_MEMORY
    disk = hint.disk or DEFAULT_SERVE_DISK
    regions = [accelerator.region] if accelerator.region else None

    if accelerator.platform is Platform.GPU:
        resources = ResourceConfig.with_gpu(
            accelerator.gpu_type or "H100",
            count=accelerator.gpu_count,
            cpu=cpu,
            ram=memory,
            disk=disk,
            regions=regions,
        )
        if serve.backend is ServeBackend.VLLM:
            engine: VllmEngineConfig | LevanterEngineConfig = VllmEngineConfig(
                launcher=VllmLauncherType.CUDA,
                source=VllmSource.MARIN_FORK,
                startup_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
                max_num_batched_tokens=EVAL_SERVE_MAX_NUM_BATCHED_TOKENS,
                extra_args=(*extra_args, *_QUIET_VLLM_ARGS),
            )
            environment = create_environment(
                setup_scripts=[default_setup_script(packages=["marin-core"])],
                env_vars=dict(env_vars),
            )
        else:
            engine = LevanterEngineConfig()
            environment = create_environment(extras=["gpu"], env_vars=dict(env_vars))
    else:
        if accelerator.tpu_type is None:
            raise ValueError("TPU accelerator choice requires tpu_type")
        resources = ResourceConfig.with_tpu(
            accelerator.tpu_type,
            cpu=cpu,
            ram=memory,
            disk=disk,
            regions=regions,
        )
        if serve.backend is ServeBackend.VLLM:
            engine = VllmEngineConfig(
                startup_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
                max_num_batched_tokens=EVAL_SERVE_MAX_NUM_BATCHED_TOKENS,
                extra_args=(*extra_args, *_QUIET_VLLM_ARGS),
            )
            environment = create_environment(extras=["tpu", "vllm"], env_vars=dict(env_vars))
        else:
            engine = LevanterEngineConfig()
            environment = create_environment(extras=["tpu"], env_vars=dict(env_vars))

    return RemoteInferenceConfig(
        model=ServedModelConfig(
            weights=model.location,
            revision=model.revision,
            api_model=api_model,
            tokenizer=model.tokenizer or model.location,
            max_model_len=max_model_len,
            tensor_parallel_size=serve.tensor_parallel_size,
            chat_template_content=serve.chat_template,
        ),
        engine=engine,
        iris=IrisConfig(
            worker_resources=resources,
            worker_environment=environment,
            endpoint_ready_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
        ),
        instances=instances,
        broker=broker,
        capability_origin=capability_origin,
    )
