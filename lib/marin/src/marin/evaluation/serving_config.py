# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure conversion from evaluation serving policy to inference configurations."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

from fray.types import ResourceConfig, create_environment
from huggingface_hub import hf_hub_download
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import EndpointAccess

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import (
    ModelConfig,
    ServeBackend,
    has_vllm_option,
    resolve_serve_variant,
    serve_config_vllm_args,
)
from marin.inference.config import (
    BrokerConfig,
    IrisConfig,
    LevanterEngineConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)

logger = logging.getLogger(__name__)

ENDPOINT_READY_TIMEOUT_SECONDS = 2400
EVAL_SERVE_MAX_NUM_BATCHED_TOKENS = 512
_QUIET_VLLM_ARGS = ("--uvicorn-log-level", "warning")


@dataclass(frozen=True)
class ServeSpec:
    """Serving topology and resources for one evaluation group."""

    backend: ServeBackend = ServeBackend.VLLM
    tpu_type: str | None = "v6e-8"
    gpu_type: str | None = None
    gpu_count: int | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    dtype: str = "bfloat16"
    region: str | None = None
    serve_cpu: float = 8.0
    serve_memory: str = "64g"
    serve_disk: str = "100g"
    max_num_batched_tokens: int = EVAL_SERVE_MAX_NUM_BATCHED_TOKENS
    vllm_extra_args: tuple[str, ...] = ()
    chat_template_content: str | None = None
    instances: int = 1
    broker: BrokerConfig | None = None
    endpoint_access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE
    auto_overrides: bool = True

    def __post_init__(self) -> None:
        has_tpu = self.tpu_type is not None
        has_gpu = self.gpu_count is not None
        if has_tpu == has_gpu:
            raise ValueError("ServeSpec requires exactly one of tpu_type or gpu_count")
        if self.gpu_count is not None and self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if self.instances <= 0:
            raise ValueError("instances must be positive")


@dataclass(frozen=True)
class InferenceLaunch:
    """Fully resolved inputs to :func:`marin.inference.iris.remote_inference`."""

    model: ServedModelConfig
    engine: VllmEngineConfig | LevanterEngineConfig
    iris: IrisConfig
    instances: int
    broker: BrokerConfig | None
    endpoint_access: int


@dataclass(frozen=True)
class EvaluationServingConfig:
    """Model and serving policy resolved by the evaluation worker."""

    model: str
    tokenizer: str
    spec: ServeSpec
    revision: str | None = None
    served_model_name: str | None = None


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
    """Inspect an HF config and fill portable vLLM defaults."""
    try:
        with open(hf_hub_download(model, "config.json", revision=revision)) as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Could not inspect config.json for %s; using explicit serving options: %s", model, exc)
        return existing_extra_args, max_model_len
    return _auto_serve_overrides_from_config(model, config, max_model_len, existing_extra_args)


def build_inference_launch(
    model: str,
    tokenizer: str,
    spec: ServeSpec,
    *,
    env_vars: Mapping[str, str],
    revision: str | None = None,
    served_model_name: str | None = None,
) -> InferenceLaunch:
    """Resolve ``spec`` into the shared inference layer's model, engine, and Iris configs."""
    vllm_extra_args = spec.vllm_extra_args
    max_model_len = spec.max_model_len
    if spec.backend is ServeBackend.VLLM and spec.auto_overrides:
        vllm_extra_args, max_model_len = auto_serve_overrides(
            model,
            spec.max_model_len,
            spec.vllm_extra_args,
            revision=revision,
        )

    if spec.gpu_count is not None:
        resources = ResourceConfig.with_gpu(
            spec.gpu_type or "H100",
            count=spec.gpu_count,
            cpu=spec.serve_cpu,
            ram=spec.serve_memory,
            disk=spec.serve_disk,
            regions=[spec.region] if spec.region else None,
        )
        if spec.backend is ServeBackend.VLLM:
            engine: VllmEngineConfig | LevanterEngineConfig = VllmEngineConfig(
                launcher=VllmLauncherType.CUDA,
                source=VllmSource.MARIN_FORK,
                startup_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
                max_num_batched_tokens=spec.max_num_batched_tokens,
                extra_args=(*vllm_extra_args, *_QUIET_VLLM_ARGS),
            )
            environment = create_environment(
                setup_scripts=[default_setup_script(packages=["marin-core"])],
                env_vars=dict(env_vars),
            )
        else:
            engine = LevanterEngineConfig()
            environment = create_environment(extras=["gpu"], env_vars=dict(env_vars))
    else:
        assert spec.tpu_type is not None
        resources = ResourceConfig.with_tpu(
            spec.tpu_type,
            cpu=spec.serve_cpu,
            ram=spec.serve_memory,
            disk=spec.serve_disk,
            regions=[spec.region] if spec.region else None,
        )
        if spec.backend is ServeBackend.VLLM:
            engine = VllmEngineConfig(
                startup_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
                max_num_batched_tokens=spec.max_num_batched_tokens,
                extra_args=(*vllm_extra_args, *_QUIET_VLLM_ARGS),
            )
            environment = create_environment(extras=["tpu", "vllm"], env_vars=dict(env_vars))
        else:
            engine = LevanterEngineConfig()
            environment = create_environment(extras=["tpu"], env_vars=dict(env_vars))

    return InferenceLaunch(
        model=ServedModelConfig(
            model=model,
            revision=revision,
            served_model_name=served_model_name,
            tokenizer=tokenizer,
            dtype=spec.dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=spec.tensor_parallel_size,
            chat_template_content=spec.chat_template_content,
        ),
        engine=engine,
        iris=IrisConfig(
            worker_resources=resources,
            worker_environment=environment,
            endpoint_ready_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
        ),
        instances=spec.instances,
        broker=spec.broker,
        endpoint_access=spec.endpoint_access,
    )


def resolve_inference_launch(
    config: EvaluationServingConfig,
    env_vars: Mapping[str, str],
) -> InferenceLaunch:
    """Resolve a worker payload into shared inference-layer inputs."""
    return build_inference_launch(
        config.model,
        config.tokenizer,
        config.spec,
        env_vars=env_vars,
        revision=config.revision,
        served_model_name=config.served_model_name,
    )


def serve_spec_for_model(model: ModelConfig, accelerator: AcceleratorChoice) -> ServeSpec:
    """Map a canonical model and selected accelerator onto serving policy."""
    serve = resolve_serve_variant(model.serve, accelerator.label)
    extra_args = serve_config_vllm_args(serve)
    if model.revision is not None:
        extra_args = (*extra_args, "--revision", model.revision)
    common = dict(
        backend=serve.backend,
        tensor_parallel_size=serve.tensor_parallel_size,
        max_model_len=serve.max_model_len,
        region=accelerator.region,
        vllm_extra_args=extra_args,
        chat_template_content=serve.chat_template,
        auto_overrides=serve.auto_overrides,
    )
    if accelerator.platform is Platform.GPU:
        spec = ServeSpec(
            tpu_type=None,
            gpu_type=accelerator.gpu_type,
            gpu_count=accelerator.gpu_count,
            **common,
        )
    else:
        spec = ServeSpec(
            tpu_type=accelerator.tpu_type,
            gpu_type=None,
            gpu_count=None,
            **common,
        )
    if serve.serve_memory is not None:
        spec = replace(spec, serve_memory=serve.serve_memory)
    return spec
