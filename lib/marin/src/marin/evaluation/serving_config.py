# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower evaluation model policy into remote-inference configuration."""

import json
import logging
import math
from collections.abc import Mapping
from functools import cache
from pathlib import Path, PurePosixPath

from fray.types import ResourceConfig, create_environment
from huggingface_hub import HfApi, hf_hub_download
from iris.cluster.setup_scripts import default_setup_script
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ServeBackend, ServeConfig, has_vllm_option, serve_config_vllm_args
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

logger = logging.getLogger(__name__)

ENDPOINT_READY_TIMEOUT_SECONDS = 2400
EVAL_SERVE_MAX_NUM_BATCHED_TOKENS = 512
DEFAULT_SERVE_CPU = 8.0
_HF_CONFIG_FILENAME = "config.json"
_MAX_POSITION_EMBEDDINGS_KEY = "max_position_embeddings"
_QWEN_NEXT_MODEL_MARKERS = ("qwen3.5", "qwen3-next")
DEFAULT_SERVE_DISK = "100g"
_QUIET_VLLM_ARGS = ("--uvicorn-log-level", "warning")

# vLLM loads a checkpoint through host memory: the weight files land in the page cache and the
# loader stages shard buffers on the way to device memory. Both are charged to the serve child's
# cgroup, so its host-memory request has to cover the checkpoint plus per-rank runtime.
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin")
_BYTES_PER_GIB = 1024**3
# Loader buffers and not-yet-reclaimed page cache on top of the checkpoint's own bytes.
_CHECKPOINT_MEMORY_FACTOR = 1.5
# CUDA context, torch runtime, and compile workers for one model-parallel rank on the host.
_HOST_MEMORY_PER_RANK_GB = 12
# Small models are dominated by the runtime rather than the weights; do not request less than this.
_MIN_SERVE_MEMORY_GB = 32


def _auto_serve_overrides_from_config(
    model: str,
    config: dict,
    max_model_len: int | None,
    existing_extra_args: tuple[str, ...],
) -> tuple[tuple[str, ...], int | None]:
    """Derive portable text-eval vLLM flags without overriding explicit choices."""
    serialized_config = json.dumps(config).lower()
    architectures = " ".join(config.get("architectures") or ()).lower()
    model_lower = model.lower()
    is_qwen_next = any(marker in model_lower for marker in _QWEN_NEXT_MODEL_MARKERS)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config

    derived: tuple[tuple[str, str], ...] = ()
    if (
        "gated_delta_net" in serialized_config
        or "linear_attn" in serialized_config
        or "qwen3next" in architectures
        or "qwen3_5" in architectures
        or is_qwen_next
    ):
        derived += (("--gdn-prefill-backend", "triton"),)
    if config.get("vision_config") or "forconditionalgeneration" in architectures:
        derived += (("--limit-mm-per-prompt", '{"image":0,"video":0}'),)
    if "qwen" in model_lower and ("thinking" in model_lower or is_qwen_next):
        derived += (("--reasoning-parser", "qwen3"),)

    merged = list(existing_extra_args)
    for option, value in derived:
        if not has_vllm_option(existing_extra_args, option):
            merged.extend((option, value))

    native_max_model_len = text_config.get(_MAX_POSITION_EMBEDDINGS_KEY) or config.get(_MAX_POSITION_EMBEDDINGS_KEY)
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
        config_path = StoragePath(model) / _HF_CONFIG_FILENAME
    elif Path(model).is_dir():
        config_path = StoragePath(str(Path(model) / _HF_CONFIG_FILENAME))
    else:
        config_path = StoragePath(hf_hub_download(model, _HF_CONFIG_FILENAME, revision=revision))
    config = json.loads(config_path.read_text())
    return _auto_serve_overrides_from_config(model, config, max_model_len, existing_extra_args)


def _is_weight_file(name: str) -> bool:
    """Whether a checkpoint entry, named relative to its root, is a weight file the server loads.

    Only top-level entries count. A Hugging Face repo often ships a second copy of the same weights
    in a subdirectory (``original/`` on the Llama repos) that vLLM never reads; counting it would
    roughly double the measured size of the checkpoint.
    """
    return "/" not in name and name.endswith(_WEIGHT_FILE_SUFFIXES)


@cache
def _checkpoint_file_sizes(location: str, revision: str | None) -> dict[str, int]:
    """Byte size of every file at ``location``, keyed by name relative to the checkpoint root.

    ``location`` is an object-store export directory, a local directory, or a Hugging Face repo id,
    matching :func:`auto_serve_overrides`. Reads metadata only: a directory listing or the Hub's
    file-metadata API, never the weights themselves. The two directory listings are shallow; the Hub
    reports a repo's whole tree, so its names can carry a subdirectory prefix.
    """
    if "://" in location:
        fs, path = filesystem_for(location)
        return {PurePosixPath(entry["name"]).name: entry.get("size") or 0 for entry in fs.ls(path, detail=True)}
    directory = Path(location)
    if directory.is_dir():
        return {child.name: child.stat().st_size for child in directory.iterdir() if child.is_file()}
    info = HfApi().model_info(location, revision=revision, files_metadata=True)
    return {sibling.rfilename: sibling.size or 0 for sibling in info.siblings or ()}


def _serve_host_memory(files: Mapping[str, int], ranks: int) -> str:
    """Host memory an inference worker needs to load the checkpoint described by ``files``.

    The weights count once per host, not once per rank: page cache is charged to the cgroup once no
    matter how many ranks map the same shards.
    """
    weight_bytes = sum(size for name, size in files.items() if _is_weight_file(name))
    if not weight_bytes:
        raise ValueError(
            f"no {' or '.join(_WEIGHT_FILE_SUFFIXES)} weight files among {sorted(files)}, so the serve child's "
            "host memory cannot be sized from the checkpoint; set resource_hint.memory for this model"
        )
    weight_gb = weight_bytes / _BYTES_PER_GIB
    required_gb = _CHECKPOINT_MEMORY_FACTOR * weight_gb + _HOST_MEMORY_PER_RANK_GB * ranks
    return f"{max(_MIN_SERVE_MEMORY_GB, math.ceil(required_gb))}g"


def _serve_memory(model: ModelConfig, accelerator: AcceleratorChoice) -> str:
    """The inference worker's host-memory request, from the model's own hint or its checkpoint."""
    if model.resource_hint.memory is not None:
        return model.resource_hint.memory
    # A TPU slice serves one host process for the whole slice; a GPU slice runs one rank per device.
    ranks = accelerator.gpu_count if accelerator.platform is Platform.GPU else 1
    memory = _serve_host_memory(_checkpoint_file_sizes(model.location, model.revision), ranks)
    logger.info("Sized %s serve host memory at %s from its checkpoint (%d ranks)", model.name, memory, ranks)
    return memory


def _vllm_engine_config(
    serve: ServeConfig,
    platform: Platform,
    extra_args: tuple[str, ...],
) -> VllmEngineConfig:
    """Build the evaluation engine settings shared by GPU and TPU workers."""
    launcher = VllmLauncherType.CUDA if platform is Platform.GPU else VllmLauncherType.TPU
    source = VllmSource.MARIN_FORK if platform is Platform.GPU else VllmSource.UPSTREAM
    return VllmEngineConfig(
        launcher=launcher,
        source=source,
        startup_timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS,
        max_num_batched_tokens=(
            serve.max_num_batched_tokens
            if serve.max_num_batched_tokens is not None
            else EVAL_SERVE_MAX_NUM_BATCHED_TOKENS
        ),
        max_num_seqs=serve.max_num_seqs,
        extra_args=(*extra_args, *_QUIET_VLLM_ARGS),
    )


def inference_config_for_model(
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    *,
    env_vars: Mapping[str, str],
    priority: int,
    capability_origin: str | None = None,
    api_model: str | None = None,
    instances: int = 1,
    broker: BrokerConfig | None = None,
) -> RemoteInferenceConfig:
    """Lower one model and selected accelerator into remote inference configuration."""
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
    memory = _serve_memory(model, accelerator)
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
            engine: VllmEngineConfig | LevanterEngineConfig = _vllm_engine_config(
                serve, accelerator.platform, extra_args
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
            engine = _vllm_engine_config(serve, accelerator.platform, extra_args)
            environment = create_environment(extras=["tpu"], env_vars=dict(env_vars))
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
            priority=priority,
        ),
        instances=instances,
        broker=broker,
        capability_origin=capability_origin,
    )
