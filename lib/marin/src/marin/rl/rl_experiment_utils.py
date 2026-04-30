# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import importlib
import logging
import uuid
from urllib.parse import urlparse

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointDebugConfig, CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers.attention import AttentionBackend
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.fsspec_utils import join_path
from levanter.utils.mesh import MeshConfig

from marin.execution.executor import ExecutorMainConfig, ExecutorStep, InputName, output_path_of, this_output_path
from marin.execution.remote import remote
from marin.rl.curriculum import CurriculumConfig
from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.placement import marin_prefix_for_region, resolve_launcher_region, singleton_region_list
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode

logger = logging.getLogger(__name__)

RL_EXECUTOR_STEP_RESOURCES = ResourceConfig.with_cpu(cpu=0.5, ram="4g", disk="30g")


ModelArtifact = ExecutorStep | InputName | str


@dataclasses.dataclass
class ModelConfig:
    name: str
    type: str
    artifact: ModelArtifact
    config_class_path: str
    pip_dependency_groups: list[str] | None = None

    @property
    def safe_name(self) -> str:
        return self.name.replace("/", "-").lower()


@dataclasses.dataclass
class RLStepConfig:
    """Runtime config for an RL executor step."""

    name: str
    experiment_config: "RLExperimentConfig"
    curriculum: CurriculumConfig
    model_path: str
    output_path: str


@dataclasses.dataclass
class RLExperimentConfig:
    """Shared configuration for RL experiments."""

    model_config: ModelConfig
    rl_loss: RLLossModule
    experiment_name_suffix: str

    # trainer params
    train_batch_size: int = 1024
    per_device_parallelism: int = 16
    num_train_steps: int = 500
    steps_per_eval: int = 100
    checkpointer_save_interval: int = 600
    delete_previous_temporary_checkpoint_after_save: bool = True
    checkpoint_debug: CheckpointDebugConfig = dataclasses.field(default_factory=CheckpointDebugConfig)

    # wandb
    project_name: str = "marin_post_training"
    tags: list[str] = dataclasses.field(default_factory=lambda: ["rl", "math"])

    # optimization
    learning_rate: float = 1e-7
    max_grad_norm: float = 1.00
    weight_decay: float = 0.0
    warmup: int = 0
    lr_schedule: str = "constant"

    # sampling / tokens
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    n_prompts: int = 64
    n_generations_per_prompt: int = 16

    # replay buffer
    replay_buffer_capacity: int = 4096
    replay_buffer_alpha: float = 3.0
    replay_buffer_max_samples: int = 1

    # execution
    debug_mode: bool = False
    inflight_weight_updates: bool = False
    max_rollout_step_delay: int = 0

    # weight transfer
    weight_transfer_sync_interval_steps: int = 1
    max_weight_transfer_wait_time: int = 0

    # inference context
    inference_tensor_parallel_size: int = 4
    inference_gpu_memory_utilization: float = 0.90
    inference_top_k: int = 4096  # Workaround for vllm-project/tpu-inference#1386
    inference_n: int = 8

    # run config (TPU slice info)
    train_tpu_type: str = "v5p-8"
    inference_tpu_type: str = "v5p-8"
    num_train_slices: int = 1
    num_rollout_workers: int = 1
    train_ram: str | None = None
    inference_ram: str | None = None
    zone: str | None = None


def launcher_region_for_rl_experiment(config: RLExperimentConfig) -> str:
    """Return the concrete region that should host the RL launcher and workers."""
    return resolve_launcher_region(config.train_tpu_type, config.inference_tpu_type)


def executor_step_resources_for_rl_experiment(config: RLExperimentConfig) -> ResourceConfig:
    """Return executor-step resources pinned to the RL launch region."""
    return dataclasses.replace(
        RL_EXECUTOR_STEP_RESOURCES,
        regions=singleton_region_list(launcher_region_for_rl_experiment(config)),
    )


def executor_main_config_for_rl_experiment(config: RLExperimentConfig) -> ExecutorMainConfig:
    """Return executor config with a region-local Marin prefix."""
    region = launcher_region_for_rl_experiment(config)
    return ExecutorMainConfig(prefix=marin_prefix_for_region(region))


def get_stop_tokens(model_type: str) -> list[str]:
    """Get model-specific stop tokens."""
    if model_type == "llama":
        return ["<|eot_id|>"]
    elif model_type == "qwen":
        return ["<|im_end|>"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _resolve_model_artifact_path(artifact: ModelArtifact) -> InputName | str:
    if isinstance(artifact, ExecutorStep):
        return output_path_of(artifact)

    return artifact


def _vllm_load_format_for_model_path(model_path: str) -> str:
    if urlparse(model_path).scheme in {"gs", "s3"}:
        # vLLM requires the streaming loader for object-store checkpoints.
        # `dummy` works for some local/HF cases where weights arrive later via
        # Arrow Flight, but remote object-store paths now fail validation.
        return "runai_streamer"

    return "dummy"


def config_class_path(config_class: type[HFCompatConfig]) -> str:
    return f"{config_class.__module__}.{config_class.__qualname__}"


def _resolve_config_class(config_class_path: str) -> type[HFCompatConfig]:
    module_name, _, qualname = config_class_path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"Invalid config class path: {config_class_path}")

    obj = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)

    if not isinstance(obj, type):
        raise TypeError(f"Resolved config class is not a type: {config_class_path}")

    return obj


def _build_rl_job_config(
    name: str,
    config: RLExperimentConfig,
    curriculum: CurriculumConfig,
    model_path: str,
    output_path: str,
    *,
    instance_id: str | None = None,
) -> RLJobConfig:
    model_config_class = _resolve_config_class(config.model_config.config_class_path)
    converter = HFCheckpointConverter(
        LevConfigClass=model_config_class,
        reference_checkpoint=model_path,
        tokenizer=model_path,
    )
    model_config_cls = model_config_class.from_hf_config(converter.default_hf_config)

    model_config = dataclasses.replace(
        model_config_cls,
        max_seq_len=config.max_input_tokens + config.max_output_tokens,
        tokenizer=model_path,
        attn_backend=AttentionBackend.SPLASH,
    )

    tags = [*config.tags, config.model_config.name.split("/")[-1]]
    checkpoints_path = join_path(output_path, "checkpoints")
    rollout_storage_path = join_path(output_path, "rollouts")

    trainer_config = TrainerConfig(
        tracker=WandbConfig(
            project=config.project_name,
            name=name,
            tags=tags,
        ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=config.train_batch_size,
        per_device_parallelism=config.per_device_parallelism,
        num_train_steps=config.num_train_steps,
        steps_per_eval=config.steps_per_eval,
        checkpointer=CheckpointerConfig(
            base_path=checkpoints_path,
            save_interval=datetime.timedelta(seconds=config.checkpointer_save_interval),
            delete_previous_temporary_checkpoint_after_save=config.delete_previous_temporary_checkpoint_after_save,
            debug=config.checkpoint_debug,
        ),
        mesh=MeshConfig(
            axes={"context": 1, "model": 1},
            shared_mapping={"mlp": "model", "heads": "model", "position": "context"},
        ),
    )

    opt_config = AdamConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup=config.warmup,
        lr_schedule=config.lr_schedule,
        max_grad_norm=config.max_grad_norm,
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=rollout_storage_path,
    )
    weight_transfer = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=config.weight_transfer_sync_interval_steps,
        max_weight_transfer_wait_time=config.max_weight_transfer_wait_time,
        coordinator_name=f"wt-coord-{instance_id}" if instance_id is not None else f"weight_transfer_coordinator_{name}",
    )

    curriculum_config = curriculum
    if instance_id is not None:
        curriculum_config = dataclasses.replace(curriculum, actor_name=f"curriculum-{instance_id}")

    # Create RLJobConfig using the unified interface
    job_config = RLJobConfig(
        model=model_config,
        vocab_size=converter.default_hf_config.vocab_size,
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=opt_config,
            rl_loss=config.rl_loss,
            replay_buffer=ReplayBufferConfig(
                capacity=config.replay_buffer_capacity,
                alpha=config.replay_buffer_alpha,
                max_samples=config.replay_buffer_max_samples,
                max_rollout_step_delay=config.max_rollout_step_delay,
            ),
        ),
        curriculum=curriculum_config,
        tokenizer=model_path,
        inference_type="vllm",
        inference_config=vLLMInferenceContextConfig(
            model_name=model_path,
            canonical_model_name=config.model_config.name,
            max_model_len=config.max_input_tokens + config.max_output_tokens,
            tensor_parallel_size=config.inference_tensor_parallel_size,
            gpu_memory_utilization=config.inference_gpu_memory_utilization,
            sampling_params=VLLMSamplingConfig(
                temperature=1.0,
                n=config.inference_n,
                max_tokens=config.max_output_tokens,
                stop=get_stop_tokens(config.model_config.type),
                include_stop_str_in_output=True,
                logprobs=1,
                top_k=config.inference_top_k,
            ),
            load_format=_vllm_load_format_for_model_path(model_path),
        ),
        initial_checkpoint=model_path,
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        run_id=name,
        log_freq=1,
        run_config=RunConfig(
            train_tpu_type=config.train_tpu_type,
            num_train_slices=config.num_train_slices,
            num_rollout_workers=config.num_rollout_workers,
            inference_tpu_type=config.inference_tpu_type,
            train_ram=config.train_ram,
            inference_ram=config.inference_ram,
            regions=singleton_region_list(launcher_region_for_rl_experiment(config)),
            zone=config.zone,
        ),
        inflight_weight_updates=config.inflight_weight_updates,
        instance_id=instance_id,
        rollout_tracker=RolloutTrackerConfig(
            project=config.project_name,
            name=f"{name}-rollout",
            tags=[*config.tags, "rollout", config.model_config.name.split("/")[-1]],
        ),
        pip_dependency_groups=(
            config.model_config.pip_dependency_groups if config.model_config.pip_dependency_groups else ["vllm", "math"]
        ),
    )

    return job_config


def _run_rl_experiment_step(config: RLStepConfig) -> PathMetadata:
    instance_id = f"{config.name}-{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    job_config = _build_rl_job_config(
        name=config.name,
        config=config.experiment_config,
        curriculum=config.curriculum,
        model_path=config.model_path,
        output_path=config.output_path,
        instance_id=instance_id,
    )
    logger.info("Launching RL executor step run %s with instance_id=%s", config.name, instance_id)
    RLJob(job_config).run(config.name)
    return PathMetadata(path=config.output_path)


def make_rl_step(name: str, config: RLExperimentConfig, curriculum: CurriculumConfig) -> ExecutorStep:
    model_path = _resolve_model_artifact_path(config.model_config.artifact)
    runtime_model_config = dataclasses.replace(config.model_config, artifact=model_path)
    runtime_config = dataclasses.replace(config, model_config=runtime_model_config)

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=remote(resources=executor_step_resources_for_rl_experiment(config))(_run_rl_experiment_step),
        config=RLStepConfig(
            name=name,
            experiment_config=runtime_config,
            curriculum=curriculum,
            model_path=model_path,
            output_path=this_output_path(),
        ),
    )
