# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration dataclasses for RL training using draccus."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizerConfig:
    """Optimizer configuration parameters."""

    init_lr: float = 0.0
    end_lr: float = 3e-5
    lr: float = 3e-4
    lr_warmup_steps: int = 3000
    lr_decay_steps: int = 300000
    b1: float = 0.9
    b2: float = 0.95
    clip_gradient: float = 1.0
    weight_decay: float = 0.1
    bf16_momentum: bool = False
    multiply_by_parameter_scale: bool = False
    weight_decay_exclusions: tuple = field(default_factory=tuple)
    schedule: str = "cos"
    grad_accum_steps: int = 1


@dataclass
class GenerationConfig:
    """Inference generation configuration parameters."""

    max_output_length: int = 1025
    temperature: float = 1.0
    stop_tokens: list[list[int]] = field(default_factory=list)
    n_generations: int = 64


@dataclass
class AttentionKernelConfig:
    """Attention kernel configuration parameters."""

    kernel_type: str
    block_size: int | None = None
    page_size: int | None = None
    pages_per_compute_block: int | None = None
    inline_seq_dim: bool | None = None
    use_int8: bool | None = None


@dataclass
class LoggerConfigData:
    """Logger configuration parameters."""

    online: bool = True
    prefix: str | None = None
    prefix_to_id: bool = True
    experiment_id: str | None = None
    enable: bool | None = None
    config_to_log: dict[str, Any] | None = None


@dataclass
class CheckpointerConfigData:
    """Checkpointer configuration parameters."""

    save_optimizer_state: bool = False
    save_float_dtype: str = "bf16"


@dataclass
class ModelPathsConfig:
    """Model paths configuration."""

    params: str | None = None

    # RP: I'm not sure what this actually contains, it's a a path to something.
    config: str | None = None
    tokenizer: str = ""
    default_config_name: str | None = None
    remove_dict_prefix: str | None = None
    train_state: str | None = None


@dataclass
class TokenizerOverrideConfig:
    """Tokenizer override configuration parameters."""

    truncation_side: str = "right"
    padding_side: str = "right"
    pad_token: str = "<|reserved_special_token_0|>"


@dataclass
class ModelOverrideConfig:
    """Model configuration override parameters."""

    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: int = 128002
    max_sequence_length: int = 2048
    remat_block: str = "nothing_saveable"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    initializer_range: float = 0.02


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    model_paths: ModelPathsConfig
    inference_param_dtype: str = "bf16"
    inference_activation_dtype: str = "bf16"
    training_param_dtype: str = "fp32"
    training_activation_dtype: str = "fp32"
    model_config_override: ModelOverrideConfig = field(default_factory=ModelOverrideConfig)
    tokenizer_override: TokenizerOverrideConfig = field(default_factory=TokenizerOverrideConfig)
    train_attention_kernel_config: str = "splash:{}"
    prefill_attention_kernel_config: str = "splash:{}"
    generate_attention_kernel_config: str = "paged:{}"


@dataclass
class TrainingHyperparameters:
    """Training hyperparameter configuration."""

    num_train_steps: int
    max_input_length: int
    max_output_length: int
    train_bsize: int
    decode_bsize: int
    prefill_bsize: int
    reference_logprobs_bsize: int
    n_prompts_per_step: int
    optim_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    pad_token_id: int = 128002
    kl_coef: float = 0.0


@dataclass
class LoggingConfig:
    """Logging and evaluation configuration."""

    log_freq: int
    num_eval_examples: int
    save_model_freq: int
    wandb_project: str
    logger_config: LoggerConfigData = field(default_factory=LoggerConfigData)
    save_initial_checkpoint: bool = False
    log_initial_step: bool = True
    max_checkpoints: int | None = None


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    train_environments_path: str = "environments.json"
    test_environments_path: str = "environments.json"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    sharding: list[int]
    physical_axis_splitting: bool = False
    jax_distributed_initalize_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    output_dir: str
    model: ModelConfig
    hyperparameters: TrainingHyperparameters
    logging: LoggingConfig
    environment: EnvironmentConfig
    distributed: DistributedConfig
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    test_generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    checkpointer_config: CheckpointerConfigData = field(default_factory=CheckpointerConfigData)


@dataclass
class TrainWorkerConfig:
    """Training worker specific configuration."""

    rollout_queue_bucket: str
    rollout_queue_path: str = "rollout_queue"
    batch_timeout: float = 60.0
    max_idle_time: float = 300.0


@dataclass
class InferenceWorkerConfig:
    """Inference worker specific configuration."""

    environment_spec: str
    checkpoint_source_path: str
    rollout_output_path: str
    checkpoint_poll_interval: float = 30.0
    rollout_batch_size: int = 32
    n_generations: int = 64
    n_examples_per_batch: int = 16
    use_gcs: bool = False
    max_rollouts: int | None = None
    checkpoint_timeout: float = 300.0
