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

"""Configuration dataclasses for RL training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizerConfig:
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
    max_output_length: int = 1025
    temperature: float = 1.0
    stop_tokens: list[list[int]] = field(default_factory=list)
    n_generations: int = 64


@dataclass
class AttentionKernelConfig:
    kernel_type: str
    block_size: int | None = None
    page_size: int | None = None
    pages_per_compute_block: int | None = None
    inline_seq_dim: bool | None = None
    use_int8: bool | None = None


@dataclass
class CheckpointerConfigData:
    save_optimizer_state: bool = False
    save_float_dtype: str = "bf16"
    save_model_freq: int = 1


@dataclass
class ModelPathsConfig:
    params: str | None = None

    # RP: I'm not sure what this actually contains, it's a a path to something.
    config: str | None = None
    tokenizer: str = ""
    default_config_name: str | None = None
    remove_dict_prefix: str | None = None
    train_state: str | None = None


@dataclass
class TokenizerOverrideConfig:
    truncation_side: str = "right"
    padding_side: str = "right"
    pad_token: str = "<|reserved_special_token_0|>"


@dataclass
class ModelOverrideConfig:
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: int = 128002
    max_sequence_length: int = 2048
    remat_block: str = "nothing_saveable"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    initializer_range: float = 0.02


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
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
    log_freq: int
    num_eval_examples: int
    wandb_project: str
    save_initial_checkpoint: bool = False
    log_initial_step: bool = True
    max_checkpoints: int | None = None
    num_groups_to_log: int = 4

    online: bool = True
    prefix: str | None = None
    prefix_to_id: bool = True
    experiment_id: str | None = None
    enable: bool | None = None
    config_to_log: dict[str, Any] | None = None


@dataclass
class EnvironmentConfig:
    train_environments_path: str = "environments.json"
    test_environments_path: str = "environments.json"


@dataclass
class DistributedConfig:
    train_sharding: list[int]
    inference_sharding: list[int]
    physical_axis_splitting: bool = False
    jax_distributed_initialize_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    output_dir: str
    model: ModelConfig
    hyperparameters: TrainingHyperparameters
    logging: LoggingConfig
    distributed: DistributedConfig
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    test_generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    checkpoint: CheckpointerConfigData = field(default_factory=CheckpointerConfigData)
