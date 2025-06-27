"""
RL training.
"""

import dataclasses
from typing import Any

from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, executor_main, versioned
from marin.resources import TpuPodConfig


@dataclasses.dataclass
class LoadModelConfig:
    params: str
    tokenizer: str
    config: str


@dataclasses.dataclass
class OptimizerConfig:
    init_lr: float = 5e-7
    end_lr: float = 5e-7
    lr: float = 5e-7
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 2048
    b1: float = 0.9
    b2: float = 0.95
    clip_gradient: float = 1.0
    weight_decay: float = 0.0
    bf16_momentum: bool = False
    multiply_by_parameter_scale: bool = False
    weight_decay_exclusions: list[str] = dataclasses.field(default_factory=list)
    schedule: str = "cos"
    grad_accum_steps: int = 16


@dataclasses.dataclass
class GenerationConfig:
    max_output_length: int
    temperature: float
    stop_tokens: list[list[int]]
    n_generations: int


@dataclasses.dataclass
class LoggerConfig:
    online: bool
    prefix: str
    prefix_to_id: bool


@dataclasses.dataclass
class CheckpointerConfig:
    save_optimizer_state: bool
    save_float_dtype: str


@dataclasses.dataclass
class AttentionKernelConfig:
    block_size: int = 256


@dataclasses.dataclass
class PagedAttentionKernelConfig:
    page_size: int = 256
    pages_per_compute_block: int = 1
    inline_seq_dim: bool = True
    use_int8: bool = False


@dataclasses.dataclass
class GrpoTrainConfig:
    """Configuration for GRPO training that includes TPU resource specification."""

    resources: TpuPodConfig

    # Core configs
    load_model: LoadModelConfig
    output_dir: str = THIS_OUTPUT_PATH  # Let the executor manage the output path
    sharding: str = "1,4,1,-1"
    num_train_steps: int = 2048

    # Batch sizes
    train_bsize: int = 64
    decode_bsize: int = 1024
    prefill_bsize: int = 16
    reference_logprobs_bsize: int = 256
    n_prompts_per_step: int = 16

    # Lengths and IDs
    max_input_length: int = 256
    max_output_length: int = 1025
    pad_token_id: int = 128002

    # Logging and Saving
    log_freq: int = 8
    num_eval_examples: int = 1024
    save_model_freq: int = 0
    wandb_project: str = "math_rloo_math_test_experiments"

    # Data types
    inference_param_dtype: str = "bf16"
    inference_activation_dtype: str = "bf16"
    training_param_dtype: str = "fp32"
    training_activation_dtype: str = "bf16"

    # GRPO specific
    kl_coef: float = 1e-3

    # Nested configurations
    optim_config: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    logger_config: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)
    checkpointer_config: CheckpointerConfig = dataclasses.field(default_factory=CheckpointerConfig)
    generation_config: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    test_generation_config: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    model_config_override: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Attention kernels
    train_attention_kernel_config: AttentionKernelConfig = dataclasses.field(default_factory=AttentionKernelConfig)
    prefill_attention_kernel_config: AttentionKernelConfig = dataclasses.field(default_factory=AttentionKernelConfig)
    generate_attention_kernel_config: PagedAttentionKernelConfig = dataclasses.field(
        default_factory=PagedAttentionKernelConfig
    )


def run_grpo_training_on_tpu(config: GrpoTrainConfig):
    from marin.post_training.train import main as grpo_training_entrypoint

    # Extract the training config (everything except resources)
    training_config = dataclasses.replace(config)
    # Remove the resources field since it's only for pod management
    training_config = dataclasses.replace(training_config, resources=None)

    # Call the training entrypoint
    grpo_training_entrypoint(training_config)


def create_grpo_math_experiment(
    name: str,
    tpu_type: str = "v5p-8",
    tpu_zone: str = "us-east5-a",
    train_bsize: int = 64,
    kl_coef: float = 1e-3,
    learning_rate: float = 5e-7,
    num_train_steps: int = 2048,
    **kwargs,
) -> ExecutorStep:
    """Helper function to create GRPO experiment configurations."""

    # Default stop tokens for math problems
    default_stop_tokens = [
        [524, 9399],
        [694, 9399],
        [4005, 9399],
        [6199, 9399],
        [8217, 9399],
        [9169, 9399],
        [12817, 9399],
        [19203, 9399],
        [20264, 9399],
        [22246, 9399],
        [27147, 9399],
        [128001],
    ]

    # Create TPU resource configuration
    resources = TpuPodConfig(
        tpu_type=versioned(tpu_type),
        zone=versioned(tpu_zone),
    )

    config = GrpoTrainConfig(
        resources=resources,
        train_bsize=versioned(train_bsize),
        kl_coef=versioned(kl_coef),
        num_train_steps=num_train_steps,
        load_model=LoadModelConfig(
            params="gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/params.msgpack",
            tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
            config="gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/config.json",
        ),
        logger_config=LoggerConfig(
            online=True,
            prefix=name,
            prefix_to_id=True,
        ),
        checkpointer_config=CheckpointerConfig(
            save_optimizer_state=False,
            save_float_dtype="bf16",
        ),
        optim_config=OptimizerConfig(
            init_lr=learning_rate,
            end_lr=learning_rate,
            lr=learning_rate,
            lr_decay_steps=num_train_steps,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["b1", "b2", "clip_gradient", "weight_decay", "schedule", "grad_accum_steps"]
            },
        ),
        generation_config=GenerationConfig(
            max_output_length=1025,
            temperature=1.0,
            stop_tokens=default_stop_tokens,
            n_generations=64,
        ),
        test_generation_config=GenerationConfig(
            max_output_length=1025,
            temperature=0.0,
            stop_tokens=default_stop_tokens,
            n_generations=1,
        ),
        model_config_override={
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "pad_token_id": 128002,
            "max_sequence_length": 2048,
            "remat_block": "nothing_saveable",
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        },
        train_attention_kernel_config=AttentionKernelConfig(block_size=256),
        prefill_attention_kernel_config=AttentionKernelConfig(block_size=256),
        generate_attention_kernel_config=PagedAttentionKernelConfig(
            page_size=256,
            pages_per_compute_block=1,
            inline_seq_dim=True,
            use_int8=False,
        ),
    )

    return ExecutorStep(
        name=f"grpo_math/{name}",
        fn=run_grpo_training_on_tpu,
        config=config,
        description=f"GRPO math training experiment: {name}",
        pip_dependency_groups=["post_training"],
    )


def main():
    experiments = [
        # Baseline experiment on v5p-8
        create_grpo_math_experiment(
            name="llama3_8b_math_baseline",
            tpu_type="v5p-8",
            tpu_zone="us-central1-a",
            train_bsize=64,
            kl_coef=1e-3,
            learning_rate=5e-7,
            num_train_steps=2048,
        ),
    ]

    executor_main(
        steps=experiments,
        description="GRPO math training experiments on Llama 3.1 8B",
    )


if __name__ == "__main__":
    main()
