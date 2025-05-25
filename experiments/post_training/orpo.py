import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec
import ray
from datasets import load_dataset
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer

from marin.execution.executor import ExecutorStep, executor_main, this_output_path


@dataclass(frozen=True)
class ORPOTrainingConfig:
    """Configuration for the complete ORPO training pipeline."""

    repo_path: str
    prompt_length: int
    max_length: int
    dataset_name: str
    dataset_split: str
    cache_dir: str
    output_path: str

    # Training hyperparameters
    model_name: str = "marin-8b-instruct-orpo"
    total_batch_size: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 8e-6
    beta: float = 0.1
    save_steps: int = 1000
    warmup_steps: int = 0
    clip_grad: float = 1.0
    use_wandb: bool = True

    sharding_axis_dims: Sequence[int] = (1, -1, 1, 1)  # DP, FSDP, TP, SP

    save_total_limit: int = 0
    weight_distribution_log_steps: int = 100
    report_steps: int = 10
    log_steps: int = 5


@ray.remote
def orpo_training_pipeline(config: ORPOTrainingConfig):
    """Complete ORPO training pipeline in a single step."""
    # NOTE: why should EasyDeL always be first import?
    # - rewrite some jax functions.
    # - enable easydel fusions.
    # - auto set jax tpulib flags
    import easydel as ed  # type:ignore

    logging.info("Starting ORPO training pipeline")

    # Setup tokenizer
    processor = AutoTokenizer.from_pretrained(config.repo_path)
    processor.pad_token_id = processor.eos_token_id

    # Setup model

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        config.repo_path,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=config.sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=config.max_length,
            mask_max_position_embeddings=config.max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        cache_dir=config.cache_dir,
    )

    # Load dataset
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    # Setup training arguments
    arguments = ed.ORPOConfig(
        model_name=config.model_name,
        num_train_epochs=config.num_train_epochs,
        total_batch_size=config.total_batch_size,
        gradient_accumulation_steps=1,
        save_directory=config.output_path,
        do_eval=True,
        use_wandb=config.use_wandb,
        learning_rate=config.learning_rate,
        beta=config.beta,
        do_last_save=True,
        max_prompt_length=config.prompt_length,
        max_length=config.max_length,
        max_completion_length=config.max_length,
        max_training_steps=None,
        max_evaluation_steps=None,
        max_sequence_length=config.max_length,
        loss_config=ed.LossConfig(z_loss=0.0),
        track_memory=False,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_optimizer_state=False,
        per_epoch_training_steps=None,
        per_epoch_evaluation_steps=None,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        clip_grad=config.clip_grad,
        weight_distribution_log_steps=config.weight_distribution_log_steps,
        report_steps=config.report_steps,
        log_steps=config.log_steps,
        warmup_steps=config.warmup_steps,
        progress_bar_type="json",
    )

    # Perform training
    trainer = ed.ORPOTrainer(
        arguments=arguments,
        model=model,
        processing_class=processor,
        train_dataset=dataset,
    )

    trainer.train()

    # Save training results and metadata
    training_results = {
        "model_name": config.model_name,
        "repo_path": config.repo_path,
        "dataset_name": config.dataset_name,
        "num_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "beta": config.beta,
        "prompt_length": config.prompt_length,
        "max_length": config.max_length,
        "total_batch_size": config.total_batch_size,
        "status": "completed",
    }

    results_path = os.path.join(config.output_path, "training_results.json")
    with fsspec.open(results_path, "w") as f:
        json.dump(training_results, f, indent=2)

    logging.info(f"ORPO training pipeline completed, results saved to {config.output_path}")


if __name__ == "__main__":
    repo_path = "erfanzar/Marin-8B-Instruct-eformat"
    prompt_length = 2048
    max_length = prompt_length * 2

    orpo_step = ExecutorStep(
        name="orpo_training/complete",
        description="Complete ORPO training pipeline for Marin-8B model",
        fn=orpo_training_pipeline,
        config=ORPOTrainingConfig(
            repo_path=repo_path,
            prompt_length=prompt_length,
            max_length=max_length,
            dataset_name="orpo-explorers/OpenHermesPreferences-10k",
            dataset_split="train",
            cache_dir="/dev/shm/marin",
            output_path=this_output_path("orpo"),
            model_name="marin-8b-instruct-orpo",
            num_train_epochs=1,
            learning_rate=8e-6,
            beta=0.1,
            save_steps=1000,
            save_total_limit=1,
            warmup_steps=0,
            clip_grad=1.0,
            use_wandb=True,
        ),
    )

    executor_main(
        steps=[orpo_step],
        description="Complete ORPO training pipeline for Marin-8B model",
    )
