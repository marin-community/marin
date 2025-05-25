import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec
import ray
from datasets import load_dataset
from jax import numpy as jnp
from transformers import AutoTokenizer

from marin.execution.executor import ExecutorStep, executor_main, this_output_path


@dataclass(frozen=True)
class GRPOTrainingConfig:
    """Configuration for the complete GRPO training pipeline."""

    repo_path: str
    prompt_length: int
    max_completion_length: int
    dataset_name: str
    dataset_split: str
    cache_dir: str
    output_path: str

    # Training hyperparameters
    model_name: str = "marin-8b-grpo"
    total_batch_size: int = 8
    num_return_sequences: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 1e-6
    learning_rate_end: float = 6e-7
    beta: float = 0.04
    save_steps: int = 1000
    warmup_steps: int = 0
    clip_grad: float = 1.0
    use_wandb: bool = True

    save_total_limit: int = 0
    sharding_axis_dims: Sequence[int] = (1, -1, 1, 1)  # DP, FSDP, TP, SP
    # Generation parameters
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.7

    # Logging parameters
    weight_distribution_log_steps: int = 100
    report_steps: int = 10
    log_steps: int = 5


@ray.remote
def grpo_training_pipeline(config: GRPOTrainingConfig):
    """Complete GRPO training pipeline in a single step."""
    # NOTE: EasyDeL should be imported first to enable optimizations
    import easydel as ed  # type:ignore
    import jax
    from math_verify import LatexExtractionConfig, parse, verify  # type:ignore

    logging.info("Starting GRPO training pipeline")

    # Setup tokenizer
    processor = AutoTokenizer.from_pretrained(config.repo_path)
    processor.padding_side = "left"
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    # Calculate sequence lengths
    max_sequence_length = config.max_completion_length + config.prompt_length

    # Setup model
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        config.repo_path,
        auto_shard_model=True,
        sharding_axis_dims=config.sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=jnp.bfloat16,
            attn_softmax_dtype=jnp.bfloat16,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
        cache_dir=config.cache_dir,
    )

    # Setup GRPO configuration
    grpo_config = ed.GRPOConfig(
        model_name=config.model_name,
        total_batch_size=config.total_batch_size,
        max_prompt_length=config.prompt_length,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        learning_rate_end=config.learning_rate_end,
        save_directory=config.output_path,
        log_steps=config.log_steps,
        report_steps=config.report_steps,
        progress_bar_type="json",
        num_train_epochs=config.num_train_epochs,
        save_total_limit=config.save_total_limit,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        do_last_save=True,
        track_memory=False,
        save_steps=config.save_steps,
        save_optimizer_state=False,
        per_epoch_training_steps=None,
        per_epoch_evaluation_steps=None,
        use_wandb=config.use_wandb,
        clip_grad=config.clip_grad,
        weight_distribution_log_steps=config.weight_distribution_log_steps,
        warmup_steps=config.warmup_steps,
        beta=config.beta,
    )

    # Setup vInference
    vinference = ed.vInference(
        model=model,
        processor_class=processor,
        generation_config=ed.vInferenceConfig(
            bos_token_id=processor.bos_token_id,
            eos_token_id=processor.eos_token_id,
            pad_token_id=processor.pad_token_id,
            max_new_tokens=config.max_completion_length,
            streaming_chunks=64,
            sampling_params=ed.SamplingParams(
                max_tokens=config.max_completion_length,
                top_k=config.top_k,
                top_p=config.top_p,
                temperature=config.temperature,
            ),
            num_return_sequences=config.num_return_sequences,
        ),
    )

    # Precompile vInference
    vinference.precompile(
        ed.vInferencePreCompileConfig(
            batch_size=config.total_batch_size,
            prefill_length=config.prompt_length,
        )
    )

    # Define reward functions
    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        rewards_list = [1.0 if match else 0.0 for match in matches]
        return rewards_list

    def accuracy_reward(prompts, completions, batch, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        solutions = processor.batch_decode(batch["solution_ids"]) * config.num_return_sequences
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions):  # noqa
            gold_parsed = parse(
                solution,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            answer_parsed = parse(
                content,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        return rewards

    # Define system prompt
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "  # noqa
        "first thinks about the think process in the mind and then provides the user with the answer. The think "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> think process here </think><answer> answer here </answer>"
    )

    # Load and prepare datasets
    train_dataset, test_dataset = load_dataset(
        config.dataset_name,
        split=["train[:100%]", "test[:100%]"],
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation, remove_columns=["messages"])
    test_dataset = test_dataset.map(make_conversation, remove_columns=["messages"])

    # Define tokenization function
    def data_tokenize_fn(batch, tokenizer, tools):
        ids = tokenizer(
            batch["prompt"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",
            max_length=config.prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        ans = tokenizer(
            batch["solution"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",
            max_length=config.prompt_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids.update({"solution_ids": ans["input_ids"]})
        return ids

    # Setup trainer
    trainer = ed.GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        processing_class=processor,
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        arguments=grpo_config,
        vinference=vinference,
        data_tokenize_fn=data_tokenize_fn,
    )

    # Perform training
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
        "max_completion_length": config.max_completion_length,
        "total_batch_size": config.total_batch_size,
        "num_return_sequences": config.num_return_sequences,
        "status": "completed",
    }

    results_path = os.path.join(config.output_path, "training_results.json")
    with fsspec.open(results_path, "w") as f:
        json.dump(training_results, f, indent=2)

    logging.info(f"GRPO training pipeline completed, results saved to {config.output_path}")


if __name__ == "__main__":
    repo_path = "erfanzar/Marin-8B-DPO-stage2"
    prompt_length = 1024
    max_completion_length = 2048

    grpo_step = ExecutorStep(
        name="grpo_training/complete",
        description="Complete GRPO training pipeline for Marin-8B model",
        fn=grpo_training_pipeline,
        config=GRPOTrainingConfig(
            repo_path=repo_path,
            prompt_length=prompt_length,
            max_completion_length=max_completion_length,
            dataset_name="AI-MO/NuminaMath-TIR",
            dataset_split="train",
            cache_dir="/dev/shm/marin",
            output_path=this_output_path("grpo"),
            model_name="marin-8b-grpo",
            total_batch_size=8,
            num_return_sequences=4,
            num_train_epochs=3,
            learning_rate=1e-6,
            learning_rate_end=6e-7,
            beta=0.04,
            save_steps=1000,
            warmup_steps=0,
            clip_grad=1.0,
            use_wandb=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        ),
    )

    executor_main(
        steps=[grpo_step],
        description="Complete GRPO training pipeline for Marin-8B model",
    )
