import json
import logging
import os
import pathlib
from dataclasses import dataclass

import fsspec
import ray
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer

from marin.execution.executor import ExecutorStep, executor_main, this_output_path


@dataclass(frozen=True)
class VSurgeEvalConfig:
    """Configuration for the complete vSurge evaluation pipeline."""

    repo_path: str
    max_prefill_length: int
    max_decode_length: int
    max_concurrent_decodes: int
    cache_dir: str
    output_path: str

    # Evaluation parameters
    tasks: list[str]
    num_fewshot: int = 3
    top_p: float = 0.95
    temperature: float = 0.1
    seed: int = 48
    vsurge_name: str = "marin"
    verbose: bool = True

    # Model sharding configuration
    sharding_axis_dims: tuple = (1, 1, -1, 1)


@ray.remote
def vsurge_evaluation_pipeline(config: VSurgeEvalConfig):
    """Complete vSurge evaluation pipeline in a single step."""
    import easydel as ed  # type:ignore
    from lm_eval import simple_evaluate

    logging.info("Starting vSurge evaluation pipeline")

    # Calculate total sequence length
    max_length = config.max_prefill_length + config.max_decode_length

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
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        cache_dir=config.cache_dir,
    )

    # Setup vSurge driver
    surge = ed.vSurge.create_vdriver(
        model=model,
        processor=processor,
        prefill_lengths=[
            config.max_prefill_length // 16,
            config.max_prefill_length // 8,
            config.max_prefill_length // 4,
            config.max_prefill_length // 2,
            config.max_prefill_length,
        ],
        max_concurrent_decodes=config.max_concurrent_decodes,
        max_prefill_length=config.max_prefill_length,
        max_length=max_length,
        vsurge_name=config.vsurge_name,
        verbose=config.verbose,
        seed=config.seed,
    )

    eval_runner = ed.VSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=max_length,
        max_new_tokens=config.max_decode_length,
        top_p=config.top_p,
        temperature=config.temperature,
    )

    try:
        logging.info(f"Starting evaluation on tasks: {config.tasks}")
        print(f"Starting evaluation on tasks: {config.tasks}")

        # Run evaluation
        results = simple_evaluate(
            model=eval_runner,
            tasks=config.tasks,
            num_fewshot=config.num_fewshot,
            batch_size=config.max_concurrent_decodes,
            device="cpu",
        )

        # Save evaluation results
        pathlib.Path(config.output_path).mkdir(parents=True, exist_ok=True)
        results_path = os.path.join(config.output_path, "evaluation_results.json")
        with fsspec.open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save evaluation metadata
        eval_metadata = {
            "model_path": config.repo_path,
            "tasks": config.tasks,
            "num_fewshot": config.num_fewshot,
            "max_prefill_length": config.max_prefill_length,
            "max_decode_length": config.max_decode_length,
            "max_concurrent_decodes": config.max_concurrent_decodes,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "seed": config.seed,
            "status": "completed",
        }

        metadata_path = os.path.join(config.output_path, "evaluation_metadata.json")
        with fsspec.open(metadata_path, "w") as f:
            json.dump(eval_metadata, f, indent=2)

        logging.info(f"Evaluation results saved to {results_path}")
        print(f"Evaluation results saved to {results_path}")

        print("Summary of results:")
        for task, metrics in results["results"].items():
            print(f"{task}: {metrics}")

        logging.info("vSurge evaluation pipeline completed successfully")

    except Exception as e:
        logging.error(f"Evaluation failed with error: {e}")
        raise
    finally:
        try:
            eval_runner.stop()
            logging.info("Model resources cleaned up")
        except Exception as e:
            logging.warning(f"Error during model cleanup: {e}")


if __name__ == "__main__":
    vsurge_eval_step = ExecutorStep(
        name="vsurge_evaluation/complete",
        description="Complete vSurge evaluation pipeline for Marin-8B model",
        fn=vsurge_evaluation_pipeline,
        config=VSurgeEvalConfig(
            repo_path="erfanzar/Marin-8B-Instruct-eformat",
            max_prefill_length=4096,
            max_decode_length=1024,
            max_concurrent_decodes=44,
            cache_dir="/dev/shm/marin",
            output_path=this_output_path("evaluation/vsurge"),
            tasks=["gsm8k"],
            num_fewshot=3,
            top_p=0.95,
            temperature=0.1,
            seed=48,
            vsurge_name="marin",
            verbose=True,
            sharding_axis_dims=(1, 1, -1, 1),
        ),
    )

    executor_main(
        steps=[vsurge_eval_step],
        description="Complete vSurge evaluation pipeline for Marin-8B model",
    )
