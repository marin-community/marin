import os
from typing import List, Tuple

from experiments.defaults import default_train, default_tokenize
from experiments.llama import llama_8b
from experiments.defaults import SimpleTrainConfig
from marin.execution.executor import ExecutorStep
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
from marin.processing.tokenize import lm_data_config # Included as requested
from marin.resources import TpuPodConfig
from experiments.evals.task_configs import MMLU_TASKS


def run_midtraining_anneal(
    name_prefix: str,
    transformed_datasets: List[Tuple[str, ExecutorStep]],
    base_model_checkpoint_path: str,
    tokenizer_path: str,
) -> List[ExecutorStep]:
    """
    Sets up annealing training runs for individual and combined datasets.
    Each dataset is first tokenized, then used for training.
    """
    training_steps: List[ExecutorStep] = []

    # Common training parameters
    learning_rate = 1e-4
    min_lr_ratio = 0.1
    weight_decay = 0.05
    lr_schedule = "linear"
    train_batch_size = 1024
    num_anneal_training_tokens = 50_000_000_000  # 50B tokens
    resources = TpuPodConfig(tpu_type="v4-128", slice_count=2)
    steps_per_export = 10000
    use_default_validation = False
    llama_max_seq_len = 4096

    # Get Initial Checkpoint Step
    imputed_checkpoint_step = 0
    if "step-" in base_model_checkpoint_path:
        try:
            # e.g. "gs://bucket/path/to/checkpoint/step-12345/metadata.json" or "gs://bucket/path/to/checkpoint/step-12345"
            step_str = base_model_checkpoint_path.split("step-")[-1]
            imputed_checkpoint_step = int(step_str.split("/")[0])
        except ValueError:
            # Log a warning or handle as appropriate for the application
            print(f"Warning: Could not parse step from checkpoint path {base_model_checkpoint_path}. Defaulting to 0.")
            pass

    # --- Individual Dataset Annealing ---
    for dataset_name_suffix, raw_text_dataset_step in transformed_datasets:
        tokenized_step_name = os.path.join(name_prefix, f"tokenized_{dataset_name_suffix}")
        tokenized_dataset_step = default_tokenize(
            name=tokenized_step_name,
            dataset=raw_text_dataset_step,
            tokenizer=tokenizer_path,
            override_output_path=tokenized_step_name, # Good practice for predictable paths
        )

        num_anneal_steps_for_current_run = num_anneal_training_tokens / (train_batch_size * llama_max_seq_len)
        current_total_steps = imputed_checkpoint_step + num_anneal_steps_for_current_run

        current_train_config = SimpleTrainConfig(
            resources=resources,
            train_batch_size=train_batch_size,
            num_train_steps=int(current_total_steps),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            min_lr_ratio=min_lr_ratio,
            steps_per_export=steps_per_export,
            lr_schedule=lr_schedule,
            initialize_from_checkpoint_path=base_model_checkpoint_path,
            warmup=0,
        )

        train_step_name = os.path.join(name_prefix, f"anneal_{dataset_name_suffix}")
        train_step = default_train(
            name=train_step_name,
            tokenized=tokenized_dataset_step,
            model_config=llama_8b,
            train_config=current_train_config,
            use_default_validation=use_default_validation,
            eval_harness_tasks=MMLU_TASKS if use_default_validation else [],
            override_output_path=train_step_name, # Good practice for predictable paths
        )
        training_steps.append(train_step)

    # --- Combined Dataset Annealing ---
    if transformed_datasets: # Ensure there's something to combine
        tokenized_individual_dataset_steps: List[ExecutorStep] = []
        for dataset_name_suffix, raw_text_dataset_step in transformed_datasets:
            tokenized_combined_part_name = os.path.join(name_prefix, f"tokenized_for_combined_{dataset_name_suffix}")
            tokenized_step = default_tokenize(
                name=tokenized_combined_part_name,
                dataset=raw_text_dataset_step,
                tokenizer=tokenizer_path,
                override_output_path=tokenized_combined_part_name, # Good practice for predictable paths
            )
            tokenized_individual_dataset_steps.append(tokenized_step)

        train_dataset_formats = [
            TextLmDatasetFormat(data_path=step, weight=1.0)
            for step in tokenized_individual_dataset_steps
        ]
        
        combined_tokenized_config = LMMixtureDatasetConfig(
            tokenizer=tokenizer_path,
            train_datasets=train_dataset_formats,
            validation_datasets={},
        )

        num_anneal_steps_for_combined_run = num_anneal_training_tokens / (train_batch_size * llama_max_seq_len)
        combined_total_steps = imputed_checkpoint_step + num_anneal_steps_for_combined_run

        combined_train_config = SimpleTrainConfig(
            resources=resources,
            train_batch_size=train_batch_size,
            num_train_steps=int(combined_total_steps),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            min_lr_ratio=min_lr_ratio,
            steps_per_export=steps_per_export,
            lr_schedule=lr_schedule,
            initialize_from_checkpoint_path=base_model_checkpoint_path,
            warmup=0,
        )

        combined_train_step_name = os.path.join(name_prefix, "anneal_combined")
        combined_train_step = default_train(
            name=combined_train_step_name,
            tokenized=combined_tokenized_config,
            model_config=llama_8b,
            train_config=combined_train_config,
            use_default_validation=use_default_validation,
            eval_harness_tasks=MMLU_TASKS if use_default_validation else [],
            override_output_path=combined_train_step_name, # Good practice for predictable paths
        )
        training_steps.append(combined_train_step)

    return training_steps
