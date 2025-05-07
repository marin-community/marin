# SFT (Supervised Fine-Tuning) Quickstart Guide

## Overview
Guide for running supervised fine-tuning (SFT) experiments using Marin. This guide covers both single dataset SFT and SFT on a mixture of datasets. It assumes the Ray dashboard is running per the setup documentation.
The examples primarily reproduce OLMO SFT or utilize Llama-3.1 style configurations.

## Key Concepts

* **`SimpleSFTConfig`**: A dataclass (`experiments/simple_sft_config.py`) that centralizes most SFT training parameters (e.g., batch size, learning rate, model paths, TPU type).
* **`default_tokenize`**: A helper function (`experiments/defaults.py`) to create a standardized tokenization step for SFT datasets.
* **`default_sft`**: A helper function (`experiments/defaults.py`) to create a standardized training step. It can handle:
    * **Single Dataset SFT**: When `mixture_weights` is not provided.
    * **Mixture Dataset SFT**: When `mixture_weights` (a dictionary mapping dataset names to their sampling weights) is provided.
* **Experiment Files**: Python scripts in the `experiments/` directory (e.g., `exp606_sft.py` for single dataset, `exp808_sft_mixture.py` for mixture SFT) define the specific datasets, tokenization, and training configurations.

## Key Steps

### 1. Basic Commands
To run an SFT experiment (e.g., marin default SFT `exp808_sft_mixture.py`):
```bash
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/exp808_sft_mixture.py
```
(Replace experiments/exp808_sft_mixture.py with the actual experiment file you are running).

### 2. Configure Dataset(s)
For a Single Dataset, you'll typically use get_instruction_dataset to load your dataset.
``` python
# In your experiment script (e.g., experiments/exp606_sft.py)
from experiments.instruction_datasets import get_instruction_dataset

dataset_hf_name = "allenai/tulu-v2-sft-mixture" # Or your dataset's Hugging Face name
instruction_dataset = get_instruction_dataset(dataset_hf_name, splits=["train"])
# This instruction_dataset (or its path) will be passed to default_tokenize
```
For a Mixture of Datasets:
Define your datasets and their corresponding Hugging Face names. You will create a tokenization step for each.
``` python
# In your experiment script (e.g., experiments/exp808_sft_mixture.py)
DATASETS = {
    "acecode_89k": "TIGER-Lab/AceCode-89K",
    "smoltalk": "HuggingFaceTB/smoltalk",
    # ... other datasets
}

# Tokenization steps will be created for each dataset in DATASETS.
# See exp808_sft_mixture.py for the full structure including creating
# tokenized_datasets and mixture_weights.
```
Please look at experiments/instruction_datasets.py for examples on how to process SFT datasets.

### 3. Tokenization Configuration (using default_tokenize)
The default_tokenize function standardizes the tokenization process. Key parameters when calling it in your experiment script:
```python
# In your experiment script
from experiments.defaults import default_tokenize
from experiments.your_tokenizer_module import your_tokenizer # e.g., marin_tokenizer, llama3_tokenizer
from levanter.data.text import ChatLmDatasetFormat # For chat-formatted data

# For a single dataset tokenization step:
tokenization_step = default_tokenize(
    name="short_name_for_your_tokenized_data", # e.g., "tulu_v2_tokenized". Output path prefix will be "tokenized/"
    dataset=instruction_dataset / "**/*.jsonl.gz", # Or path pattern to your raw data
    tokenizer=your_tokenizer, # Must match the base model's tokenizer
    format=ChatLmDatasetFormat(
        input_field="user",         # Field name for user/input messages in your JSONL
        output_field="assistant"    # Field name for assistant/output messages
    )
    # cache_path is automatically managed by this_output_path() within default_tokenize
    # seq_len is typically managed by the model configuration (SimpleSFTConfig.max_seq_len)
)

# For mixture SFT, you'll create a similar step for each dataset.
# See experiments/exp808_sft_mixture.py create_tokenization_step function.
```

### 4. Training Configuration (using SimpleSFTConfig and default_sft)
Training is configured using SimpleSFTConfig and then passed to default_sft.

#### a. Define SimpleSFTConfig:
```
# In your experiment script
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.your_tokenizer_module import your_tokenizer_path # e.g., "meta-llama/Llama-3.1-8B-Instruct"

sft_run_config = SimpleSFTConfig(
    # Core training
    train_batch_size=128,
    num_train_steps=10000, # Adjust based on your dataset size and epochs
    learning_rate=5e-6,
    # Hardware
    tpu_type="v4-128", # e.g., "v4-8", "v4-64", "v4-128" based on needs
    # Model & Tokenizer
    tokenizer=your_tokenizer_path, # String identifier for tokenizer, must match base model
    model_name_or_path="meta-llama/Llama-3.1-8B", # Path to base model checkpoint (HF or GCS)
    max_seq_len=4096, # Maximum sequence length
    # Roles for chat data (must match fields used in ChatLmDatasetFormat during tokenization)
    input_role="user",
    output_role="assistant",
    # Checkpointing
    steps_per_checkpoint=1000,
    steps_per_hf_export=500,
    # For mixture SFT, these might be relevant:
    # mixture_block_size=2048,
    # stop_strategy="restart",
    seed=0
    # ... other parameters. See experiments/simple_sft_config.py for all options.
)
```
#### b. Define default_sft training step:
```
# In your experiment script
from experiments.defaults import default_sft
from experiments.your_model_module import your_base_model_config # e.g., llama_8b from experiments.llama

# For single dataset SFT:
# tokenization_step is the output from default_tokenize defined earlier
training_step = default_sft(
    name="your_experiment_name", # e.g., "llama3.1_tulu_sft". Output path prefix "checkpoints/"
    tokenized=tokenization_step, # The ExecutorStep from default_tokenize
    model_config=your_base_model_config, # Levanter model config (e.g., LlamaConfig)
    sft_config=sft_run_config # The SimpleSFTConfig instance from above
    # mixture_weights is None for single dataset SFT
)

# For mixture SFT (see experiments/exp808_sft_mixture.py for full context):
# tokenized_datasets_config = lm_mixture_data_config(tokenized_executor_steps_dict, mixture_weights_dict, ...)
# mixture_weights_dict = {"dataset1_short_name": weight1, "dataset2_short_name": weight2, ...}
#
# training_step = default_sft(
#     name="your_mixture_experiment_name",
#     tokenized=tokenized_datasets_config, # This is an LMMixtureDatasetConfig or similar
#     model_config=your_base_model_config,
#     sft_config=sft_run_config,
#     mixture_weights=mixture_weights_dict # This enables mixture mode
# )
```
The name parameter for default_sft (e.g., "your_experiment_name") will be part of the checkpoint path, like checkpoints/your_experiment_name_seed0.

### 5. Running the Experiment Script
Your experiment script (e.g., my_sft_experiment.py) should typically end with:
```
# In your experiment script (e.g., my_sft_experiment.py)
from marin.execution.executor import executor_main

# Assuming tokenization_step and training_step are defined as above for single SFT
# Or for mixture: tokenization_steps_list and training_step

# For single SFT:
# executor_main(steps=[tokenization_step, training_step])

# For mixture SFT (example from exp808_sft_mixture.py):
# tokenized_datasets_values = list(tokenized_datasets.values()) # list of tokenization ExecutorSteps
# executor_main(steps=[*tokenized_datasets_values, training_step])
```

Force specific steps:
```
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/my_sft_experiment.py --force_run '["name_of_step_to_force"]'
```
(e.g., --force_run '["tokenized/short_name_for_your_tokenized_data"]' or --force_run '["checkpoints/your_experiment_name_seed0"]')

## Storage Paths
All experiment names should follow the convention below, which is specifying the top level directory undernead marin-us-central2. For example, the name for the tokenize_step should start with the prefix tokenized/
* Base models: gs://levanter-checkpoints/
* Your experiments: gs://marin-us-central2/checkpoints/
* Tokenized cache: gs://marin-us-central2/tokenized/sft_cache/

Common TPU Configurations
* Small experiments: "v4-64"
* Production runs: "v4-128"
* Batch sizes should be scaled accordingly, I get 57 MFU with batch size 1024 on v4-128

## Tips
* Match tokenizer to base model
* Monitor via Ray dashboard and W&B
* Use --force_run during development to re-run jobs or just delete executor state
* Adjust batch size based on TPU size

## Troubleshooting

* Verify TPU availability before large runs
* Check tokenizer/model compatibility
* Ensure GCS paths are accessible
* Monitor memory usage with large batch sizes