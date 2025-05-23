# How to Run a Supervised Fine-Tuning (SFT) Experiment

Supervised Fine-Tuning (SFT) is a process used to adapt a pre-trained large language model (LLM) to specific tasks or datasets. By training the model on a labeled dataset, SFT helps the model learn to generate responses that are more aligned with the desired behavior for those tasks.

This tutorial explains how to run SFT experiments using the Marin framework. We will cover how to run SFT with both single datasets and mixture datasets.

The examples in this tutorial are based on the following scripts:
*   `exp606_sft.py` (for single dataset SFT)
*   `experiments/exp808_sft_mixture.py` (for mixture dataset SFT)

## Prerequisites

Before running SFT experiments, ensure you have the following:

*   **A pretrained model:** The examples in this tutorial use a Llama-3.1 8B model. You can replace this with any other compatible pretrained model.
*   **An instruction dataset or a mixture of instruction datasets:** This dataset will be used to fine-tune the model. It should be in a format that is compatible with the Marin framework.
*   **Familiarity with Marin concepts:** You should have a basic understanding of tokenization and model configuration within the Marin framework. Refer to the relevant documentation if needed.

## Overview

Running an SFT experiment in Marin involves the following main steps:

1.  **Preparing your dataset(s):** This involves fetching your instruction dataset(s) and then tokenizing them so they can be consumed by the model.
2.  **Configuring the SFT parameters:** You'll need to set up a `SimpleSFTConfig` object. This configuration includes specifying the pretrained model, batch size, learning rate, training duration, and other hyperparameters.
3.  **Running the SFT training job:** The `default_sft` function is used to define the SFT job, and `executor_main` is used to launch the training process.
4.  **Evaluating the fine-tuned model:** After training, you'll typically want to evaluate your model's performance. While this tutorial focuses on the training aspect, evaluation is a crucial subsequent step.

## Steps

### 1. Configure a Single Dataset for SFT

This section describes how to configure a single instruction dataset for SFT, as demonstrated in `exp606_sft.py`.

**Fetching the Dataset**

The Marin framework provides the `get_instruction_dataset` function to easily fetch instruction datasets from various sources. You need to specify the dataset name. For example, to fetch the "allenai/tulu-3-sft-mixture" dataset:

```python
from marin.pretrain.data import get_instruction_dataset

dataset = get_instruction_dataset(dataset_name="allenai/tulu-3-sft-mixture")
```

**Tokenizing the Dataset**

Once the dataset is fetched, it needs to be tokenized. The `default_tokenize` function handles this process. You need to provide the dataset, specify the tokenizer to use (e.g., `llama3_instruct_tokenizer`), and the chat format (e.g., `llama3_instruct_chat_format`).

```python
from marin.pretrain.data import default_tokenize
from marin.models.llama3.tokenizer import llama3_instruct_tokenizer
from marin.models.llama3.format import llama3_instruct_chat_format

# These are typically pre-configured objects/variables
TOKENIZER = llama3_instruct_tokenizer
FORMAT = llama3_instruct_chat_format

tokenized_single_dataset_step = default_tokenize( # Renamed for clarity in later steps
    dataset=dataset,
    tokenizer=TOKENIZER, # Pass the variable directly
    format=FORMAT,       # Pass the variable directly
)
```
This approach is exemplified in `exp606_sft.py`. Later sections will build upon this tokenized dataset.

### 2. Configure a Mixture of Datasets for SFT (Optional)

Using a mixture of datasets can be beneficial when you want to fine-tune your model on diverse instructions or to prevent catastrophic forgetting of capabilities learned from different sources. `experiments/exp808_sft_mixture.py` provides an example of this.

**Defining Datasets and Tokenization Steps**

First, define a dictionary where keys are dataset names and values are their configurations. A helper function, like `create_tokenization_step` in `exp808_sft_mixture.py`, can simplify creating tokenization steps for each dataset. This function typically takes the dataset name, tokenizer, and format as input.

```python
from marin.pretrain.data import get_instruction_dataset, default_tokenize
from marin.models.llama3.tokenizer import llama3_instruct_tokenizer # Ensure these are imported
from marin.models.llama3.format import llama3_instruct_chat_format  # Ensure these are imported
from marin.core import Step

# (Tokenizer and format are assumed to be defined, e.g., TOKENIZER and FORMAT from previous step)

def create_tokenization_step(dataset_name: str, tokenizer, format) -> Step:
    """Helper to create a tokenization step for a given dataset."""
    return default_tokenize(
        dataset=get_instruction_dataset(dataset_name=dataset_name),
        tokenizer=tokenizer,
        format=format,
    )

# These steps will be passed to executor_main later
tokenized_tulu_step = create_tokenization_step(
    dataset_name="allenai/tulu-3-sft-mixture",
    tokenizer=TOKENIZER,
    format=FORMAT,
)
tokenized_orca_step = create_tokenization_step(
    dataset_name="Open-Orca/OpenOrca",
    tokenizer=TOKENIZER,
    format=FORMAT,
)

DATASETS_FOR_MIXTURE = { # Renamed for clarity
    "tulu": tokenized_tulu_step,
    "orca": tokenized_orca_step,
}
```

**Defining Mixture Weights**

Next, assign weights to each dataset in the mixture. These weights determine the proportion of examples drawn from each dataset during training.

```python
mixture_weights = {
    "tulu": 0.5,
    "orca": 0.5,
}
```

**Combining Datasets with `lm_mixture_data_config`**

Finally, use `lm_mixture_data_config` to combine the tokenized datasets and their corresponding weights into a single configuration for the SFT job.

```python
from marin.pretrain.data_config import lm_mixture_data_config

mixture_data_cfg = lm_mixture_data_config(
    datasets=DATASETS_FOR_MIXTURE, # Uses the steps directly
    mixture_weights=mixture_weights,
    max_seq_len=4096, # Example sequence length
)
```
This `mixture_data_cfg` can then be used in the SFT configuration, which will be discussed in the next section. This approach is exemplified in `experiments/exp808_sft_mixture.py`.

### 3. Set Up SFT Configuration

The `SimpleSFTConfig` class is used to define the parameters for your SFT experiment. It encapsulates various settings related to the model, training process, and resources.

Key parameters include:

*   `data_config`: This should be the result of the tokenization step (e.g., `tokenized_single_dataset_step` from step 1) or the mixture data configuration (e.g., `mixture_data_cfg` from step 2).
*   `model_name_or_path`: The identifier for the pretrained model you want to fine-tune (e.g., "meta-llama/Meta-Llama-3.1-8B").
*   `tokenizer`: The tokenizer instance (e.g., `TOKENIZER` defined earlier).
*   `max_seq_len`: Maximum sequence length for the model. This should match the `max_seq_len` used in data tokenization.
*   `train_batch_size`: The batch size to use for training.
*   `num_train_steps`: The total number of training steps.
*   `learning_rate`: The learning rate for the optimizer.
*   `resources`: Defines the computational resources to use for training.

Here's an example for a **single dataset SFT config**:

```python
from marin.sft.sft_config import SimpleSFTConfig
from marin.core.trainer_config import JaxDistributedConfig # Example

# sft_config_single for a single dataset
sft_config_single = SimpleSFTConfig(
    data_config=tokenized_single_dataset_step, # From Step 1
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B",
    tokenizer=TOKENIZER, # Defined in Step 1
    max_seq_len=4096,
    train_batch_size=8,
    num_train_steps=1000, 
    learning_rate=2e-5,
    resources=JaxDistributedConfig(),
)
```

And here's an example for a **mixture dataset SFT config**:
```python
# sft_config_mixture for a mixture of datasets
sft_config_mixture = SimpleSFTConfig(
    data_config=mixture_data_cfg,  # From Step 2
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B",
    tokenizer=TOKENIZER, # Defined in Step 1/2
    max_seq_len=4096, 
    train_batch_size=8,
    num_train_steps=1000, 
    learning_rate=2e-5,
    resources=JaxDistributedConfig(),
)
```
Adjust parameters like `num_train_steps` and `learning_rate` based on your specific dataset(s) and experimental goals.

### 4. Run the SFT Experiment

With the data and SFT configurations prepared, you can define and run the SFT experiment.

**Defining the SFT Training Step with `default_sft`**

The `default_sft` function is used to create the actual SFT training step. It takes the SFT configuration object prepared in the previous step.

Key argument:
*   `sft_config`: The `SimpleSFTConfig` object that holds all parameters for the SFT job, including data configuration, model details, and training hyperparameters.
*   `name` (optional): A name for this training job (e.g., "sft_training").

```python
from marin.sft.default_sft import default_sft

# For a single dataset setup:
sft_train_step_single = default_sft(
    name="sft_single_dataset_job",
    sft_config=sft_config_single, # Using the config from Step 3
)

# For a mixture dataset setup:
sft_train_step_mixture = default_sft(
    name="sft_mixture_dataset_job",
    sft_config=sft_config_mixture, # Using the mixture config from Step 3
)
```
The `default_sft` function sets up the training process using the provided configuration, handling aspects like model loading, data loading, and the training loop.

**Running the Steps with `executor_main`**

The `executor_main` function from `marin.core.execute` is used to run the defined steps. This typically includes the data tokenization step(s) and the SFT training step.

For a **single dataset** experiment, you would pass the tokenization step for that dataset and the SFT training step:

```python
from marin.core.execute import executor_main

if __name__ == "__main__":
    # Running a single dataset SFT experiment (e.g., exp606_sft.py)
    # The 'tokenized_single_dataset_step' is defined in Step 1
    # The 'sft_train_step_single' is defined above
    executor_main(tokenized_single_dataset_step, sft_train_step_single)
```

For a **mixture of datasets** experiment (e.g., `experiments/exp808_sft_mixture.py`), you need to pass all individual dataset tokenization steps followed by the SFT training step that uses the mixture configuration:

```python
from marin.core.execute import executor_main

if __name__ == "__main__":
    # Running a mixture dataset SFT experiment (e.g., exp808_sft_mixture.py)
    # The individual tokenization steps ('tokenized_tulu_step', 'tokenized_orca_step') are defined in Step 2
    # The 'sft_train_step_mixture' is defined above
    executor_main(
        tokenized_tulu_step,  # First dataset's tokenization
        tokenized_orca_step,  # Second dataset's tokenization
        # ... any other dataset tokenization steps ...
        sft_train_step_mixture # The SFT job that uses the mixture
    )
```
`executor_main` will handle the execution of these steps in sequence, ensuring that the outputs of earlier steps (like tokenized data) are available to later steps (like the training job).

This concludes the main steps for running an SFT experiment using the Marin framework. Remember to adapt file paths, model names, dataset names, and hyperparameters to your specific needs.

## Configuration Options

Most of the key configurations for an SFT experiment are centralized within the `SimpleSFTConfig` object you create in Step 3. However, you have fine-grained control over various aspects of the training process. Here are some of the most common parameters you might want to adjust:

*   **`model_name_or_path`**: (Inside `SimpleSFTConfig`)
    *   **Purpose**: Specifies the pretrained model to be fine-tuned.
    *   **Adjustment**: Change this to use a different base model (e.g., another variant of Llama, or a completely different model architecture compatible with the framework). Ensure your chosen model is accessible.

*   **`tokenizer`**: (Inside `SimpleSFTConfig`)
    *   **Purpose**: Defines the tokenizer to be used for processing your data.
    *   **Adjustment**: While often tied to the `model_name_or_path`, you might need to specify a custom tokenizer or a particular configuration of a standard tokenizer if your data or model requires it.

*   **`num_train_steps`**: (Inside `SimpleSFTConfig`)
    *   **Purpose**: Controls the total duration of the training.
    *   **Adjustment**: This can be set to a fixed number of steps. Alternatively, it's often calculated based on the number of epochs you want to train for, the size of your dataset, and the `train_batch_size`. For example: `num_train_steps = (dataset_size // global_batch_size) * num_epochs`. Adjust `num_epochs` or the calculation method to train for shorter or longer periods.

*   **`train_batch_size`**: (Inside `SimpleSFTConfig`)
    *   **Purpose**: Determines how many samples are processed before the model's weights are updated. This is the per-device batch size. The global batch size (total samples per update) will be `train_batch_size * number_of_devices`.
    *   **Adjustment**: Larger batch sizes can lead to more stable gradients but require more memory. Smaller batch sizes can introduce more noise but might offer better generalization. Adjust based on your available hardware (especially GPU memory) and training dynamics.

*   **`learning_rate`**: (Inside `SimpleSFTConfig`)
    *   **Purpose**: Controls how much the model's weights are adjusted during each update.
    *   **Adjustment**: This is a critical hyperparameter for fine-tuning. Too high, and the training might diverge; too low, and the training might be too slow or get stuck in suboptimal solutions. It often requires experimentation. Values like `1e-5`, `2e-5`, `5e-5` are common starting points for SFT.

*   **`max_seq_len`**: (Inside `SimpleSFTConfig` and often in data tokenization/configuration like `lm_mixture_data_config`)
    *   **Purpose**: Defines the maximum number of tokens in a sequence that the model can handle.
    *   **Adjustment**: This should be consistent between your data preparation (tokenization) and the model configuration. Longer sequences can capture more context but require significantly more memory and computational power. Ensure this value is appropriate for your model and dataset.

*   **`resources`**: (Inside `SimpleSFTConfig`)
    *   **Purpose**: Specifies the computational hardware for training.
    *   **Adjustment**: Configure this (e.g., using `JaxDistributedConfig` or similar classes) to define whether to use CPUs, GPUs, or TPUs, and how many devices. This is critical for scaling your training.

**Dataset-Specific Configurations:**

Parameters related to specific datasets, such as the dataset names/paths in `get_instruction_dataset` or `create_tokenization_step`, or the `mixture_weights` in a dataset mixture, are configured where those elements are defined in your script (as shown in Step 1 and Step 2 of this tutorial). You'll modify these directly in your dataset definition or mixture setup code.

Experimenting with these configuration options is key to achieving optimal performance for your specific SFT task. Always refer to the documentation of the specific functions and classes in Marin for detailed information on all available parameters.

## Example Experiments

The concepts and code snippets in this tutorial are based on the following example experiment scripts. You can refer to them for complete, runnable examples:

*   **Single Dataset SFT:** [`experiments/exp606_sft.py`](../../experiments/exp606_sft.py)
    *   This script demonstrates how to set up and run an SFT experiment using a single instruction dataset.

*   **Mixture Dataset SFT:** [`experiments/exp808_sft_mixture.py`](../../experiments/exp808_sft_mixture.py)
    *   This script shows how to configure and run an SFT experiment using a mixture of multiple instruction datasets with different weights.

These examples provide practical implementations of the steps discussed in this tutorial and can serve as a starting point for your own SFT experiments.

## Troubleshooting

Here are some common issues you might encounter when running SFT experiments and how to address them:

*   **Dataset Issues:**
    *   **Invalid Paths**: Double-check the paths provided to `get_instruction_dataset` or any custom data loading functions. Ensure the dataset exists at the specified location and is accessible.
    *   **Format Mismatches**: SFT often requires specific instruction or chat formats (e.g., Llama 3 Instruct format). Verify that your data is correctly formatted and that the `format` argument in `default_tokenize` (or your custom tokenization logic) matches your dataset's structure. Incorrect formatting can lead to the model not understanding the instructions or learning unintended patterns.
    *   **Empty or Corrupted Data**: Ensure your dataset files are not empty or corrupted.

*   **Model Checkpoint Problems:**
    *   **Invalid Path**: Verify that `model_name_or_path` in `SimpleSFTConfig` points to a valid pretrained model checkpoint that is compatible with the Marin framework.
    *   **Access Issues**: Ensure you have the necessary permissions and network access if the model is being downloaded from a hub.

*   **Configuration Parameter Errors (`SimpleSFTConfig`):**
    *   **Inconsistent `max_seq_len`**: The `max_seq_len` in `SimpleSFTConfig` must be consistent with the sequence length used during data tokenization (e.g., in `lm_mixture_data_config` or `default_tokenize`).
    *   **Batch Size Too Large**: If `train_batch_size` is too large for your available GPU/TPU memory, you'll encounter resource exhaustion errors. Reduce the batch size or use more devices.
    *   **Resource Misconfiguration**: Ensure the `resources` parameter (e.g., `JaxDistributedConfig`) is correctly set up for your environment (CPU, GPU, TPU, number of devices).

*   **Hardware and Resource Errors:**
    *   **CUDA/TPU Errors**: These can be due to driver issues, incorrect CUDA/TPU library versions, or hardware problems. Check your environment setup.
    *   **Out-of-Memory (OOM)**: This is common. Reduce `train_batch_size`, `max_seq_len`, or use gradient accumulation if supported. If using multiple GPUs/TPUs, ensure your model and data are distributed correctly.
    *   **Disk Space**: Ensure you have enough disk space for datasets, model checkpoints, and experiment outputs.

*   **Tokenizer Configuration:**
    *   **Incorrect Tokenizer**: Ensure the `tokenizer` specified in `SimpleSFTConfig` matches the `model_name_or_path`. Using a mismatched tokenizer will lead to incorrect tokenization and poor model performance.
    *   **Special Tokens**: Verify that any special tokens required by your model or chat format are correctly configured in the tokenizer.

*   **General Debugging Tips:**
    *   **Start Simple**: Begin with a small subset of your data and a small number of training steps to quickly identify configuration or data issues.
    *   **Check Logs**: Carefully examine the logs produced by the Marin framework. Error messages often provide clues about the root cause.
    *   **Isolate Changes**: If you're modifying an existing script, change one thing at a time to easily pinpoint what might have caused an issue.

## Next Steps

Once your SFT experiment has successfully run, consider the following:

*   **Evaluate the Model:**
    *   Use standard academic benchmarks (e.g., AlpacaEval, MT-Bench, HELM, LM Evaluation Harness) to assess your model's performance on various tasks.
    *   If you have specific downstream tasks, create custom evaluation sets that reflect those tasks.

*   **Compare and Iterate:**
    *   If you ran multiple experiments (e.g., different learning rates, batch sizes, dataset mixtures, or base models), compare their evaluation results to understand the impact of these changes.
    *   Use these insights to iterate on your SFT strategy. You might try different hyperparameter values, refine your dataset mixture, or experiment with different data augmentation techniques.

*   **Qualitative Analysis:**
    *   Beyond quantitative metrics, perform qualitative analysis by interacting with your fine-tuned model. Test its responses on a diverse set of prompts to understand its strengths, weaknesses, and any potential biases.

*   **Model Deployment:**
    *   If the model meets your performance criteria, you can proceed to deploy it for its intended application. This might involve converting it to a serving format, setting up an inference server, or integrating it into a larger system.

*   **Further Fine-Tuning:**
    *   Depending on the results, you might consider further stages of fine-tuning, such as reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO), to further align the model's behavior.

Supervised fine-tuning is often an iterative process. Continuous evaluation and refinement are key to developing high-performing and reliable models.
