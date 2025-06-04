"""
A quick-start experiment to run SFT on a small mixture of two datasets.

The default below trains a small model for a small number of training steps.
If you wish to fine-tune a different model on a different datasets,
you will need to change the model, data, and number of training steps
accordingly.
"""

from levanter.data.text import ChatLmDatasetFormat
from levanter.models.llama import LlamaConfig
from experiments.defaults import default_sft, default_tokenize
from experiments.marin_models import marin_tokenizer
from experiments.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig


def create_tokenization_step(dataset_name: str) -> ExecutorStep:
    """
    Creates a tokenization ExecutorStep for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'TIGER-Lab/AceCode-89K',
    'bespokelabs/Bespoke-Stratos-17k')

    Returns:
        ExecutorStep configured for tokenizing the specified dataset
    """
    # Get the dataset with only train split
    dataset = get_instruction_dataset(dataset_name, splits=["train"])

    # Get the last part of the path and clean it up
    short_name = dataset_name.split("/")[-1].lower().replace("-", "_")
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer_how_to_asdf1",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


# Dataset configurations
DATASETS = {
    "acecode_89k": "TIGER-Lab/AceCode-89K",
    "bespoke_stratos_17k": "bespokelabs/Bespoke-Stratos-17k",
}

NUM_TRAIN_STEPS = 100  # A small number 

# Create tokenization steps for multiple datasets
tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

# Dataset weights set with the naive baseline of the number of documents per dataset
mixture_weights = {
    "acecode_89k": 87149,
    "bespoke_stratos_17k": 16710,
}


assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

SEQ_LEN = 2048

# Define an SFT config appropriate for mixture training
mixture_sft_config = SimpleSFTConfig(
    train_batch_size=32,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-6,
    resources=TpuPodConfig(tpu_type="v4-8"),
    tokenizer=marin_tokenizer,
    max_seq_len=SEQ_LEN,
    initialize_from_hf=False,
    initialize_from_checkpoint_path="gs://marin-us-central2/checkpoints/dclm_1b_1x_how_to-58c8f0/checkpoints/step-54598",
    seed=0,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=True,
    missing_weights_are_validation=True,
)

llama_1_4b_dclm = LlamaConfig(
    seq_len=SEQ_LEN,  # Maximum sequence length for processing context
    hidden_dim=2048,  # Dimension of hidden representations
    intermediate_dim=8192,  # Dimension of feedforward layers (4x hidden_dim)
    num_heads=16,  # Number of attention heads
    num_kv_heads=16,  # Number of key/value heads (equal to num_heads = no grouped-query attention)
    num_layers=24,  # Number of transformer layers
    use_flash_attention=True,
)

# Configure mixture-based SFT training
training_step = default_sft(
    name="dclm_1b_1x_tokenize_sft_how_to_asdf1",
    tokenized=mixture_config,
    model_config=llama_1_4b_dclm,
    sft_config=mixture_sft_config,
    tags=["HOWTOS", "DCLM_1B_1X", "SFT"],
)


if __name__ == "__main__":
    # Run all steps
    executor_main(steps=[*list(tokenized_datasets.values()), training_step])
