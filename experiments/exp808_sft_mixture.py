from datetime import timedelta

import jmp
from instruction_datasets import get_instruction_dataset
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import SupervisedUrlSourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.rotary import Llama3RotaryEmbeddingsConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft
from marin.training.training import TrainSFTMixturePodConfig, run_levanter_sft_mixture


def create_tokenization_step(dataset_name: str) -> ExecutorStep:
    """Creates a tokenization ExecutorStep for a given dataset."""
    # Get the dataset with only train split
    dataset = get_instruction_dataset(dataset_name, splits=["train"])

    # Get the last part of the path and clean it up
    short_name = dataset_name.split("/")[-1].lower().replace("-", "_")

    return ExecutorStep(
        name=f"tokenized/{short_name}_llama3_instruct_tokenizer",
        fn=levanter_tokenize_sft,
        config=TokenizeConfig(
            train_paths=[output_path_of(dataset, "**/*.jsonl.gz")],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer="meta-llama/Llama-3.1-8B-Instruct",
            input_field="user",
            output_field="assistant",
        ),
        description="Tokenize SFT data",
    )


# Dataset configurations
DATASETS = [
    "TIGER-Lab/AceCode-89K",
    "HuggingFaceTB/smoltalk",
    "PrimeIntellect/verifiable-math-problems",
    "cognitivecomputations/dolphin-r1-nonreasoning",
    "cognitivecomputations/dolphin-r1-reasoning",
    "bespokelabs/Bespoke-Stratos-17k",
    "open-r1/OpenThoughts-114k-math",
    "allenai/tulu-3-sft-mixture",
    "facebook/natural_reasoning",
]

NUM_TRAIN_STEPS = 19086  # From your YAML config


# Add training step
def create_training_step(tokenization_steps: list[ExecutorStep], seed: int = 0) -> ExecutorStep:
    # Create a mapping of cache dirs for each dataset from tokenization steps
    supervised_data = {}
    mixture_weights = {
        "tulu": 939343,
        "openthoughts": 89120,
        "prime_verified_math": 777457,
        "acecode": 87149,
        "smoltalk": 1043917,
        "natural_reasoning": 1145824,
        "stratos": 16710,
    }

    for step in tokenization_steps:
        # Extract short name from step name
        short_name = step.name.split("/")[-1].replace("_llama3_instruct_tokenizer", "")
        supervised_data[short_name] = SupervisedUrlSourceConfig(
            cache_dir=output_path_of(step),
            train_urls=[output_path_of(step, "**/*.jsonl.gz")],
            input_field="user",
            output_field="assistant",
        )

    return ExecutorStep(
        name=f"checkpoints/llama3.1_mixture_total_seed{seed}",
        fn=run_levanter_sft_mixture,  # Use the train function from sft_mixture.py
        config=TrainSFTMixturePodConfig(
            trainer=TrainerConfig(
                seed=seed,
                tracker=WandbConfig(project="marin", tags=["dolma", "olmo", "llama", "mixture"]),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=128,
                num_train_steps=19086,
                steps_per_eval=1000,
                tensor_parallel_axes=["mlp", "heads"],
                fsdp_axis="embed",
                batch_axis="batch",
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=30),
                    keep=[dict(every=25000)],
                ),
            ),
            model=LlamaConfig(
                seq_len=4096,
                hidden_dim=4096,
                intermediate_dim=14336,
                num_layers=32,
                num_heads=32,
                num_kv_heads=8,
                use_flash_attention=True,
                flash_attention_block_size=512,
                use_bias=False,
                use_layer_norm_weight=True,
                initializer_range=0.02,
                rope=Llama3RotaryEmbeddingsConfig(
                    theta=500000,
                    factor=8.0,
                    low_freq_factor=1.0,
                    high_freq_factor=4.0,
                    original_max_position_embeddings=8192,
                ),
            ),
            optimizer=AdamConfig(
                learning_rate=5e-6,
                weight_decay=0.0,
                min_lr_ratio=0.0,
                lr_schedule="linear",
                warmup=0.03,
            ),
            # Mixture specific configuration
            supervised_data=supervised_data,
            mixture_weights=mixture_weights,
            mixture_block_size=2048,
            stop_strategy="restart",
            # Model loading configuration
            max_seq_len=4096,
            tokenizer="meta-llama/Llama-3.1-8B-Instruct",
            model_name_or_path="meta-llama/Llama-3.1-8B",
            initialize_from_hf=True,
            # HF checkpoint saving
            hf_save_steps=1000,
            # Chat format configuration
            messages_field="messages",
            input_role="user",
            output_role="assistant",
        ),
    )


if __name__ == "__main__":
    # Create tokenization steps for all datasets
    tokenization_steps = [create_tokenization_step(dataset_name) for dataset_name in DATASETS]

    # Create training step that depends on tokenization
    training_step = create_training_step(tokenization_steps)

    # Run all steps
    executor_main(steps=[*tokenization_steps, training_step])
