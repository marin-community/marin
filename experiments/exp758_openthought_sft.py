from instruction_datasets import get_instruction_dataset

from experiments.defaults import default_sft
from experiments.llama import llama3_tokenizer, llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft

# Get instruction dataset
openthoughts_dataset = get_instruction_dataset("open-r1/OpenThoughts-114k-math")

# TODO: tune this for a good number of steps
NUM_TRAIN_STEPS = 2500

# Add tokenization step
openthoughts_llama_tokenize_step = ExecutorStep(
    name="tokenized/openthoughts_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(openthoughts_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize chat SFT data",
)


def create_sft_step(tokenization_step: ExecutorStep, seed: int = 1) -> ExecutorStep:
    """Creates an ExecutorStep for training a Llama-3.1 model on OpenThoughts dataset."""

    sft_config = SimpleSFTConfig(
        train_batch_size=128,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=5e-6,
        tpu_type="v4-128",
        tokenizer=llama3_tokenizer,
        model_name_or_path="meta-llama/Llama-3.1-8B",
        max_seq_len=4096,
        seed=seed,
        # Additional parameters from original config
        weight_decay=0.0,
        warmup=0.03,
        cooldown=0.0,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        steps_per_hf_export=500,
    )

    return default_sft(
        name="tulu3_sft_openthoughts",
        tokenized=tokenization_step,
        model_config=llama_8b,
        sft_config=sft_config,
        use_mixture=False,
        tags=["openthoughts", "llama", "sft"],
    )


if __name__ == "__main__":
    # Create training step that depends on tokenization
    training_step = create_sft_step(openthoughts_llama_tokenize_step)
    # Run all steps
    executor_main(steps=[openthoughts_llama_tokenize_step, training_step])
