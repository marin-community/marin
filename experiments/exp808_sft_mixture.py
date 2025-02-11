from instruction_datasets import get_instruction_dataset

from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft

# Get instruction dataset
acecode_dataset = get_instruction_dataset("TIGER-Lab/AceCode-89K")
smoltalk_dataset = get_instruction_dataset("HuggingFaceTB/smoltalk")
prime_verified_math_dataset = get_instruction_dataset("PrimeIntellect/verifiable-math-problems")

# tokenization steps

prime_verified_math_llama_tokenize_step = ExecutorStep(
    name="tokenized/prime_verified_math_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(prime_verified_math_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize SFT data",
)

acecode_llama_tokenize_step = ExecutorStep(
    name="tokenized/acecode_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(acecode_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize SFT data",
)

smoltalk_llama_tokenize_step = ExecutorStep(
    name="tokenized/smoltalk_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(smoltalk_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize SFT data",
)

if __name__ == "__main__":
    executor_main(
        steps=[acecode_llama_tokenize_step, smoltalk_llama_tokenize_step, prime_verified_math_llama_tokenize_step]
    )
