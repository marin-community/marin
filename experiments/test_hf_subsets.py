from instruction_datasets import get_instruction_dataset

from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft

# Get instruction dataset

smoltalk_train = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["train"])
smoltalk_test = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["test"])

# tokenization steps


smoltalk_train_llama_tokenize_step = ExecutorStep(
    name="tokenized/smoltalk_train_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(smoltalk_train, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize SFT data for smoltalk train",
)

smoltalk_test_llama_tokenize_step = ExecutorStep(
    name="tokenized/smoltalk_test_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(smoltalk_test, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize SFT data for smoltalk test",
)

if __name__ == "__main__":
    executor_main(steps=[smoltalk_train_llama_tokenize_step, smoltalk_test_llama_tokenize_step])
