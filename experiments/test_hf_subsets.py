from experiments.defaults import default_tokenize
from experiments.exp964_custom_chat_tokenizer import llama3_instruct_chat_format
from experiments.instruction_datasets import get_instruction_dataset
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

# Get instruction dataset

smoltalk_train = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["train"])
smoltalk_test = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["test"])

# tokenization steps


smoltalk_train_llama_tokenize_step = default_tokenize(
    name="smoltalk_train_llama3_tokenizer",
    dataset=smoltalk_train / "**/*.jsonl.gz",
    tokenizer=llama3_tokenizer,
    format=llama3_instruct_chat_format,
)

smoltalk_test_llama_tokenize_step = default_tokenize(
    name="smoltalk_test_llama3_tokenizer",
    dataset=smoltalk_test / "**/*.jsonl.gz",
    tokenizer=llama3_tokenizer,
    format=llama3_instruct_chat_format,
    is_validation=True,
)


if __name__ == "__main__":
    executor_main(steps=[smoltalk_train_llama_tokenize_step, smoltalk_test_llama_tokenize_step])
