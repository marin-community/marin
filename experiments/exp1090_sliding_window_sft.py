from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.transform.books.transform_books_window import (
    SingleBookTokenWindowConfig,
    transform_single_book_to_token_window_sft,
)

sliding_window_step = ExecutorStep(
    name="documents/books_token_window/great_gatsby_llama3",
    fn=transform_single_book_to_token_window_sft,
    config=SingleBookTokenWindowConfig(
        input_path="gs://marin-us-central2/documents/books/great_gatsby-b71c3c/matches.jsonl.gz",
        output_path=this_output_path(),
        tokenizer_name="meta-llama/Llama-3.1-8B",
        prompt_tokens=50,
        response_tokens=50,
        slice_length=2000,
        cursor_inc=10,
        row_index=0,
        shard_size=10000,
    ),
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(steps=[sliding_window_step])
