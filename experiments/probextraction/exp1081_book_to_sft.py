from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.transform.books.transform_books import SingleBookToSFTConfig, transform_single_book_to_sft

# -----------------------------------------------------------------------------
# Configurations (edit these for your specific book)
# -----------------------------------------------------------------------------

# Path to the compressed JSONL file that contains the book data
INPUT_BOOK_FILE = "gs://marin-us-central2/documents/books/great_gatsby-b71c3c/matches.jsonl.gz"  # <-- change me

# Row index inside INPUT_BOOK_FILE that contains the book you want to process
ROW_INDEX = 0  # <-- change me

# Output base path (the executor will prepend its prefix automatically)

# Sliding-window parameters for splitting the book
WINDOW_SIZE = 500
STEP_SIZE = 10
SPLIT_RATIO = 0.4  # 40 % prompt, 60 % response
START_OFFSET = 0
SHARD_SIZE = 10_000


# -----------------------------------------------------------------------------
# Executor step
# -----------------------------------------------------------------------------

book_to_sft_step = ExecutorStep(
    name="documents/books_windowed/great_gatsby",
    fn=transform_single_book_to_sft,
    config=SingleBookToSFTConfig(
        input_path=INPUT_BOOK_FILE,
        output_path=this_output_path(),
        start_offset=START_OFFSET,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        split_ratio=SPLIT_RATIO,
        shard_size=SHARD_SIZE,
        row_index=ROW_INDEX,
    ),
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # The executor prefix (e.g., gs://bucket) is provided via --prefix or $MARIN_PREFIX.
    executor_main(steps=[book_to_sft_step])
