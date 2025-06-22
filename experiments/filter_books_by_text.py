from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.transform.books.transform_books import FilterBooksByTextConfig, filter_books_by_text

# -----------------------------------------------------------------------------
# Editable parameters
# -----------------------------------------------------------------------------

# Substring to search for (e.g., a phrase from the desired book title or content)
SUBSTRING = "Path-relinking is introduced"  # <-- change me

# Whether the match should be case sensitive
CASE_SENSITIVE = False

# -----------------------------------------------------------------------------
# Step definition
# -----------------------------------------------------------------------------

filter_step = ExecutorStep(
    name="documents/books/filter-by-text",
    fn=filter_books_by_text,
    config=FilterBooksByTextConfig(
        input_path="gs://marin-us-central2/raw/books3/00.jsonl.zst",
        output_path=this_output_path("filtered"),
        substring=SUBSTRING,
        case_sensitive=CASE_SENSITIVE,
    ),
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(steps=[filter_step])
