import dataclasses
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import draccus
import fsspec
import pandas as pd
import ray

from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.core.runtime import cached_or_construct_output
from marin.transform.conversation.transform_conversation import create_shard_output_directory
from marin.utils import fsspec_glob, fsspec_mkdirs, fsspec_rm, rebase_file_path

try:
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover
    zstd = None  # We will error out at runtime if .zst is encountered without this dep


@dataclasses.dataclass
class BooksToSFTConfig:
    """Configuration to transform books into synthetic SFT dataset.

    Attributes:
        input_path: The path to the input books dataset (compressed JSONL files).
        output_path: The path to the output SFT dataset.
        start_offset: Number of characters to skip from the beginning of each book.
        window_size: Size of each sliding window in characters.
        step_size: Number of characters to advance for each new window.
        split_ratio: Fraction of window to use as prompt (rest becomes response).
        shard_size: Number of examples per output shard.
        min_book_length: Minimum book length in characters to process.
    """

    input_path: str
    output_path: str
    start_offset: int = 0
    window_size: int = 500
    step_size: int = 10
    split_ratio: float = 0.6
    shard_size: int = 10000
    min_book_length: int = 1000


@dataclasses.dataclass
class SingleBookToSFTConfig(BooksToSFTConfig):
    """Configuration for transforming a *single* book (one row in the JSONL) into an SFT dataset.

    Inherits all parameters from `BooksToSFTConfig` and adds:
        row_index: The zero-based line (row) number in the input JSONL.GZ file that contains the book to transform.
    """

    # Zero-based row/line index inside the input file identifying the book to process.
    row_index: int = 0


@dataclasses.dataclass
class FilterBooksByTextConfig:
    """Configuration to extract rows (books) whose text contains a substring.

    The `input_path` can be:
      • A single file (`*.jsonl`, `*.jsonl.gz`, `*.jsonl.zst`)
      • A directory / GCS prefix containing many compressed jsonl files. If a directory
        is provided, the script will recurse through `**/*.jsonl*` and aggregate all
        matches into the same `output_path` file.

    Attributes:
        input_path:  File or directory path. Supports local paths and any fsspec URL
                     (e.g., `gs://bucket/books/`).
        output_path: Path to *file* where all matching rows will be written.  The
                     extension determines compression (`.gz` handled via fsspec gzip`).
        substring:   Substring to search for in each row's `text` field.
        case_sensitive: Whether the match should be case-sensitive (default `False`).
    """

    input_path: str
    output_path: str
    substring: str
    case_sensitive: bool = False


def generate_sliding_windows(text: str, config: BooksToSFTConfig) -> list[dict[str, Any]]:
    """Generate sliding windows from book text and create prompt-response pairs.
    Args:
        text: The full book text
        config: Configuration for window generation
    Returns:
        List of prompt-response pairs with metadata
    """
    if len(text) < config.min_book_length:
        return []

    examples = []
    text_length = len(text)

    # Start from the configured offset
    current_offset = config.start_offset

    while current_offset + config.window_size <= text_length:
        # Extract window
        window_text = text[current_offset : current_offset + config.window_size]

        # Split into prompt and response
        split_point = int(len(window_text) * config.split_ratio)
        prompt = window_text[:split_point]
        response = window_text[split_point:]

        # Skip if prompt or response are too short
        if len(prompt.strip()) < 10 or len(response.strip()) < 5:
            current_offset += config.step_size
            continue

        # Create example
        example = {
            "prompt": prompt.strip(),
            "response": response.strip(),
            "window_offset": current_offset,
            "window_size": config.window_size,
        }
        examples.append(example)

        # Advance to next window
        current_offset += config.step_size

    return examples


def create_dolma_conversation_output(
    example: dict[str, Any], book_id: str, window_index: int
) -> DolmaConversationOutput:
    """Convert a prompt-response example to Dolma conversation format.
    Args:
        example: Dictionary with prompt, response, and metadata
        book_id: Unique identifier for the source book
        window_index: Index of this window within the book
    Returns:
        DolmaConversationOutput formatted for SFT
    """
    # Create OpenAI format messages
    messages = [
        OpenAIChatMessage(role="user", content=example["prompt"]),
        OpenAIChatMessage(role="assistant", content=example["response"]),
    ]

    return DolmaConversationOutput(
        id=f"{book_id}_window_{window_index:06d}",
        source="books-synthetic",
        messages=[msg.model_dump() for msg in messages],
        added=datetime.now(timezone.utc).isoformat(),
        created="",
        metadata={
            "book_id": book_id,
            "window_index": window_index,
            "window_offset": example["window_offset"],
            "window_size": example["window_size"],
            "prompt_length": len(example["prompt"]),
            "response_length": len(example["response"]),
        },
    )


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _process_book_file(input_file_path: str, output_dir: str, config: BooksToSFTConfig):
    """Process a single book file and generate SFT examples.
    Args:
        input_file_path: Path to the input JSONL.gz file
        output_dir: Directory to write output shards
        config: Configuration for transformation
    """
    # Read the compressed JSONL file
    book_data = pd.read_json(input_file_path, lines=True, compression="gzip")

    all_examples = []

    for idx, row in book_data.iterrows():
        if "text" not in row:
            continue

        book_text = row["text"]
        if not isinstance(book_text, str):
            continue

        # Generate book ID from file path and row index
        book_id = f"{os.path.basename(input_file_path).replace('.jsonl.gz', '')}_{idx}"

        # Generate sliding window examples
        examples = generate_sliding_windows(book_text, config)

        # Convert to Dolma format
        for window_idx, example in enumerate(examples):
            dolma_output = create_dolma_conversation_output(example, book_id, window_idx)
            all_examples.append(dolma_output.model_dump())

    # Write examples to sharded output files
    shard_count = 0
    for i in range(0, len(all_examples), config.shard_size):
        shard_examples = all_examples[i : i + config.shard_size]
        shard_path = os.path.join(output_dir, f"shard_{shard_count:05d}.jsonl.gz")

        with fsspec.open(shard_path, "wt", compression="gzip") as f:
            for example in shard_examples:
                f.write(f"{json.dumps(example)}\n")

        shard_count += 1

    return len(all_examples)


@ray.remote
def _process_books_dataset(config: BooksToSFTConfig):
    """Process all book files in the dataset.
    Args:
        config: Configuration for the transformation
    """
    # Find all JSONL.gz files in the input path
    file_paths = fsspec_glob(os.path.join(config.input_path, "**/*.jsonl.gz"))

    if not file_paths:
        raise ValueError(f"No JSONL.gz files found in {config.input_path}")

    max_tasks_in_flight = 50
    responses = []

    for input_filepath in file_paths:
        # Wait if too many tasks are running
        if len(responses) >= max_tasks_in_flight:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            ray.get(ready_refs)

        # Create output directory for this file
        output_filepath = rebase_file_path(config.input_path, input_filepath, config.output_path)
        output_dir = create_shard_output_directory(output_filepath)

        # Process the file
        result_ref = _process_book_file.options(memory=8 * 1024 * 1024 * 1024, num_cpus=2).remote(  # 8GB memory
            input_filepath, output_dir, config
        )

        responses.append(result_ref)

    # Wait for all remaining tasks
    results = ray.get(responses)
    total_examples = sum(results)

    print(f"Generated {total_examples} SFT examples from {len(file_paths)} book files")
    return total_examples


@draccus.wrap()
def transform_books_to_sft(config: BooksToSFTConfig):
    """Main function to transform books into synthetic SFT dataset.
    Args:
        config: Configuration for the transformation
    """
    return ray.get(_process_books_dataset.remote(config))


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _process_single_book(input_file_path: str, row_index: int, output_dir: str, config: BooksToSFTConfig):
    """Process *one* book (identified by its row index) inside a JSONL.gz file and generate SFT examples.

    Args:
        input_file_path: Path to the input JSONL.gz file.
        row_index:       Zero-based index of the line containing the book.
        output_dir:      Directory to write output shards.
        config:          Transformation configuration (window size, shard size, etc.).
    """
    # Load only the requested line to avoid reading the full file into memory.
    with fsspec.open(input_file_path, "rt", compression="gzip") as f:
        selected_line: str | None = None
        for idx, line in enumerate(f):
            if idx == row_index:
                selected_line = line
                break

    if selected_line is None:
        raise IndexError(f"Row index {row_index} not found in {input_file_path}")

    row_data = json.loads(selected_line)

    if "text" not in row_data or not isinstance(row_data["text"], str):
        raise ValueError(f"Row {row_index} in {input_file_path} does not contain valid 'text' field")

    book_text = row_data["text"]
    book_id = f"{os.path.basename(input_file_path).replace('.jsonl.gz', '')}_{row_index}"

    # Generate sliding window examples for this single book.
    examples = generate_sliding_windows(book_text, config)

    # Convert to Dolma format
    all_examples: list[dict[str, Any]] = []
    for window_idx, example in enumerate(examples):
        dolma_output = create_dolma_conversation_output(example, book_id, window_idx)
        all_examples.append(dolma_output.model_dump())

    # Write examples to sharded output files
    shard_count = 0
    for i in range(0, len(all_examples), config.shard_size):
        shard_examples = all_examples[i : i + config.shard_size]
        shard_path = os.path.join(output_dir, f"shard_{shard_count:05d}.jsonl.gz")
        with fsspec.open(shard_path, "wt", compression="gzip") as f:
            for example in shard_examples:
                f.write(f"{json.dumps(example)}\n")
        shard_count += 1

    return len(all_examples)


@draccus.wrap()
def transform_single_book_to_sft(config: SingleBookToSFTConfig):
    """Transform a single book (specified by `row_index`) into an SFT dataset.

    The output directory will be `<output_path>/<input_filename_without_suffix>_row_<row_index>`.
    """
    # Derive an output file path to compute shard directory (re-uses utility from conversation transform).
    input_basename = os.path.basename(config.input_path).replace(".jsonl.gz", "")
    output_filename = os.path.join(config.output_path, f"{input_basename}_row_{config.row_index}.jsonl.gz")
    output_dir = create_shard_output_directory(output_filename)

    total_examples = ray.get(
        _process_single_book.remote(
            config.input_path,
            config.row_index,
            output_dir,
            config,
        )
    )

    print(f"Generated {total_examples} SFT examples for row {config.row_index} in {config.input_path} → {output_dir}")
    return total_examples


# -----------------------------------------------------------------------------
# Internal utility for streaming JSONL
# -----------------------------------------------------------------------------


def _iter_jsonl_lines(path: str):
    """Yield lines from a JSONL file, transparently handling .gz and .zst.

    Args:
        path: Path (local or remote via fsspec) to the file.

    Yields:
        str: Each line of the decompressed file.
    """
    if path.endswith(".zst"):
        if zstd is None:
            raise ImportError("Install `zstandard` to read .zst files.")
        stream = zstd.ZstdDecompressor().stream_reader(fsspec.open(path, "rb").open())
        yield from io.TextIOWrapper(stream, encoding="utf-8")
    else:
        # fsspec infers gzip/bz2/xz/zip etc. from file extension.
        with fsspec.open(path, "rt", compression="infer") as f:
            yield from f


# -----------------------------------------------------------------------------
# Ray map task: filter a single file and write matches to a temporary part file
# -----------------------------------------------------------------------------


@ray.remote
def _filter_single_file(file_path: str, cfg: FilterBooksByTextConfig, part_dir: str) -> tuple[str, int]:
    """Return (temporary_part_path, number_of_matches) for one JSONL shard."""

    match_substr = cfg.substring if cfg.case_sensitive else cfg.substring.lower()

    part_suffix = ""
    part_filename = f"{os.path.basename(file_path)}.{uuid.uuid4().hex}.part{part_suffix}"
    part_path = os.path.join(part_dir, part_filename)

    matches = 0
    with fsspec.open(part_path, "wt", compression=None) as fout:
        for line in _iter_jsonl_lines(file_path):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text_val = obj.get("text", "")
            if not isinstance(text_val, str):
                continue

            haystack = text_val if cfg.case_sensitive else text_val.lower()
            if match_substr in haystack:
                fout.write(line if line.endswith("\n") else line + "\n")
                matches += 1

    return part_path, matches


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _filter_books_by_text(input_path: str, output_path: str, config: FilterBooksByTextConfig):
    """Scan `input_path` (file or directory) in parallel and write all matching rows to `output_path`."""

    # Determine actual output *file* path. If the provided output_path does not
    # look like a filename (no .json/.jsonl/.gz suffix), treat it as a directory
    # and create a default filename inside it.
    if any(output_path.endswith(ext) for ext in (".jsonl", ".jsonl.gz", ".jsonl.zst")):
        final_output_path = output_path
    else:
        final_output_path = os.path.join(output_path, "matches.jsonl.gz")

    # Ensure parent directory for output exists
    output_dir = os.path.dirname(final_output_path)
    if output_dir:
        fsspec_mkdirs(output_dir)

    # Identify input files
    if any(input_path.endswith(ext) for ext in (".jsonl", ".jsonl.gz", ".jsonl.zst")):
        file_paths = [input_path]
    else:
        file_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl*"))

    if not file_paths:
        raise FileNotFoundError(f"No jsonl files found under {input_path}")

    # Temporary directory to hold part files (one per shard)
    part_dir = os.path.join(os.path.dirname(final_output_path), f"_parts_{uuid.uuid4().hex}")
    fsspec_mkdirs(part_dir)

    # Launch mapper tasks in parallel
    futures = [_filter_single_file.remote(fp, config, part_dir) for fp in file_paths]
    results = ray.get(futures)  # List of (part_path, matches)

    total_matches = sum(m for _p, m in results)

    # Write final output, compressing once if filename ends with .gz
    final_compression = "gzip" if final_output_path.endswith(".gz") else None

    write_mode = "wb" if final_compression else "wt"
    # for binary writing when gzip; for text writing otherwise

    with fsspec.open(final_output_path, write_mode, compression=final_compression) as fout:
        for part_path, _ in results:
            with fsspec.open(part_path, "rb", compression=None) as fin:
                fout.write(fin.read())

            # Clean up temporary part file
            try:
                fsspec_rm(part_path)
            except Exception:
                pass

    print(
        f"Found {total_matches} matching rows '{config.substring}' across {len(file_paths)} file(s) under {input_path}. "
        f"Wrote results to {final_output_path}."
    )
    return total_matches


@draccus.wrap()
def filter_books_by_text(config: FilterBooksByTextConfig):
    """User-facing wrapper to launch Ray task that filters books by substring."""

    return ray.get(_filter_books_by_text.remote(config.input_path, config.output_path, config))


if __name__ == "__main__":
    transform_books_to_sft()
