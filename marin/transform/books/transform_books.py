import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import draccus
import fsspec
import pandas as pd
import ray

from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.core.runtime import cached_or_construct_output
from marin.transform.conversation.transform_conversation import create_shard_output_directory
from marin.utils import fsspec_glob, rebase_file_path


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
        window_text = text[current_offset:current_offset + config.window_size]
        
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
    example: dict[str, Any], 
    book_id: str, 
    window_index: int
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
        OpenAIChatMessage(role="assistant", content=example["response"])
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
        }
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
        shard_examples = all_examples[i:i + config.shard_size]
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
        result_ref = _process_book_file.options(
            memory=8 * 1024 * 1024 * 1024,  # 8GB memory
            num_cpus=2
        ).remote(input_filepath, output_dir, config)
        
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


if __name__ == "__main__":
    transform_books_to_sft() 