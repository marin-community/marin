# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import os
from datetime import datetime, timezone
from typing import Any

import fsspec
import ray
from transformers import AutoTokenizer

from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.core.runtime import cached_or_construct_output
from marin.transform.conversation.transform_conversation import (
    create_shard_output_directory,
)

# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class SingleBookTokenWindowConfig:
    """Configuration for sliding-window token-based SFT generation from a *single* book.

    Attributes
    ----------
    input_path: str
        Path to the compressed ``*.jsonl.gz`` file containing the book shard.
    output_path: str
        Directory or GCS prefix where the output shards will be written.
    tokenizer_name: str
        HuggingFace tokenizer name (e.g. ``"gpt2"``).
    prompt_tokens: int
        Exact number of tokens to use for the prompt.
    response_tokens: int
        Exact number of tokens to use for the response.
    slice_length: int, default 2000
        Number of *characters* to read from the book at each cursor position before
        tokenising.
    cursor_inc: int, default 1
        How many *characters* to move the cursor forward after processing a slice.
    row_index: int, default 0
        Zero-based index of the row in ``input_path`` identifying the book.
    shard_size: int, default 10000
        Number of examples per output shard.
    """

    input_path: str
    output_path: str
    tokenizer_name: str = "gpt2"
    prompt_tokens: int = 50
    response_tokens: int = 50
    slice_length: int = 2000
    cursor_inc: int = 1
    row_index: int = 0
    shard_size: int = 10000

    # Convenience: total tokens per example
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _generate_token_windows(text: str, tokenizer, cfg: SingleBookTokenWindowConfig) -> list[dict[str, Any]]:
    """Generate prompt/response pairs using token-based sliding windows.

    The algorithm mirrors ``chunk_text_to_sliding_window_token_chunks`` from
    *sliding_new.md* but splits the first ``cfg.total_tokens`` tokens into a
    prompt and a response and discards the remainder of the slice.
    """

    examples: list[dict[str, Any]] = []
    text_cursor = 0
    text_length = len(text)

    while text_cursor < text_length:
        # Character-level slice
        start_idx = text_cursor
        end_idx_plus_one = min(text_cursor + cfg.slice_length, text_length)
        text_slice = text[start_idx:end_idx_plus_one]

        # Tokenise
        encoding = tokenizer(
            text_slice,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"]

        # Need at least prompt_tokens + response_tokens
        if len(input_ids) >= cfg.total_tokens:
            prompt_ids = input_ids[: cfg.prompt_tokens]
            response_ids = input_ids[cfg.prompt_tokens : cfg.total_tokens]

            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Compute approximate character end position for metadata
            decoded_total = tokenizer.decode(input_ids[: cfg.total_tokens], skip_special_tokens=True)
            decoding_len = len(decoded_total)
            end_idx = start_idx + decoding_len - 1

            examples.append(
                {
                    "prompt": prompt_text.strip(),
                    "response": response_text.strip(),
                    "window_offset": start_idx,
                    "window_size_chars": cfg.slice_length,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "text_len": decoding_len,
                }
            )

        # Slide the character cursor
        text_cursor += cfg.cursor_inc

    return examples


def _create_dolma_output(
    example: dict[str, Any],
    book_id: str,
    window_index: int,
    cfg: SingleBookTokenWindowConfig,
) -> DolmaConversationOutput:
    """Convert one example into ``DolmaConversationOutput``."""

    messages = [
        OpenAIChatMessage(role="user", content=example["prompt"]),
        OpenAIChatMessage(role="assistant", content=example["response"]),
    ]

    return DolmaConversationOutput(
        id=f"{book_id}_window_{window_index:06d}",
        source="books-synthetic-token-window",
        messages=[m.model_dump() for m in messages],
        added=datetime.now(timezone.utc).isoformat(),
        created="",
        metadata={
            "book_id": book_id,
            "window_index": window_index,
            "window_offset": example["window_offset"],
            "slice_length_chars": cfg.slice_length,
            "cursor_inc": cfg.cursor_inc,
            "prompt_tokens": cfg.prompt_tokens,
            "response_tokens": cfg.response_tokens,
            "tokenizer_name": cfg.tokenizer_name,
        },
    )


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
# The first two parameters are used by the caching decorator
def _process_single_book_token_window(
    input_file_path: str,
    output_dir: str,
    row_index: int,
    cfg: SingleBookTokenWindowConfig,
):
    """Process one book (by ``row_index``) and write token-window SFT shards."""

    # Instantiate tokenizer inside the worker for determinism
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    # Stream until we reach the desired row
    with fsspec.open(input_file_path, "rt", compression="gzip") as f:
        selected_line: str | None = None
        for idx, line in enumerate(f):
            if idx == row_index:
                selected_line = line
                break
    if selected_line is None:
        raise IndexError(f"Row {row_index} not found in {input_file_path}")

    row_data = json.loads(selected_line)
    if not isinstance(row_data.get("text"), str):
        raise ValueError("Selected row does not contain valid 'text' field")

    book_text = row_data["text"]
    book_id = f"{os.path.basename(input_file_path).replace('.jsonl.gz', '')}_{row_index}"

    # Generate examples
    examples = _generate_token_windows(book_text, tokenizer, cfg)

    # Convert to Dolma format
    dolma_examples: list[dict[str, Any]] = []
    for w_idx, ex in enumerate(examples):
        dolma_output = _create_dolma_output(ex, book_id, w_idx, cfg)
        dolma_examples.append(dolma_output.model_dump())

    # Write shards
    shard_count = 0
    for i in range(0, len(dolma_examples), cfg.shard_size):
        shard_examples = dolma_examples[i : i + cfg.shard_size]
        shard_path = os.path.join(output_dir, f"shard_{shard_count:05d}.jsonl.gz")
        with fsspec.open(shard_path, "wt", compression="gzip") as fout:
            for obj in shard_examples:
                fout.write(json.dumps(obj) + "\n")
        shard_count += 1

    return len(dolma_examples)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def transform_single_book_to_token_window_sft(
    cfg: SingleBookTokenWindowConfig,
):
    """User-facing wrapper to generate token-window SFT for one book."""

    # Derive output directory name
    input_basename = os.path.basename(cfg.input_path).replace(".jsonl.gz", "")
    outfile = os.path.join(
        cfg.output_path,
        f"{input_basename}_row_{cfg.row_index}.jsonl.gz",
    )
    output_dir = create_shard_output_directory(outfile)

    total_examples = ray.get(_process_single_book_token_window.remote(cfg.input_path, output_dir, cfg.row_index, cfg))

    print(f"Generated {total_examples} token-window SFT examples for row {cfg.row_index} → {output_dir}")
    return total_examples
