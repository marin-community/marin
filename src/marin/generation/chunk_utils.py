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

from enum import Enum
from dataclasses import dataclass
from typing import Any
import json
import re

import fsspec
import ray

from marin.core.runtime import map_files_in_directory, TaskConfig


class ChunkStrategy(str, Enum):
    # ``chunk_strategy`` determines how a document should be split into
    # smaller pieces before inference. Options are:
    #   - ``ChunkStrategy.CHAR``: split into fixed-size character windows
    #     determined by ``chunk_size``.
    #   - ``ChunkStrategy.PARAGRAPH``: split on newlines (``"\n"``) and treat
    #     each line-separated paragraph as a separate example.
    # If ``chunk_strategy`` is ``None`` no chunking is applied.
    CHAR = "char"
    PARAGRAPH = "paragraph"
    PASSAGE = "passage"


@dataclass
class ChunkingConfig:
    input_path: str
    output_path: str
    filetype: str
    chunk_strategy: ChunkStrategy
    chunk_size: int | None = None


def chunk_text(
    example: dict[str, Any],
    strategy: ChunkStrategy,
    chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    """Split an example into multiple smaller examples.

    Parameters
    ----------
    example:
        A dictionary representing a single row in the dataset. Must contain a
        ``"text"`` field and may optionally contain an ``"id"`` field.
    strategy:
        The chunking strategy to apply. ``ChunkStrategy.CHAR`` splits the
        document into fixed-size windows based on ``chunk_size`` characters.
        ``ChunkStrategy.PARAGRAPH`` splits on newlines (``"\n"``) and treats
        each resulting segment as a separate chunk.
    chunk_size:
        Size of each chunk when ``strategy`` is ``ChunkStrategy.CHAR``. Ignored for other
        strategies.

    Returns
    -------
    list[dict[str, Any]]
        A list of new examples with updated ``text`` (and ``id`` if provided).
    """

    text = example.get("text", "")

    if "id" in example:
        example_id = example.get("id")
    elif "metadata" in example:  # Extract from DCLM
        example_id = example.get("metadata").get("WARC-Record-ID")

    results: list[dict[str, Any]] = []

    if strategy is ChunkStrategy.CHAR:
        if not chunk_size or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer for 'char' strategy")
        for idx in range(0, len(text), chunk_size):
            chunk = text[idx : idx + chunk_size]
            new_example = example.copy()
            new_example_metadata = new_example.get("metadata", {})
            new_example["text"] = chunk
            if example_id is not None:
                new_example["id"] = f"{example_id}_{idx // chunk_size}"
                new_example_metadata["source_document_id"] = example_id
                new_example["metadata"] = new_example_metadata
            results.append(new_example)
    elif strategy is ChunkStrategy.PARAGRAPH:
        for idx, para in enumerate(filter(None, text.split("\n"))):
            new_example = example.copy()
            new_example_metadata = new_example.get("metadata", {})
            new_example["text"] = para
            if example_id is not None:
                new_example["id"] = f"{example_id}_{idx}"
                new_example_metadata["source_document_id"] = example_id
                new_example["metadata"] = new_example_metadata
            results.append(new_example)
    elif strategy is ChunkStrategy.PASSAGE:  # Based on https://arxiv.org/pdf/2410.20796
        # Default maximum tokens per merged passage
        max_tokens = chunk_size if (chunk_size and chunk_size > 0) else 350

        def simple_tokenize(s: str) -> list[str]:
            return re.findall(r"\S+", s)

        # 1) Split on line breaks and 2) remove empty passages
        initial_passages = [p.strip() for p in text.split("\n") if p.strip()]

        # 3) For passages exceeding max_tokens, split on sentence-ending chars followed by whitespace
        refined_passages: list[str] = []
        sentence_splitter = re.compile(r"(?<=[\.!\?])\s+")
        for p in initial_passages:
            if len(simple_tokenize(p)) <= max_tokens:
                refined_passages.append(p)
            else:
                sentences = [s.strip() for s in sentence_splitter.split(p) if s.strip()]
                # If any sentence still exceeds max_tokens (rare), break it by tokens
                for s in sentences:
                    tokens = simple_tokenize(s)
                    if len(tokens) <= max_tokens:
                        refined_passages.append(s)
                    else:
                        # Greedy token slicing to respect max_tokens
                        for start in range(0, len(tokens), max_tokens):
                            refined_passages.append(" ".join(tokens[start : start + max_tokens]))

        # 4) Merge consecutive passages up to max_tokens
        current_tokens: list[str] = []
        merged_texts: list[str] = []
        for p in refined_passages:
            p_tokens = simple_tokenize(p)
            if not current_tokens:
                current_tokens = p_tokens
                continue
            if len(current_tokens) + len(p_tokens) <= max_tokens:
                # Merge by adding a newline between passages to preserve boundaries
                current_text = " ".join(current_tokens) + "\n" + " ".join(p_tokens)
                current_tokens = simple_tokenize(current_text)
            else:
                merged_texts.append(" ".join(current_tokens))
                # Start new merge with the last passage as specified
                current_tokens = p_tokens

        if current_tokens:
            merged_texts.append(" ".join(current_tokens))

        for idx, m in enumerate(merged_texts):
            new_example = example.copy()
            new_example_metadata = new_example.get("metadata", {})

            if len(m) < 100:  # NOTE(chris): Drop if less than 100 characters, expose this as a parameter?
                continue

            new_example["text"] = m
            if example_id is not None:
                new_example["id"] = f"{example_id}_{idx}"
                new_example_metadata["source_document_id"] = example_id
                new_example_metadata["chunk_idx"] = idx
                new_example["metadata"] = new_example_metadata
            results.append(new_example)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return results


@ray.remote
def _chunk_single_file(input_path: str, output_path: str, config: ChunkingConfig) -> str:
    with (
        fsspec.open(input_path, "r", compression="infer") as in_f,
        fsspec.open(output_path, "w", compression="infer") as out_f,
    ):
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            for chunk in chunk_text(example, config.chunk_strategy, config.chunk_size):
                out_f.write(json.dumps(chunk))
                out_f.write("\n")
    return output_path


def chunk_with_config(config: ChunkingConfig) -> list[str]:
    # Launch remote tasks, one per input file
    refs = map_files_in_directory(
        _chunk_single_file.remote,
        config.input_path,
        f"**/*.{config.filetype}",
        config.output_path,
        TaskConfig(),
        False,
        config,
    )
    ray.get(refs)
