from enum import Enum
from dataclasses import dataclass
from typing import Any
import json

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
    example_id = example.get("id")
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
