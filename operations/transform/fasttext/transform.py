"""
transform.py

This script converts FastText formatted files to Dolma JSONL format. Each output document includes a unique ID,
the text, the source, and the timestamp when it was added.

Usage:
    python transform.py --input_path /path/to/input.fasttext --output_path /path/to/output.jsonl.gz --source SOURCE_NAME
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output


def generate_id(line: str, line_number: str) -> str:
    """
    Generate a unique ID for a line based on its content and line number.

    Args:
        line (str): The content of the line.
        line_number (int): The line number in the file.

    Returns:
        str: A SHA-256 hash hexdigest representing the unique ID.
    """
    unique_string = f"{line_number}:{line}"
    hash_object = hashlib.sha256(unique_string.encode("utf-8"))

    return hash_object.hexdigest()


@ray.remote(memory=1 * 1024 * 1024 * 1024, num_cpus=1)  # 1 GB of RAM, 1 CPU by Default
@cached_or_construct_output(success_suffix="SUCCESS")  # Make idempotent / setup ledger for easy resumption
def convert_fasttext_to_dolma_format(input_path: str, output_path: str, source: str) -> bool:
    """
    Convert a FastText formatted file to Dolma JSONL format.

    Reads the input FastText file line by line, extracts the text, generates a unique ID, and writes it to the output
    Dolma-formatted JSONL file.

    Args:
        input_path (str): The path to the input FastText file.
        output_path (str): The path where the output Dolma JSONL file will be saved (compressed with gzip).
        source (str): The source identifier to be included in the output documents.

    Returns:
        bool: True if the conversion was successful.
    """
    with (
        fsspec.open(input_path, "rt") as f,
        fsspec.open(output_path, "wt", compression="gzip") as output_jsonl_gz,
    ):
        for line_number, line in enumerate(f):
            # Split the line to separate the fasttext label and text and remove trailing newline
            try:
                text = line[line.index(" ") + 1 :].rstrip("\n")
            except ValueError:  # skip malformed lines (missing space indicates no label)
                continue

            doc = {
                "id": generate_id(line, line_number),
                "text": text,
                "source": source,
                "added": datetime.now(timezone.utc).isoformat(),
            }

            output_jsonl_gz.write(f"{json.dumps(doc)}\n")

    return True


@dataclass
class TransformFasttextToDolmaConfig:
    """
    Configuration for transforming FastText formatted files to Dolma format.

    Attributes:
        input_path (str): The path to the input FastText file.
        output_path (str): The path to the output Dolma JSONL file (compressed with gzip).
        source (str): The source identifier to include in the output documents.
    """

    input_path: str
    output_path: str
    source: str


@draccus.wrap()
def main(cfg: TransformFasttextToDolmaConfig):
    ray.get(convert_fasttext_to_dolma_format.remote(cfg.input_path, cfg.output_path, cfg.source))


if __name__ == "__main__":
    main()
