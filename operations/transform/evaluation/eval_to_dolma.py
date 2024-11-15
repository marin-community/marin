"""Given a directory of evaluation files, convert them to Dolma format.

The Dolma format is a JSONL file with a "text" field.
"""

import json
from dataclasses import dataclass

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory


@dataclass
class ConvertEvalToDolmaConfig:
    input_path: str
    output_path: str


def map_row(row: dict):
    row["text"] = row["prompt"] + "\n" + row["response"]
    return row


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _process_file(input_filename: str, output_filename: str):
    """Convert an evaluation file with "prompt" and "response" fields to Dolma format with a "text" field."""
    with fsspec.open(input_filename, "rt") as f_in:
        with fsspec.open(output_filename, "wt") as f_out:
            for line in f_in:
                row = json.loads(line)
                dolma_row = map_row(row)
                f_out.write(f"{json.dumps(dolma_row)}\n")


@draccus.wrap()
def convert_eval_to_dolma(cfg: ConvertEvalToDolmaConfig):
    ray.get(map_files_in_directory.remote(_process_file.remote, cfg.input_path, "**/*.jsonl.gz", cfg.output_path))


if __name__ == "__main__":
    convert_eval_to_dolma()
