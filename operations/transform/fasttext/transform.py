import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output


# Function to generate a unique hash ID based on line content and line number
def generate_id(line, line_number):
    unique_string = f"{line_number}:{line}"
    hash_object = hashlib.sha256(unique_string.encode("utf-8"))

    return hash_object.hexdigest()


@ray.remote(memory=4 * 1024 * 1024 * 1024, num_cpus=1)  # 4 GB of RAM, 1 CPU by Default
@cached_or_construct_output(success_suffix="SUCCESS")  # Make idempotent / setup ledger for easy resumption
def convert_to_dolma_format(input_path, output_path, source):
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


@dataclass
class TransformFasttextToDolmaConfig:
    input_path: str
    output_path: str
    source: str


@draccus.wrap()
def main(cfg: TransformFasttextToDolmaConfig):
    convert_to_dolma_format.remote(cfg.input_path, cfg.output_path, cfg.source)
