import json
from collections.abc import Callable
from dataclasses import dataclass

import draccus
import fsspec
import ray

from marin.core.runtime import map_files_in_directory


@dataclass
class ReassignLabelConfig:
    input_path: str
    output_path: str
    label_func: Callable


@ray.remote
def _reassign_label(
    input_path: str,
    output_path: str,
    label_func: Callable,
):
    with fsspec.open(input_path, "rt", compression="infer") as f_in:
        with fsspec.open(output_path, "wt", compression="infer") as f_out:
            for line in f_in:
                row = json.loads(line)

                assert "label" in row
                assert "text" in row

                new_label = label_func(row["text"], row["label"])
                row["label"] = new_label

                f_out.write(json.dumps(row) + "\n")


@draccus.wrap()
def reassign_label(config: ReassignLabelConfig):
    responses = map_files_in_directory(
        _reassign_label.remote,
        config.input_path,
        "**/*.jsonl.gz",
        config.output_path,
        label_func=config.label_func,
    )

    ray.get(responses)
