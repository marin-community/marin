import json
import os
from dataclasses import dataclass

import draccus
import fsspec
import pandas as pd
import ray

# from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.utils import fsspec_glob

SCORE_OPTIONS_DICT = {
    "Great": 5,
    "Good": 4,
    "Okay": 3,
    "Poor": 2,
    "Useless": 1,
}


@dataclass
class CountNumLabelsConfig:
    input_path: str
    output_path: str
    # Ray data outputs it into json even if it's actually a jsonl file, we will rewrite it to jsonl.gz
    # to follow dolma format
    input_filetype: str = "jsonl.gz"


@ray.remote(memory=64 * 1024 * 1024 * 1024)
def count_labels(file_paths: str, output_path: str):
    assert len(file_paths) > 0, "No files found"

    lines_flag = True if file_paths[0].endswith(".jsonl.gz") else False
    df_list = [pd.read_json(file_path, lines=lines_flag, compression="infer") for file_path in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    label_dist = df["label"].value_counts().to_dict()
    with fsspec.open(f"{output_path}/label_dist.json", "w") as f:
        json.dump(label_dist, f)

    return label_dist


@draccus.wrap()
def count_num_labels_func(config: CountNumLabelsConfig):
    files = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.input_filetype}"))

    label_dist = ray.get(count_labels.remote(files, config.output_path))

    return label_dist


if __name__ == "__main__":
    count_num_labels_func()
