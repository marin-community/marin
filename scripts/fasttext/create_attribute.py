"""
create_attribute.py

Writes list of quality labels as an attribute for each document in group of datasets. Each dataset gets its own set of labels.
"""

import draccus

import fsspec
import json
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory

from typing import List
from dataclasses import dataclass
from marin.utils import rebase_file_path

@cached_or_construct_output(success_suffix="SUCCESS")
def write_labels(input_file_path: str, output_file_path: str, labels: List[str]) -> bool:
    """
    Creates an attribute "labels" from input list of labels for each document.

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output attribute JSONL file (gzip compressed).
        labels (List[str]): List of quality labels.

    Returns:
        bool: True if the process is successful.
    """
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
        for line in f_in:
            json_obj = json.loads(line)
            attributes = {"labels": labels}
            f_out.write(json.dumps({"id": json_obj["id"],
                                    "source": json_obj["source"],
                                    "attributes": attributes
                                    }) + "\n")

    return True

@dataclass
class DataLabelingConfig:
    """
    Configuration class for data labeling (i.e., what labels to write to attributes/ for which dataset).

    Attributes:
        path (str): Base path of the dataset (i.e., gs://{BUCKET}/documents).
        dataset (str): Dataset identifier. (e.g., reddit/v0).
        experiment (str): Experiment identifier.
        labels (List[str]): List of quality labels.
    """
    path: str
    dataset: str
    experiment: str
    labels: List[str]

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Base path for output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        data_cfgs (List[DataLabelingConfig]): List of DataLabelingConfig objects.
    """
    output_path: str
    experiment: str
    data_cfgs: List[DataLabelingConfig]

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()

    for data_cfg in cfg.data_cfgs:
        # curry write_labels so that we can pass it to map_files_in_directory
        @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
        def processing_func(input_file_path,output_file_path):
            return write_labels(input_file_path,output_file_path,data_cfg.labels)

        input_dir = f'{data_cfg.path}/documents/{data_cfg.dataset}/{data_cfg.experiment}'
        output_dir = f'{cfg.output_path}/attributes/{data_cfg.dataset}/{cfg.experiment}'
        
        responses = map_files_in_directory(processing_func.remote, input_dir, "**/*.jsonl.gz", output_dir)
        try:
            ray.get(responses)
        except Exception as e:
            print(f"Error processing {data_cfg.dataset}: {e}")

if __name__ == '__main__':
    main()