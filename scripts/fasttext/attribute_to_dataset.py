"""
attribute_to_dataset.py

Creates a fastText training dataset from a group of datasets and labels specified by an attribute "quality-labels". Intended
to be one example of mapping attributes to a fastText training dataset.
"""

from dataclasses import dataclass
import json
import random
from typing import List, Optional

import draccus
import fsspec
import ray

from marin.utils import rebase_file_path
from marin.core.runtime import cached_or_construct_output, map_files_in_directory

@cached_or_construct_output(success_suffix="SUCCESS")
def write_fasttext_lines(input_file_path, output_file_path, attr_file_path, labels, sampling_rate, seed):
    random.seed(seed)
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(attr_file_path, "rt", compression="gzip") as f_attr, \
                fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
        for input_line,attr_line in zip(f_in,f_attr):
            data = json.loads(input_line)
            attribs = json.loads(attr_line)

            text = data["text"].replace("\n"," ")
            label_string = ''.join([f" __label__{label}" for label in attribs["attributes"]["quality-labels"] if label in labels])
            
            line = label_string + " " + text + "\n"

            p = random.random()
            if p < sampling_rate:
                f_out.write(line)

    return True

@dataclass
class LabeledDatasetConfig:
    """
    Configuration class for a labeled dataset.

    Attributes:
        path (str): Base path of the dataset (i.e., gs://{BUCKET}/documents).
        dataset (str): Dataset identifier (e.g., reddit/v0).
        doc_experiment (str): Experiment identifier under documents/ directory.
        attr_experiment (str): Experiment identifier under attributes/ directory.
        labels (List[str]): List of quality labels associated with this dataset.
        sampling_rate (float): Fraction of documents from the dataset to add to fastText training dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    path: str
    dataset: str
    doc_experiment: str
    attr_experiment: str
    labels: List[str]
    sampling_rate: float
    seed: int

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Base path for output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        data_cfgs (List[LabeledDatasetConfig]): List of LabeledDatasetConfig objects from which to construct fastText training dataset.
    """
    output_path: str
    experiment: str
    data_cfgs: List[LabeledDatasetConfig]

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()
    
    for data_cfg in cfg.data_cfgs:
        # curry write_fasttext_lines so that we can pass it to map_files_in_directory
        @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
        def processing_func(input_file_path,output_file_path):
            attr_file_path = rebase_file_path(f'{data_cfg.path}/documents/{data_cfg.doc_experiment}',input_file_path,f'{data_cfg.path}/attributes/{data_cfg.attr_experiment}')
            return write_fasttext_lines(input_file_path,output_file_path,attr_file_path,data_cfg.labels,data_cfg.sampling_rate,data_cfg.seed)

        input_dir = f'{data_cfg.path}/documents/{data_cfg.doc_experiment}/{data_cfg.dataset}'
        output_dir = rebase_file_path(f'{data_cfg.path}/documents/{data_cfg.doc_experiment}', 
                                      f'{data_cfg.path}/documents/{data_cfg.doc_experiment}/{data_cfg.dataset}', 
                                      f'{cfg.output_path}/classifiers/{cfg.experiment}/data'
                                      )
        
        responses = map_files_in_directory(processing_func.remote, input_dir, "**/*.jsonl.gz", output_dir)
        try:
            ray.get(responses)
        except Exception as e:
            print(f"Error processing {data_cfg.dataset}: {e}")

if __name__ == '__main__':
    main()