"""
attribute_to_dataset.py

Creates a fastText training dataset from a group of datasets and labels specified by an attribute "labels". Intended
to be one example of mapping attributes to a fastText training dataset.
"""

from dataclasses import dataclass
import json
from typing import List, Optional

import draccus
import fsspec
import ray
import numpy as np

from marin.utils import rebase_file_path
from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.classifiers.fasttext.utils import preprocess as preprocess_for_fasttext

@cached_or_construct_output(success_suffix="SUCCESS")
def write_fasttext_lines(input_file_path : str, output_file_path : str, attr_file_path : str, labels : List[str], sampling_rate : float, seed : int) -> bool:
    """
    Labels each line of input file according to fastText training format and writes to an output file.
    Only a fraction of the lines, determined by the sampling rate, are written to the output file (eg, to control size 
    of training dataset and/or weight different domains).

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output file (gzip compressed).
        labels (List[str]): List of labels to be added to each line.
        sampling_rate (float): Fraction of lines to be written to the output file.
        seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(attr_file_path, "rt", compression="gzip") as f_attr, \
                fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
        for input_line, attr_line in zip(f_in, f_attr):
            data = json.loads(input_line)
            attribs = json.loads(attr_line)

            text = preprocess_for_fasttext(data["text"])
            label_string = ''.join([f" __label__{label}" for label in attribs["attributes"]["labels"] if label in labels])
            
            line = label_string + " " + text + "\n"

            p = rng.random()
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
        doc_dir = f'{data_cfg.path}/documents/{data_cfg.doc_experiment}'
        attr_dir = f'{data_cfg.path}/attributes/{data_cfg.attr_experiment}'

        # curry write_fasttext_lines so that we can pass it to map_files_in_directory
        @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
        def processing_func(input_file_path : str,output_file_path : str) -> bool:
            attr_file_path = rebase_file_path(doc_dir,input_file_path,attr_dir)
            return write_fasttext_lines(input_file_path,output_file_path,attr_file_path,data_cfg.labels,data_cfg.sampling_rate,data_cfg.seed)

        input_dir = f'{data_cfg.path}/documents/{data_cfg.doc_experiment}/{data_cfg.dataset}'
        output_dir = rebase_file_path(doc_dir, 
                                      f'{doc_dir}/{data_cfg.dataset}', 
                                      f'{cfg.output_path}/classifiers/{cfg.experiment}/data'
                                      )
        
        responses = map_files_in_directory(processing_func.remote, input_dir, "**/*.jsonl.gz", output_dir)
        try:
            ray.get(responses)
        except Exception as e:
            print(f"Error processing {data_cfg.dataset}: {e}")

if __name__ == '__main__':
    main()