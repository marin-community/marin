"""
attribute_to_dataset.py

Creates a fastText training dataset from a group of datasets and labels specified by an attribute "label". Intended
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

def get_label(data : dict, attribs : dict) -> List[str]:
    """
    Extracts label from attributes dictionary.

    Args:
        data (dict): Data dictionary (i.e., from documents/..).
        attribs (dict): Attributes dictionary (i.e., from attributes/..).

    Returns:
        str: Quality classifier label.
    """
    return attribs["attributes"]["label"]

@cached_or_construct_output(success_suffix="SUCCESS")
def write_fasttext_lines(input_file_path : str, output_file_path : str, attr_file_path : str, sampling_rate : float, seed : int) -> bool:
    """
    Labels each line of input file according to fastText training format and writes to an output file.
    Only a fraction of the lines, determined by the sampling rate, are written to the output file (eg, to control size 
    of training dataset and/or weight different domains).

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output file (gzip compressed).
        label (str): Label to be added to each line.
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

            fasttext_data = {
                "text": data["text"],
                "label": get_label(data,attribs)
            }

            if rng.random() < sampling_rate:
                f_out.write(json.dumps(fasttext_data) + "\n")

    return True

@dataclass
class LabeledDatasetConfig:
    """
    Configuration class for a labeled dataset.

    Attributes:
        doc_path (str): Path to documents (i.e., gs://{BUCKET}/documents/reddit/v0/<doc_experiment>).
        attr_path (str): Path to attributes (i.e., gs://{BUCKET}/attributes/reddit/v0/<attr_experiment>).
        sampling_rate (float): Fraction of documents from the dataset to add to fastText training dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    doc_path: str
    attr_path: str
    sampling_rate: float
    seed: int

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        output_base_path (str): Base path for output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        datasets (List[LabeledDatasetConfig]): List of LabeledDatasetConfig objects from which to construct fastText training dataset.
    """
    output_base_path: str
    experiment: str
    datasets: List[LabeledDatasetConfig]

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()
    
    for dataset in cfg.datasets:
        # curry write_fasttext_lines so that we can pass it to map_files_in_directory
        @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
        def processing_func(input_file_path : str,output_file_path : str) -> bool:
            attr_file_path = rebase_file_path(dataset.doc_path,input_file_path,dataset.attr_path)
            return write_fasttext_lines(input_file_path,output_file_path,attr_file_path,dataset.sampling_rate,dataset.seed)

        # HACK: ok to keep?
        doc_path_prefix = dataset.doc_path.split('/documents')[0]
        output_path = rebase_file_path(f'{doc_path_prefix}/documents', 
                                      dataset.doc_path, 
                                      f'{cfg.output_base_path}/classifiers/{cfg.experiment}/data'
                                      )
        
        responses = map_files_in_directory(processing_func.remote, dataset.doc_path, "**/*.jsonl.gz", output_path)
        try:
            ray.get(responses)
        except Exception as e:
            print(f"Error processing {dataset.doc_path}: {e}")

if __name__ == '__main__':
    main()