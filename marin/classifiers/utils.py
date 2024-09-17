"""
utils.py

Utility functions for building quality classifiers.
"""

from typing import List, Optional, Callable
import logging

import fsspec
import numpy as np
import json
import ray

from marin.utils import rebase_file_path
from marin.core.runtime import cached_or_construct_output, map_files_in_directory

@cached_or_construct_output(success_suffix="SUCCESS")
def write_label_attribute(input_file_path: str, output_file_path: str, label: str) -> bool:
    """
    Creates an attribute "label" from input label for each document.

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output attribute JSONL file (gzip compressed).
        label (str): Quality classifier label.

    Returns:
        bool: True if the process is successful.
    """
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
        for line in f_in:
            json_obj = json.loads(line)
            attributes = {"label": label}
            f_out.write(json.dumps({"id": json_obj["id"],
                                    "source": json_obj["source"],
                                    "attributes": attributes
                                    }) + "\n")

    return True

@cached_or_construct_output(success_suffix="SUCCESS")
def write_examples(
        input_file_path: str, 
        output_file_path: str, 
        attr_file_path: str, 
        sampling_rate: float, 
        seed: int, 
        get_label: Callable[[dict,dict], str]
    ) -> bool:
    """
    Writes training examples to an output file.
    Only a fraction of the examples, determined by the sampling rate, are written to the output file (eg, to control size 
    of training dataset and/or weight different domains).

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output file (gzip compressed).
        attr_file_path (str): Path to the attribute JSONL file (gzip compressed).
        sampling_rate (float): Fraction of lines to be written to the output file.
        seed (int): Seed for random number generator to ensure reproducibility.
        get_label (Callable[[dict,dict], str]): Function to extract label from documents and attributes.

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(attr_file_path, "rt", compression="gzip") as f_attr, \
                fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
        for input_line, attr_line in zip(f_in, f_attr):
            if rng.random() > sampling_rate:
                continue

            data = json.loads(input_line)
            attribs = json.loads(attr_line)

            example = {
                "text": data["text"],
                "label": get_label(data,attribs)
            }

            f_out.write(json.dumps(example) + "\n")

    return True

def merge_shards_and_split(
        shard_paths: List[str], 
        train_path: str, 
        val_path: str, 
        val_split: float, 
        seed: int, 
        format_example: Callable[[dict],str]
    ) -> bool:
    """
    Merges multiple shard files into training and validation datasets.

    Args:
        shard_paths (List[str]): List of paths to shard files.
        train_path (str): Path to the output training dataset file.
        val_path (str): Path to the output validation dataset file.
        val_split (float): Fraction of data to be used for validation.
        seed (int): Seed for random number generator to ensure reproducibility.
        format_example (Callable[[dict],str]): Function to format example into correct training format (e.g., BERT, fastText, etc.).

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(train_path, "wt") as f_train, fsspec.open(val_path, "wt") as f_val:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "rt", compression = "gzip") as f_in:
                for input_line in f_in:
                    data = json.loads(input_line)
                    output_line = format_example(data)

                    if rng.random() < val_split:
                        f_val.write(output_line + "\n")
                    else:
                        f_train.write(output_line + "\n")

    return True

def create_label_attribute(input_doc_path: str, output_attr_path: str, label: str) -> bool:
    """
    Create attribute for quality classifier label.

    Args:
        input_doc_path (str): Path to documents (i.e., gs://{BUCKET}/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://{BUCKET}/attributes/.../<experiment>).
        label (str): Quality classifier label to write as attribute.
    
    Returns:
        bool: True if the process is successful.
    """
    # curry write_label so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def processing_func(input_file_path,output_file_path):
        return write_label_attribute(input_file_path,output_file_path,label)
    
    responses = map_files_in_directory(processing_func.remote, input_doc_path, "**/*.jsonl.gz", output_attr_path)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing {input_doc_path}: {e}")
    
    return True

def attributes_to_dataset(
        experiment_path: str, 
        doc_path: str, 
        attr_path: str, 
        sampling_rate: float, 
        seed: int, 
        get_label: Callable[[dict,dict], str] = lambda data,attribs : attribs["attributes"]["label"]
    ) -> bool:
    """
    Converts documents and attributes to quality classifier training data (text,label) pairs.

    Args:
        experiment_path (str): Path for output data (i.e., gs://{BUCKET}/classifiers/$EXPERIMENT).
        doc_path (str): Path to documents (i.e., gs://{BUCKET}/documents/reddit/v0/<doc_experiment>).
        attr_path (str): Path to attributes (i.e., gs://{BUCKET}/attributes/reddit/v0/<attr_experiment>).
        sampling_rate (float): Fraction of documents from the dataset to add to fastText training dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
        get_label (Callable[[dict,dict], str]): Function to extract label from documents and attributes. Defaults to get_label.
    
    Returns:
        bool: True if the process is successful.
    """
    logger = logging.getLogger("ray")

    # curry write_fasttext_lines so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def processing_func(input_file_path : str,output_file_path : str) -> bool:
        attr_file_path = rebase_file_path(doc_path,input_file_path,attr_path)
        return write_examples(input_file_path,output_file_path,attr_file_path,sampling_rate,seed,get_label)

    # HACK: ok to keep?
    doc_path_prefix = doc_path.split('/documents')[0]
    output_path = rebase_file_path(f'{doc_path_prefix}/documents', 
                                    doc_path, 
                                    f'{experiment_path}/data'
                                    )
    
    responses = map_files_in_directory(processing_func.remote, doc_path, "**/*.jsonl.gz", output_path)
    try:
        ray.get(responses)
    except Exception as e:
        logger.exception(f"Error processing {doc_path}: {e}")
        raise
    
    return True

def shuffle(input_file_path: str, output_file_path: str, seed: int) -> bool:
    """
    Shuffles the lines of a file.

    Args:
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to the output file.
        seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(input_file_path, "rt", compression = "infer") as f_in:
        lines = f_in.readlines()
    rng.shuffle(lines)
    with fsspec.open(output_file_path, "wt", compression = "infer") as f_out:
        f_out.writelines(lines)

    return True