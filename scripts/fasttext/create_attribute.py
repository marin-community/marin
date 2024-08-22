"""
create_attribute.py

Writes a label (e.g., for quality classifiers) as an attribute for each document in group of datasets. 
Each dataset gets its own set of labels.
"""

import draccus

import fsspec
import json
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory

from typing import List
from dataclasses import dataclass

@cached_or_construct_output(success_suffix="SUCCESS")
def write_label(input_file_path: str, output_file_path: str, label: str) -> bool:
    """
    Creates an attribute "label" from input list of labels for each document.

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

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        input_doc_path (str): Path to documents (i.e., gs://{BUCKET}/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://{BUCKET}/attributes/.../<experiment>).
        label (str): Quality classifier label to write as attribute.
    """
    input_doc_path: str
    output_attr_path: str
    experiment: str

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()

    # curry write_label so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def processing_func(input_file_path,output_file_path):
        return write_label(input_file_path,output_file_path,cfg.label)
    
    responses = map_files_in_directory(processing_func.remote, cfg.input_doc_path, "**/*.jsonl.gz", cfg.output_attr_path)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing {cfg.input_doc_path}: {e}")

if __name__ == '__main__':
    main()