from dataclasses import dataclass
from datetime import datetime
import json
import os
import random

import draccus
import fsspec
import ray
from google.cloud import storage

from marin.utils import fsspec_glob, rebase_file_path
from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from typing import List, Optional

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
    path: str
    dataset: str

    doc_experiment: str
    attr_experiment: str

    labels: List[str]
    sampling_rate: float
    seed: int

@dataclass
class MainConfig:
    output_path: str
    experiment: str
    
    data_cfgs: List[LabeledDatasetConfig]

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()
    
    for data_cfg in cfg.data_cfgs:
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
            print(f"Error processing: {e}")

if __name__ == '__main__':
    main()