import argparse
import json
import os

import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory

from typing import List
from marin.utils import fsspec_glob, rebase_file_path
from dataclasses import dataclass

@ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS") # We use this decorator to make this function idempotent
def write_fasttext_data(input_file_path, output_file_path, attr_file_path):
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(output_file_path, "wt", compression="gzip") as f_out, \
                fsspec.open(attr_file_path, "wt", compression="gzip") as f_attr:
        for input_line,attr_line in zip(f_in,f_attr):
            json_obj = json.loads(input_line)
            attr_obj = json.loads(attr_line)

            text = json_obj["text"].replace("\n"," ")
            labels = ''.join([f" __label__{label}" for label in attr_obj["attributes"]["quality-labels"]])
            
            line = labels + " " + text + "\n"
            f_out.write(line)

    return True

@dataclass
class LabeledDatasetConfig:
    path: str
    experiment: str
    dataset: str

    labels: List[str]

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
            attr_file_path = rebase_file_path(f'{data_cfg.path}/attributes/{data_cfg.experiment}',input_file_path,f'{cfg.path}/classifiers/{cfg.experiment}/data')
            return write_fasttext_data(input_file_path,output_file_path,attr_file_path)

        output_dir = rebase_file_path(f'{data_cfg.path}/attributes/{data_cfg.experiment}', 
                                      f'{data_cfg.path}/attributes/{data_cfg.experiment}/{data_cfg.dataset}', 
                                      f'{cfg.output_path}/classifiers/{cfg.experiment}/data'
                                      )
        
        responses = map_files_in_directory(processing_func.remote, f'{data_cfg.path}/attributes/{data_cfg.experiment}/{data_cfg.dataset}', "**/*.jsonl.gz", output_dir)
        try:
            ray.get(responses)
        except Exception as e:
            print(f"Error processing: {e}")
    
    with fsspec.open(f'{cfg.output_path}/classifiers/{cfg.experiment}/data/train.data',"wt") as f_out:
        for file in fsspec_glob(os.path.join(f'{cfg.output_path}/classifiers/{cfg.experiment}/data', "**/*.jsonl.gz")):
            with fsspec.open(file,"rt",compression="gzip") as f_in:
                for line in f_in:
                    f_out.write(line)

if __name__ == '__main__':
    main()