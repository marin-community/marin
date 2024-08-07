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
import fasttext
from typing import List, Optional

def merge_shards(shard_paths, train_path, val_path, val_split, seed):
    random.seed(seed)
    with fsspec.open(train_path, "wt") as f_train, fsspec.open(val_path, "wt") as f_val:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "rt", compression = "gzip") as f_in:
                for line in f_in:
                    p = random.random()
                    if p < val_split:
                        f_val.write(line)
                    else:
                        f_train.write(line)

    return True

@dataclass
class MainConfig:
    path: str
    experiment: str

    training_args: dict
    seed: int
    val_split: float

    memory: int
    num_cpus: int

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()
    cfg.training_args['thread'] = cfg.num_cpus

    @ray.remote(memory=cfg.memory * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=cfg.num_cpus)
    def run(cfg):
        experiment_dir = f'{cfg.path}/classifiers/{cfg.experiment}'
        shard_paths = fsspec_glob(os.path.join(f'{experiment_dir}/data', "**/*.jsonl.gz"))
        merge_shards(shard_paths,"data.train","data.val",cfg.val_split,cfg.seed)

        model = fasttext.train_supervised("data.train",**cfg.training_args)
        model.save_model("model.bin")

        with fsspec.open("model.bin","rb") as f_in, \
            fsspec.open(f'{experiment_dir}/model.bin',"wb") as f_out:
                f_out.write(f_in.read())
        with fsspec.open("data.train","rb") as f_in, \
            fsspec.open(f'{experiment_dir}/data.train',"wb") as f_out:
                f_out.write(f_in.read())
        with fsspec.open("data.val","rb") as f_in, \
            fsspec.open(f'{experiment_dir}/data.val',"wb") as f_out:
                f_out.write(f_in.read())
        
        return True
    
    response = run.remote(cfg)
    try:
        ray.get(response)
    except Exception as e:
        print(f"Error processing: {e}")

if __name__ == '__main__':
    main()