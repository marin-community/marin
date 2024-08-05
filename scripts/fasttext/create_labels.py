import draccus

import fsspec
import json
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory

from typing import List
from dataclasses import dataclass
from marin.utils import rebase_file_path

@cached_or_construct_output(success_suffix="SUCCESS")
def write_labels(input_file_path, output_file_path, labels):
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
        for line in f_in:
            json_obj = json.loads(line)
            attributes = {}
            attributes["quality-labels"] = labels
            f_out.write(json.dumps({"id": json_obj["id"],
                                    "source": json_obj["source"],
                                    "attributes": attributes
                                    }) + "\n")

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
            return write_labels(input_file_path,output_file_path,data_cfg.labels)

        output_dir = rebase_file_path(f'{data_cfg.path}/documents/{data_cfg.experiment}', 
                                      f'{data_cfg.path}/documents/{data_cfg.experiment}/{data_cfg.dataset}', 
                                      f'{cfg.output_path}/attributes/{cfg.experiment}'
                                      )
        responses = map_files_in_directory(processing_func.remote, f'{data_cfg.path}/documents/{data_cfg.experiment}/{data_cfg.dataset}', "**/*.jsonl.gz", output_dir)
        try:
            ray.get(responses)
        except Exception as e:
            print(f"Error processing: {e}")

if __name__ == '__main__':
    main()