"""
Run overlapping shards with simple backpressure.
"""

import os
from dataclasses import dataclass, field

import ray

from marin.core.runtime import simple_backpressure
from marin.utils import fsspec_glob
from train_test_overlap.run_data_overlap import DataOverlapPipelineConfig, run_data_overlap


@dataclass
class ShardedOverlapConfig:
    # Base directory (or GCS/FS path) containing JSONL shards
    base_input_dir: str
    # Scenario JSONL path or directory
    scenario_data: str
    # Base output path under which each shard's outputs will be written
    output_base: str
    # List of n-gram sizes to compute
    N: list[int] = field(default_factory=lambda: [5, 9, 13])
    # Number of processes hint (not currently used inside run_data_overlap)
    processes: int = 1
    # Max number of run_data_overlap tasks in flight at once
    max_in_flight: int = 64


@ray.remote
def run_all_shards(cfg: ShardedOverlapConfig) -> str:
    """
    Discover all compressed JSONL shards under cfg.base_input_dir and launch
    one run_data_overlap task per shard, up to cfg.max_in_flight in parallel.
    """
    # Patterns matching compressed JSONL files
    input_patterns = [
        "**/*.jsonl.gz",
        "**/*.jsonl.zst",
        "**/*.jsonl.gs",
        "**/*.json.gz",
        "**/*.json.zst",
        "**/*.jsonl",
    ]
    # Discover all matching files
    all_files = set()
    for patt in input_patterns:
        pattern = os.path.join(cfg.base_input_dir.rstrip("/"), patt)
        matches = fsspec_glob(pattern)
        all_files.update(matches)
    files = sorted(all_files)

    def make_task(file_path: str) -> DataOverlapPipelineConfig:
        # Preserve directory structure relative to base_input_dir
        prefix = cfg.base_input_dir.rstrip("/") + "/"
        if file_path.startswith(prefix):
            rel_path = file_path[len(prefix) :]
        else:
            rel_path = os.path.basename(file_path)
        out_path = os.path.join(cfg.output_base, rel_path)
        return DataOverlapPipelineConfig(
            input_data=file_path,
            scenario_data=cfg.scenario_data,
            output_path=out_path,
            N=cfg.N,
            processes=cfg.processes,
        )

    # Generator of arguments for each Ray task
    task_generator = ((make_task(f),) for f in files)

    # Launch tasks with simple backpressure
    for ref in simple_backpressure(
        run_data_overlap,
        task_generator,
        max_in_flight=cfg.max_in_flight,
        fetch_local=True,
    ):
        ray.get(ref)

    return "Sharded overlap pipeline completed!"
