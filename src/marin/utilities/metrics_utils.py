import json
import os.path
from dataclasses import dataclass
from typing import Any

import fsspec


@dataclass
class MergeConfig:
    output_path: str  # Path where we write the merged metric.json
    merge_paths: list[str]  # Paths where we read metric.jsons from and merge and write to output path


def merge(metrics_config: MergeConfig) -> dict[str, Any]:
    """
    Merges multiple metric.json files into a single metric.json file.
    """
    # Initialize the merged metrics dictionary
    merged_metrics = {}

    # Read metrics from each path and merge them
    for path in metrics_config.merge_paths:
        with fsspec.open(os.path.join(path, "metric.json"), "r") as f:
            metrics = json.load(f)
            merged_metrics.update(metrics)

    # Write the merged metrics to the output path
    with fsspec.open(os.path.join(metrics_config.output_path, "metric.json"), "w") as f:
        json.dump(merged_metrics, f)

    return merged_metrics
