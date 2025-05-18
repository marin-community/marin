#!/usr/bin/env python3
"""
Aggregate test-overlap summaries across consolidated shards.
"""
import json
import logging
import os
import re
from dataclasses import dataclass

import fsspec

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AggregateTestOverlapConfig:
    # GCS path to the consolidated directory containing subdirs of aggregated_metrics files
    consolidated_root: str
    # GCS path where aggregate summaries will be written (base will append dataset/ngram_{N})
    output_base: str
    # Which partial_overlap_spec to filter on (e.g. "binary", "jaccard", "token")
    partial_overlap_spec: str = "binary"


def aggregate_test_overlap(cfg: AggregateTestOverlapConfig) -> str:
    root = cfg.consolidated_root.rstrip("/")
    out_base = cfg.output_base.rstrip("/")
    partial = cfg.partial_overlap_spec

    # Find all aggregated_metrics JSONL files under the consolidated root
    pattern = f"{root}/**/aggregated_metrics_*.jsonl"
    metric_paths = fsspec_glob(pattern)
    if not metric_paths:
        logger.warning("No aggregated_metrics files found under %s", root)

    # summary structure: dataset -> n_val -> subset -> split -> {instance_ids:set, metrics_input_paths:dict}
    summary = {}

    for path in metric_paths:
        # Extract the n-gram size from the filename
        m = re.search(r"aggregated_metrics_(\d+)\.jsonl$", path)
        if not m:
            continue
        n_val = int(m.group(1))

        with fsspec.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                # Filter by the requested partial_overlap_spec
                if rec.get("metric_protocol_spec", {}).get("partial_overlap_spec") != partial:
                    continue

                stats_key = rec["aggregate_data_overlap_key"]["stats_key"]
                scenario = stats_key["light_scenario_key"]["scenario_spec"]
                class_name = scenario.get("class_name", "")
                dataset = class_name.split(".")[-1]
                args = scenario.get("args", {}) or {}
                subset = args.get("subset", "")
                split = stats_key["light_scenario_key"]["split"]

                inst_ids = rec.get("instance_ids", [])
                mip = rec.get("metrics_input_paths", {}) or {}

                # Initialize nested summary entries
                ds = summary.setdefault(dataset, {})
                nmap = ds.setdefault(n_val, {})
                smap = nmap.setdefault(subset, {})
                ent = smap.setdefault(split, {"instance_ids": set(), "metrics_input_paths": {}})

                # Populate instance IDs and input paths
                for inst in inst_ids:
                    ent["instance_ids"].add(inst)
                    paths = mip.get(inst, [])
                    ipaths = ent["metrics_input_paths"].setdefault(inst, set())
                    ipaths.update(paths)

    # Write out JSONL summaries under output_base/{dataset}/ngram_{N}/{partial}.jsonl
    for dataset, nmap in summary.items():
        for n_val, smap in nmap.items():
            out_dir = os.path.join(out_base, dataset, f"ngram_{n_val}")
            fsspec_mkdirs(out_dir)
            out_file = os.path.join(out_dir, f"{partial}.jsonl")
            with fsspec.open(out_file, "wt") as out_f:
                for subset, splitmap in smap.items():
                    for split, data in splitmap.items():
                        record = {
                            "subset": subset,
                            "split": split,
                            "instance_ids": sorted(data["instance_ids"]),
                            "metrics_input_paths": {
                                inst: sorted(list(paths)) for inst, paths in data["metrics_input_paths"].items()
                            },
                        }
                        out_f.write(json.dumps(record) + "\n")

    logger.info("Wrote aggregate summaries to %s", out_base)
    return "Aggregate test overlap completed!"


# Configure the step with the GCS consolidated root and an output base under this step's path
config = AggregateTestOverlapConfig(
    consolidated_root="gs://marin-us-central2/train_test_overlap/consolidated/",
    output_base=this_output_path(),
)
aggregate_step = ExecutorStep(
    name="train_test_overlap/aggregated/",
    fn=aggregate_test_overlap,
    config=config,
)

if __name__ == "__main__":
    executor_main(
        steps=[aggregate_step],
        description="Aggregate test-overlap summaries from consolidated shards",
    )
