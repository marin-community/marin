#!/usr/bin/env python3
"""
Aggregate test-overlap summaries across consolidated shards.
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field

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
    # Path to the consolidated scenarios JSONL for total instance counts
    scenario_jsonl: str
    # Which partial_overlap_spec to filter on (e.g. "binary", "jaccard", "token")
    partial_overlap_spec: str = "binary"
    # Only process these n-gram sizes; empty means process all
    n_values: list[int] = field(default_factory=list)


def aggregate_test_overlap(cfg: AggregateTestOverlapConfig) -> str:
    root = cfg.consolidated_root.rstrip("/")
    out_base = cfg.output_base.rstrip("/")
    partial = cfg.partial_overlap_spec

    # Load scenario counts for total instances per dataset/subset/split
    scenario_counts: dict = {}
    # Also build mapping from instance_id to its input text and references
    instance_data_map: dict = {}
    with fsspec.open(cfg.scenario_jsonl, "rt") as sf:
        for line in sf:
            rec = json.loads(line)
            sk = rec.get("scenario_key", {})
            spec = sk.get("scenario_spec", {})
            class_name = spec.get("class_name", "")
            dataset = class_name.split(".")[-1]
            args = spec.get("args", {}) or {}
            subset = args.get("subset") or args.get("subject") or ""
            split = sk.get("split", "")
            # Count total instances
            inst_list = rec.get("instances", [])
            scenario_counts.setdefault(dataset, {}).setdefault(subset, {})[split] = len(inst_list)
            # Record input text and references for each instance
            for inst in inst_list:
                inst_id = inst.get("id")
                input_text = inst.get("input", "")
                references = inst.get("references", [])
                instance_data_map.setdefault(dataset, {}).setdefault(subset, {}).setdefault(split, {})[inst_id] = {
                    "input": input_text,
                    "references": references,
                }

    # Find all aggregated_metrics JSONL files under the consolidated root
    pattern = f"{root}/**/aggregated_metrics_*.jsonl"
    metric_paths = fsspec_glob(pattern)
    if not metric_paths:
        logger.warning("No aggregated_metrics files found under %s", root)
    # Show progress on discovered files
    total_metrics = len(metric_paths)
    print(f"Discovered {total_metrics} aggregated_metrics files under {root}", flush=True)

    # summary for partial JSONL
    summary = {}
    # separate summary stats per part
    summary_by_part = {"input": {}, "references": {}}

    for idx, path in enumerate(metric_paths, start=1):
        print(f"Processing file {idx}/{total_metrics}: {path}", flush=True)
        # Extract the n-gram size from the filename
        m = re.search(r"aggregated_metrics_(\d+)\.jsonl$", path)
        if not m:
            continue
        n_val = int(m.group(1))
        # Skip n-gram sizes not in configured list
        if cfg.n_values and n_val not in cfg.n_values:
            print(f"Skipping file {idx}/{total_metrics} for n={n_val}: not in n_values {cfg.n_values}", flush=True)
            continue

        with fsspec.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                # Filter by metric type
                if rec.get("metric_protocol_spec", {}).get("partial_overlap_spec") != partial:
                    continue
                # Determine part: input vs references
                part = rec.get("aggregate_data_overlap_key", {}).get("part")
                if part not in summary_by_part:
                    continue

                stats_key = rec["aggregate_data_overlap_key"]["stats_key"]
                scenario = stats_key["light_scenario_key"]["scenario_spec"]
                class_name = scenario.get("class_name", "")
                dataset = class_name.split(".")[-1]
                args = scenario.get("args", {}) or {}
                subset = args.get("subset") or args.get("subject") or ""
                split = stats_key["light_scenario_key"]["split"]

                inst_ids = rec.get("instance_ids", [])
                mip = rec.get("metrics_input_paths", {}) or {}

                # accumulate for partial JSONL
                ds = summary.setdefault(dataset, {})
                nmap = ds.setdefault(n_val, {})
                smap = nmap.setdefault(subset, {})
                ent = smap.setdefault(split, {"instance_ids": set(), "metrics_input_paths": {}})

                for inst in inst_ids:
                    ent["instance_ids"].add(inst)
                    paths = mip.get(inst, [])
                    ipaths = ent["metrics_input_paths"].setdefault(inst, set())
                    ipaths.update(paths)
                # accumulate for summary stats by part
                part_ds = summary_by_part[part].setdefault(dataset, {})
                part_nmap = part_ds.setdefault(n_val, {})
                part_smap = part_nmap.setdefault(subset, {})
                part_ent = part_smap.setdefault(split, set())
                for inst in inst_ids:
                    part_ent.add(inst)

        # Finished reading this metric file
        print(f"Finished processing file {idx}/{total_metrics}", flush=True)

    # Write out JSONL summaries under output_base/{dataset}/ngram_{N}/{partial}.jsonl
    for dataset, nmap in summary.items():
        print(f"Writing summaries for dataset {dataset}", flush=True)
        for n_val, smap in nmap.items():
            print(f"  n-gram size {n_val}", flush=True)
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
                            # Map each instance ID to its input text and references
                            "instance_id_mapping": {
                                inst: instance_data_map.get(dataset, {}).get(subset, {}).get(split, {}).get(inst, {})
                                for inst in data["instance_ids"]
                            },
                        }
                        out_f.write(json.dumps(record) + "\n")

            # Write separate summary stats for inputs and references
            for part in ["input", "references"]:
                stats = summary_by_part.get(part, {}).get(dataset, {}).get(n_val, {})
                summary_file = os.path.join(out_dir, f"summary_{part}.jsonl")
                print(f"  Writing summary_{part}.jsonl for n-gram {n_val}", flush=True)
                with fsspec.open(summary_file, "wt") as sum_f:
                    for subset, splitmap in stats.items():
                        # per-split stats
                        for split, inst_set in splitmap.items():
                            total = scenario_counts.get(dataset, {}).get(subset, {}).get(split, 0)
                            overlap_count = len(inst_set)
                            fraction = (overlap_count / total) if total else None
                            sum_rec = {
                                "dataset": dataset,
                                "n_val": n_val,
                                "part": part,
                                "subset": subset,
                                "split": split,
                                "total_instances": total,
                                "overlap_count": overlap_count,
                                "overlap_fraction": fraction,
                            }
                            sum_f.write(json.dumps(sum_rec) + "\n")
                        # end of summary_{part}.jsonl writing
                # After per-subset summaries, write total for this part
                total_filename = "summary_input_total.jsonl" if part == "input" else "summary_reference_total.jsonl"
                total_file = os.path.join(out_dir, total_filename)
                print(f"  Writing {total_filename} for n-gram {n_val}", flush=True)
                # Compute total_instances and overlap_count across all subsets for split 'test'
                total_instances = sum(
                    scenario_counts.get(dataset, {}).get(subset_key, {}).get("test", 0)
                    for subset_key in scenario_counts.get(dataset, {})
                )
                stats_all = summary_by_part.get(part, {}).get(dataset, {}).get(n_val, {})
                overlap_count = sum(len(stats_all.get(subset_key, {}).get("test", set())) for subset_key in stats_all)
                fraction = (overlap_count / total_instances) if total_instances else None
                with fsspec.open(total_file, "wt") as tot_f:
                    total_rec = {
                        "dataset": dataset,
                        "n_val": n_val,
                        "part": part,
                        "split": "test",
                        "total_instances": total_instances,
                        "overlap_count": overlap_count,
                        "overlap_fraction": fraction,
                    }
                    tot_f.write(json.dumps(total_rec) + "\n")

    logger.info("Wrote aggregate summaries to %s", out_base)
    return "Aggregate test overlap completed!"


# Optionally filter which n-gram sizes to process via the N_VALUES env var (e.g. "10,15")

n_values_list = []  # empty means all
n_values_list = [10, 15]
config = AggregateTestOverlapConfig(
    consolidated_root="gs://marin-us-central2/train_test_overlap/consolidated/",
    output_base=this_output_path(),
    scenario_jsonl="gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl",
    n_values=n_values_list,
)
aggregate_step = ExecutorStep(
    name="train_test_overlap/aggregated",
    fn=aggregate_test_overlap,
    config=config,
)
# Configure the step with the GCS consolidated root and an output base under this step's path
if __name__ == "__main__":
    executor_main(
        steps=[aggregate_step],
        description="Aggregate test-overlap summaries from consolidated shards",
    )
