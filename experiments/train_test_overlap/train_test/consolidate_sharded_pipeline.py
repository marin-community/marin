#!/usr/bin/env python3
"""
Generic consolidation script for any sharded overlap pipeline.

Recursively groups shard output directories (identified by .SUCCESS markers)
by their parent directories, then at each parent:
  1) merge overlap_stats.jsonl into consolidated overlap_stats.jsonl
  2) concatenate raw_ngrams
  3) merge instance_mapping.json
  4) merge aggregated_metrics.jsonl (if present), keeping metric_scores aligned with instance_ids and
  recording all metrics_input_paths

Usage:
  from this file import ConsolidateShardedConfig, consolidate_sharded
  cfg = ConsolidateShardedConfig(input_step=some_executor_step, output_path=this_output_path(), only_stats=False)
  consolidate_sharded(cfg)
Or wire as an ExecutorStep:
  consolidate_step = ExecutorStep(
      name="train_test_overlap/consolidate_sharded",
      fn=lambda cfg: consolidate_sharded(cfg),
      config=cfg,
  )
"""
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import fsspec

from marin.utils import fsspec_glob, fsspec_mkdirs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConsolidateShardedConfig:
    """Configuration for generic consolidation of sharded outputs"""

    input_step: str  # base directory containing sharded pipeline outputs
    output_path: str  # base directory where consolidated results will be written
    only_stats: bool = True  # if True, skip raw ngrams concatenation


def consolidate_sharded(cfg: ConsolidateShardedConfig) -> str:
    base = cfg.input_step.rstrip("/")
    only_stats = cfg.only_stats
    output_base = cfg.output_path.rstrip("/")
    logger.info(f"Starting recursive consolidation from {base} into {output_base}")

    # We'll iteratively collapse the deepest .SUCCESS-marked directories up toward the root
    current_input = base
    while True:
        # Find all leaf-level .SUCCESS markers under current input
        success_pattern = f"{current_input}/**/*.SUCCESS"
        success_paths = fsspec_glob(success_pattern)
        if not success_paths:
            break
        success_dirs = sorted({p[: -len(".SUCCESS")] for p in success_paths})

        # Compute relative depths to pick only the deepest directories
        rels = [os.path.relpath(d, current_input) for d in success_dirs]
        depths = [len(r.split(os.sep)) for r in rels]
        max_depth = max(depths)

        # Group by parent directory at max depth
        parent_to_children: dict[str, list[str]] = defaultdict(list)
        for d, depth in zip(success_dirs, depths, strict=False):
            if depth == max_depth:
                parent = os.path.dirname(d)
                parent_to_children[parent].append(d)
        if not parent_to_children:
            break

        # Consolidate each group of siblings under its parent
        for parent, child_dirs in parent_to_children.items():
            rel_parent = os.path.relpath(parent, current_input)
            if rel_parent == ".":
                out_dir = output_base
            else:
                out_dir = os.path.join(output_base, rel_parent)
            fsspec_mkdirs(out_dir)
            consolidate_group_generic(out_dir, child_dirs, only_stats)
            # Write a success marker at this consolidated directory
            success_marker = os.path.join(out_dir, ".SUCCESS")
            fsspec_mkdirs(os.path.dirname(success_marker))
            with fsspec.open(success_marker, "w") as f:
                f.write("")

        # If we just collapsed the root-level directories, we're done
        if len(parent_to_children) == 1 and list(parent_to_children.keys())[0] == current_input:
            break

        # Next iteration: consume consolidated output as new input
        current_input = output_base

    # Write overall .SUCCESS marker at the root output directory after consolidation
    success_marker = os.path.join(output_base, ".SUCCESS")
    # Ensure the output directory exists
    fsspec_mkdirs(output_base)
    with fsspec.open(success_marker, "w") as f:
        f.write("")
    logger.info(f"Wrote overall success marker at {success_marker}")

    return f"Consolidated shards from {base} to {output_base}"


def consolidate_group_generic(out_dir: str, dirs: list[str], only_stats: bool) -> None:
    """Merge stats, raw ngrams, instance mappings, and aggregated metrics for one group"""
    logger.info(f"Consolidating group into {out_dir} from {len(dirs)} shards")

    # 1) merge overlap_stats.jsonl
    agg_stats: dict[tuple, dict] = {}
    for d in dirs:
        in_stats = os.path.join(d, "stats", "overlap_stats.jsonl")
        try:
            with fsspec.open(in_stats, "rt") as f:
                for line in f:
                    rec = json.loads(line)
                    key_spec = rec["data_overlap_stats_key"]
                    scenario = key_spec["light_scenario_key"]["scenario_spec"]["class_name"]
                    args = key_spec["light_scenario_key"]["scenario_spec"]["args"]
                    split = key_spec["light_scenario_key"]["split"]
                    n = key_spec["overlap_protocol_spec"]["n"]
                    key = (scenario, tuple(sorted(args.items())), split, n)
                    if key not in agg_stats:
                        agg_stats[key] = {
                            "data_overlap_stats_key": rec["data_overlap_stats_key"],
                            "num_instances": rec["num_instances"],
                            "input_ids": set(rec["instance_ids_with_overlapping_input"]),
                            "ref_ids": set(rec["instance_ids_with_overlapping_reference"]),
                        }
                    else:
                        agg_stats[key]["input_ids"].update(rec["instance_ids_with_overlapping_input"])
                        agg_stats[key]["ref_ids"].update(rec["instance_ids_with_overlapping_reference"])
        except Exception:
            logger.warning(f"Skipping missing/unreadable stats file {in_stats}")
    out_stats = os.path.join(out_dir, "overlap_stats.jsonl")
    with fsspec.open(out_stats, "wt") as out_f:
        for key in sorted(agg_stats):
            data = agg_stats[key]
            out_rec = {
                "data_overlap_stats_key": data["data_overlap_stats_key"],
                "num_instances": data["num_instances"],
                "instance_ids_with_overlapping_input": sorted(data["input_ids"]),
                "instance_ids_with_overlapping_reference": sorted(data["ref_ids"]),
            }
            out_f.write(json.dumps(out_rec) + "\n")

    # 2) concatenate raw n-grams
    if not only_stats:
        out_ngrams = os.path.join(out_dir, "raw_ngrams.jsonl")
        with fsspec.open(out_ngrams, "wb") as out_f:
            for d in dirs:
                in_ngrams = os.path.join(d, "raw_ngrams", "raw_ngrams.jsonl")
                try:
                    with fsspec.open(in_ngrams, "rb") as f:
                        out_f.write(f.read())
                except Exception:
                    logger.warning(f"Skipping missing/unreadable ngrams file {in_ngrams}")

    # 3) merge instance mappings
    mapping: dict[str, dict[str, list[str]]] = {}
    for d in dirs:
        in_map = os.path.join(d, "instance_mapping", "instance_mapping.json")
        try:
            with fsspec.open(in_map, "rt") as f:
                shard_map = json.load(f)
            for inst, overlap in shard_map.items():
                entry = mapping.setdefault(inst, {"input_overlaps": [], "reference_overlaps": []})
                for k in overlap.get("input_overlaps", []):
                    if k not in entry["input_overlaps"]:
                        entry["input_overlaps"].append(k)
                for k in overlap.get("reference_overlaps", []):
                    if k not in entry["reference_overlaps"]:
                        entry["reference_overlaps"].append(k)
        except Exception:
            logger.warning(f"Skipping missing/unreadable mapping file {in_map}")
    out_map = os.path.join(out_dir, "instance_mapping.json")
    with fsspec.open(out_map, "wt") as f:
        json.dump(mapping, f, indent=2)

    # 4) merge aggregated metrics into per-n files
    metrics_agg: dict[tuple, dict] = {}
    for d in dirs:
        metrics_paths = fsspec_glob(f"{d}/aggregate_*/aggregate_*")
        for mpath in sorted(metrics_paths):
            try:
                with fsspec.open(mpath, "rt") as mf:
                    for line in mf:
                        rec = json.loads(line)
                        key_spec = rec["aggregate_data_overlap_key"]["stats_key"]
                        scenario = key_spec["light_scenario_key"]["scenario_spec"]["class_name"]
                        args = key_spec["light_scenario_key"]["scenario_spec"]["args"]
                        split = key_spec["light_scenario_key"]["split"]
                        n = key_spec["overlap_protocol_spec"]["n"]
                        part = rec["aggregate_data_overlap_key"]["part"]
                        protocol = rec["metric_protocol_spec"]["partial_overlap_spec"]
                        key = (scenario, tuple(sorted(args.items())), split, n, part, protocol)
                        if key not in metrics_agg:
                            metrics_agg[key] = {
                                "aggregate_data_overlap_key": rec["aggregate_data_overlap_key"],
                                "metric_protocol_spec": rec["metric_protocol_spec"],
                                "instance_ids": [],
                                "metric_scores": [],
                                "metrics_input_paths": {},
                            }
                        for inst, score in zip(rec.get("instance_ids", []), rec.get("metric_scores", []), strict=False):
                            metrics_agg[key]["instance_ids"].append(inst)
                            metrics_agg[key]["metric_scores"].append(score)
                            metrics_agg[key]["metrics_input_paths"].setdefault(inst, set()).add(
                                rec.get("metrics_input_path")
                            )
            except Exception:
                logger.warning(f"Skipping missing/unreadable metrics file {mpath}")

    # write one aggregated_metrics_{n}.jsonl per n
    # collect all n values
    ns = sorted({key[3] for key in metrics_agg.keys()})
    for n_val in ns:
        out_file = os.path.join(out_dir, f"aggregated_metrics_{n_val}.jsonl")
        with fsspec.open(out_file, "wt") as out_f:
            for key in sorted(metrics_agg):
                if key[3] != n_val:
                    continue
                entry = metrics_agg[key]
                mip = {inst: sorted(list(paths)) for inst, paths in entry["metrics_input_paths"].items()}
                out_rec = {
                    "aggregate_data_overlap_key": entry["aggregate_data_overlap_key"],
                    "metric_protocol_spec": entry["metric_protocol_spec"],
                    "instance_ids": entry["instance_ids"],
                    "metric_scores": entry["metric_scores"],
                    "metrics_input_paths": mip,
                }
                out_f.write(json.dumps(out_rec) + "\n")
