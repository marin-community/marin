#!/usr/bin/env python3
"""
consolidate_stats_only.py: Consolidate overlap_stats.jsonl and instance_mapping.json across shards.
"""
import json
import logging
import os
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.train_test.overlap_pipeline_dclm_sharded import dclm_sharded_step
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConsolidateStatsOnlyConfig:
    """Configuration for consolidating only stats and mappings."""

    input_path: str  # base path of sharded outputs (InputName -> string)
    output_path: str  # where to write consolidated outputs (OutputName -> string)


def consolidate_stats_only(cfg: ConsolidateStatsOnlyConfig) -> str:
    base = cfg.input_path.rstrip("/")
    output_base = cfg.output_path.rstrip("/")
    print(f"[consolidate_stats_only] Starting: base={base}, output_base={output_base}", flush=True)

    success_pattern = f"{base}/**/*.SUCCESS"
    success_paths = fsspec_glob(success_pattern)
    if not success_paths:
        print(f"[consolidate_stats_only] ERROR: no .SUCCESS under {base}", flush=True)
        raise RuntimeError(f"No .SUCCESS files found under {base}")
    print(f"[consolidate_stats_only] Found {len(success_paths)} .SUCCESS markers", flush=True)

    # Build nested grouping: global_shard -> local_shard -> list of shard dirs
    nested: dict[str, dict[str, list[str]]] = {}
    for success in success_paths:
        processed = success[: -len(".SUCCESS")]
        rel = processed[len(base) + 1 :]
        parts = rel.split("/")
        if len(parts) < 2:
            print(f"[consolidate_stats_only] Skipping unexpected: {processed}", flush=True)
            continue
        g_shard, l_shard = parts[0], parts[1]
        nested.setdefault(g_shard, {}).setdefault(l_shard, []).append(processed)
    print(f"[consolidate_stats_only] Nested grouping: {len(nested)} global shards", flush=True)

    # Ensure output directory
    print(f"[consolidate_stats_only] Ensuring output directory: {output_base}", flush=True)
    fsspec_mkdirs(output_base)

    def merge_group(name: str, dirs: list[str]) -> None:
        print(f"[merge_group] name={name}, shards={len(dirs)}", flush=True)
        # Aggregate overlap_stats.jsonl
        agg: dict[tuple, dict] = {}
        for d in dirs:
            stats_file = os.path.join(d, "stats", "overlap_stats.jsonl")
            try:
                print(f"[merge_group] Reading stats: {stats_file}", flush=True)
                with fsspec.open(stats_file, "rt") as f:
                    for line in f:
                        rec = json.loads(line)
                        ks = rec["data_overlap_stats_key"]
                        scenario = ks["light_scenario_key"]["scenario_spec"]["class_name"]
                        args = ks["light_scenario_key"]["scenario_spec"]["args"]
                        split = ks["light_scenario_key"]["split"]
                        n = ks["overlap_protocol_spec"]["n"]
                        key = (scenario, tuple(sorted(args.items())), split, n)
                        if key not in agg:
                            agg[key] = {
                                "data_overlap_stats_key": rec["data_overlap_stats_key"],
                                "num_instances": rec["num_instances"],
                                "input_ids": set(rec["instance_ids_with_overlapping_input"]),
                                "ref_ids": set(rec["instance_ids_with_overlapping_reference"]),
                            }
                        else:
                            agg[key]["input_ids"].update(rec["instance_ids_with_overlapping_input"])
                            agg[key]["ref_ids"].update(rec["instance_ids_with_overlapping_reference"])
            except Exception:
                print(f"[merge_group] WARNING: cannot read stats file {stats_file}", flush=True)
        # Write merged stats
        out_dir = os.path.join(output_base, name)
        fsspec_mkdirs(out_dir)
        out_stats = os.path.join(out_dir, "overlap_stats.jsonl")
        print(f"[merge_group] Writing merged stats to {out_stats}", flush=True)
        with fsspec.open(out_stats, "wt") as out_f:
            for key in sorted(agg.keys()):
                data = agg[key]
                out_rec = {
                    "data_overlap_stats_key": data["data_overlap_stats_key"],
                    "num_instances": data["num_instances"],
                    "instance_ids_with_overlapping_input": sorted(data["input_ids"]),
                    "instance_ids_with_overlapping_reference": sorted(data["ref_ids"]),
                }
                out_f.write(json.dumps(out_rec) + "\n")

        # Merge instance_mapping.json
        mapping: dict[str, dict[str, list[str]]] = {}
        for d in dirs:
            map_file = os.path.join(d, "instance_mapping", "instance_mapping.json")
            try:
                print(f"[merge_group] Reading mapping: {map_file}", flush=True)
                with fsspec.open(map_file, "rt") as f:
                    data = json.load(f)
                for inst_id, overlaps in data.items():
                    entry = mapping.setdefault(inst_id, {"input_overlaps": [], "reference_overlaps": []})
                    for k in overlaps.get("input_overlaps", []):
                        if k not in entry["input_overlaps"]:
                            entry["input_overlaps"].append(k)
                    for k in overlaps.get("reference_overlaps", []):
                        if k not in entry["reference_overlaps"]:
                            entry["reference_overlaps"].append(k)
            except Exception:
                print(f"[merge_group] WARNING: cannot read mapping file {map_file}", flush=True)
        out_map = os.path.join(out_dir, "instance_mapping.json")
        print(f"[merge_group] Writing merged mapping to {out_map}", flush=True)
        with fsspec.open(out_map, "wt") as f:
            json.dump(mapping, f, indent=2)

    # Consolidate per-global and per-local
    for g_shard, locals in nested.items():
        # Global aggregate
        all_dirs = [d for group in locals.values() for d in group]
        merge_group(g_shard, all_dirs)
        # Local under global
        for l_shard, dirs in locals.items():
            merge_group(f"{g_shard}/{l_shard}", dirs)

    return f"Consolidation only written to {output_base}"


# Set up ExecutorStep
consolidate_config = ConsolidateStatsOnlyConfig(
    input_path=output_path_of(dclm_sharded_step),
    output_path=this_output_path(),
)
consolidate_step = ExecutorStep(
    name="train_test_overlap/consolidate_stats_only",
    fn=lambda cfg: consolidate_stats_only(cfg),
    config=consolidate_config,
)

if __name__ == "__main__":
    executor_main(steps=[consolidate_step], description="Consolidate stats and mappings only")
