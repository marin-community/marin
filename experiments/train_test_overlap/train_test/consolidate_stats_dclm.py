#!/usr/bin/env python3
"""
consolidate_stats.py: Consolidate per-shard data-overlap outputs into per-local and per-global shards for DCLM.
"""
import json
import logging
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.train_test.overlap_pipeline_dclm_sharded import dclm_sharded_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConsolidateStatsConfig:
    """Configuration for consolidating per-shard stats"""

    input_step: ExecutorStep  # the sharded-overlap pipeline step
    output_path: str  # base output path for consolidated stats
    only_stats: bool = False  # if True, skip raw ngrams consolidation and only merge stats and mappings


def consolidate_stats(cfg: ConsolidateStatsConfig) -> str:
    """Read all .SUCCESS markers under the sharded output merge overlap_stats, raw_ngrams, instance_mapping per-group."""
    base = cfg.input_step.rstrip("/")
    # Flag to control whether to only merge stats and mappings (skip raw ngrams)
    only_stats = cfg.only_stats
    output_base = cfg.output_path.rstrip("/")
    print(f"[consolidate_stats] Starting consolidation: base={base}, output_base={output_base}", flush=True)
    logger.info(f"Consolidating shards under {base} into {output_base}")

    # find completed shards
    success_pattern = f"{base}/**/*.SUCCESS"
    success_paths = fsspec_glob(success_pattern)
    if not success_paths:
        print(f"[consolidate_stats] ERROR: No .SUCCESS files found under {base}", flush=True)
        raise RuntimeError(f"No .SUCCESS files found under {base}")
    print(f"[consolidate_stats] Found {len(success_paths)} .SUCCESS markers", flush=True)

    # build nested shard grouping: global_shard -> local_shard -> [dirs]
    nested: dict[str, dict[str, list[str]]] = {}
    for success in success_paths:
        processed = success[: -len(".SUCCESS")]
        rel = processed[len(base) + 1 :]
        parts = rel.split("/")
        if len(parts) < 2:
            print(f"[consolidate_stats] Skipping unexpected path: {processed}", flush=True)
            continue
        g_shard, l_shard = parts[0], parts[1]
        nested.setdefault(g_shard, {}).setdefault(l_shard, []).append(processed)
    print(f"[consolidate_stats] Found {len(nested)} global shards", flush=True)
    # ensure top-level output directory exists
    print(f"[consolidate_stats] Ensuring output directory: {output_base}", flush=True)
    fsspec_mkdirs(output_base)

    def consolidate_group(group_name: str, dirs: list[str]) -> None:
        logger.info(f"Consolidating group {group_name} ({len(dirs)} shards)")
        print(f"[consolidate_group] group={group_name}, shards={len(dirs)}", flush=True)
        out_dir = f"{output_base}/{group_name}"
        fsspec_mkdirs(out_dir)

        # 1) merge overlap_stats.jsonl
        agg: dict[tuple, dict] = {}
        for d in dirs:
            in_stats = f"{d}/stats/overlap_stats.jsonl"
            try:
                print(f"[consolidate_group] Merging stats from {in_stats}", flush=True)
                with fsspec.open(in_stats, "rt") as f:
                    for line in f:
                        rec = json.loads(line)
                        key_spec = rec["data_overlap_stats_key"]
                        scenario = key_spec["light_scenario_key"]["scenario_spec"]["class_name"]
                        args = key_spec["light_scenario_key"]["scenario_spec"]["args"]
                        split = key_spec["light_scenario_key"]["split"]
                        n = key_spec["overlap_protocol_spec"]["n"]
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
                logger.warning(f"Missing or unreadable stats file {in_stats}, skipping")
        # write merged stats
        out_stats = f"{out_dir}/overlap_stats.jsonl"
        print(f"[consolidate_group] Writing merged stats to {out_stats}", flush=True)
        # Write the consolidated overlap statistics for this group:
        # - Sort keys to ensure deterministic ordering
        # - Convert sets of overlapping instance IDs into sorted lists
        # - Output each record as a JSON line with data_overlap_stats_key and num_instances from run_data_overlap
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
        print(f"[consolidate_group] Wrote merged stats to {out_stats}", flush=True)

        # 2) concatenate raw n-grams (skipped if only_stats flag is set)
        if not only_stats:
            out_ngrams = f"{out_dir}/raw_ngrams.jsonl"
            print(f"[consolidate_group] Writing concatenated raw n-grams to {out_ngrams}", flush=True)
            with fsspec.open(out_ngrams, "wb") as out_f:
                for d in dirs:
                    in_ngrams = f"{d}/raw_ngrams/raw_ngrams.jsonl"
                    try:
                        print(f"[consolidate_group] Adding ngrams from {in_ngrams}", flush=True)
                        with fsspec.open(in_ngrams, "rb") as f:
                            out_f.write(f.read())
                    except Exception:
                        logger.warning(f"Missing or unreadable ngrams file {in_ngrams}, skipping")
            print(f"[consolidate_group] Wrote concatenated raw n-grams to {out_ngrams}", flush=True)
        else:
            print(f"[consolidate_group] only_stats=True, skipping raw n-grams for group {group_name}", flush=True)

        # 3) merge instance mappings
        mapping: dict[str, dict[str, list[str]]] = {}
        for d in dirs:
            in_map = f"{d}/instance_mapping/instance_mapping.json"
            try:
                print(f"[consolidate_group] Merging mapping from {in_map}", flush=True)
                with fsspec.open(in_map, "rt") as f:
                    data = json.load(f)
                for inst, overlap in data.items():
                    # Merge this shard's instance-level overlap data into the consolidated mapping:
                    # Use setdefault to get an existing entry or initialize with empty overlap lists.
                    # should allow accumulating overlap keys from multiple shards without overwriting previous entries.
                    entry = mapping.setdefault(inst, {"input_overlaps": [], "reference_overlaps": []})
                    # Deduplicate: append each input overlap key only if it's not already recorded.
                    for k in overlap.get("input_overlaps", []):
                        if k not in entry["input_overlaps"]:
                            entry["input_overlaps"].append(k)
                    # Deduplicate: append each reference overlap key only if it's not already recorded.
                    for k in overlap.get("reference_overlaps", []):
                        if k not in entry["reference_overlaps"]:
                            entry["reference_overlaps"].append(k)
            except Exception:
                logger.warning(f"Missing or unreadable mapping file {in_map}, skipping")
        out_map = f"{out_dir}/instance_mapping.json"
        print(f"[consolidate_group] Writing merged instance mapping to {out_map}", flush=True)
        with fsspec.open(out_map, "wt") as f:
            json.dump(mapping, f, indent=2)
        print(f"[consolidate_group] Wrote merged instance_mapping to {out_map}", flush=True)

    # run consolidation per global and nested local shards
    for g_shard, local_shards in nested.items():
        # global-level aggregate
        all_dirs = [d for dirs in local_shards.values() for d in dirs]
        print(f"[consolidate_stats] Consolidating global shard {g_shard} with {len(all_dirs)} dirs", flush=True)
        consolidate_group(g_shard, all_dirs)
        # local-level aggregates under this global shard
        for l_shard, dirs in local_shards.items():
            print(
                f"[consolidate_stats] Consolidating local shard {l_shard} under {g_shard} with {len(dirs)} dirs",
                flush=True,
            )
            consolidate_group(f"{g_shard}/{l_shard}", dirs)

    return f"Consolidated stats written to {output_base}"


# set up ExecutorStep
consolidate_config = ConsolidateStatsConfig(
    input_step=dclm_sharded_step,
    output_path=this_output_path(),
    only_stats=False,  # set True to only merge stats and instance mappings without raw ngrams
)
consolidate_step = ExecutorStep(
    name="train_test_overlap/consolidate_stats",
    fn=lambda cfg: consolidate_stats(cfg),
    config=consolidate_config,
)

if __name__ == "__main__":
    executor_main(steps=[consolidate_step], description="Consolidate data-overlap stats across shards")
