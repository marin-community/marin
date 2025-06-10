#!/usr/bin/env python3
"""
aggregate_overlap.py

Aggregate n-gram overlap results produced by debug_sharded.py
using Marin's executor framework.
"""
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.dolma.debug_sharded import steps as dedupe_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs


@dataclass(frozen=True)
class AggregateOverlapConfig:
    # List of dedupe step ExecutorStep objects whose output_paths will be scanned
    dedupe_steps: list[ExecutorStep]
    # test dataset name (used to form attribute keys)
    dataset_name: str = "mmlu"
    # n-gram sizes to aggregate
    ngrams: tuple[int, ...] = (10, 15)
    # where to write aggregated outputs (Executor will fill this)
    output_path: str = ""


def aggregate_overlap(config: AggregateOverlapConfig):
    # use dedupe steps output paths directly
    shards = config.dedupe_steps

    for n in config.ngrams:
        attr_key = f"{config.dataset_name}_overlap_{n}"
        print(f"[DEBUG] Aggregating for n-gram {n}, attribute key={attr_key}", flush=True)
        id_to_shards: dict[str, list[str]] = defaultdict(list)
        shard_counts: dict[str, int] = {}
        total_test = 0

        for shard_dir in shards:
            print(f"[DEBUG] Processing shard output path: {shard_dir}", flush=True)
            norm = shard_dir.rstrip("/")
            # find attribute directories matching prefix attr_key + '_*'
            attr_dirs = sorted(fsspec_glob(f"{norm}/{attr_key}_*"))
            print(f"[DEBUG] Found attribute directories: {attr_dirs}", flush=True)
            if not attr_dirs:
                continue
            overlapped = set()
            for attr_dir in attr_dirs:
                print(f"[DEBUG] Scanning attribute directory: {attr_dir}", flush=True)
                # scan all JSONL files under this attribute directory
                rec_paths = fsspec_glob(f"{attr_dir}/**/*.jsonl*")
                print(f"[DEBUG] Found {len(rec_paths)} JSONL files in {attr_dir}", flush=True)
                for rec_path in rec_paths:
                    print(f"[DEBUG] Scanning file: {rec_path}", flush=True)
                    with fsspec.open(rec_path, "rt", compression="infer") as f:
                        for line in f:
                            total_test += 1
                            rec = json.loads(line)
                            # detect overlap for any attribute key with this n-gram prefix
                            for key, val in rec.get("attributes", {}).items():
                                if key.startswith(attr_key) and val:
                                    overlapped.add(rec["id"])
                                    print(
                                        f"[DEBUG] Found overlap for key={key} in file {rec_path}, id={rec['id']}",
                                        flush=True,
                                    )
                                    break
            shard_name = os.path.basename(norm)
            print(f"[DEBUG] Shard {shard_name}: total_test={total_test}, overlapped_count={len(overlapped)}", flush=True)
            shard_counts[shard_name] = len(overlapped)
            for _id in overlapped:
                # record full GCP URL of the shard, not just its basename
                id_to_shards[_id].append(norm)

        # overall metrics
        overall = len(id_to_shards)
        print(f"[DEBUG] For n-gram {n}, total_test={total_test}, total_overlapped_ids={overall}", flush=True)
        frac_all = overall / total_test if total_test else 0.0

        # write outputs under config.output_path
        base_out = config.output_path.rstrip("/")
        out_dir = os.path.join(base_out, config.dataset_name, str(n))
        fsspec_mkdirs(out_dir)
        # CSV: overall overlap fraction only
        with fsspec.open(os.path.join(out_dir, "fractions.csv"), "wt") as cf:
            writer = csv.writer(cf)
            writer.writerow(["overall_overlap_fraction"])
            writer.writerow([frac_all])
        # JSONL: mapping each ID to shards list
        with fsspec.open(os.path.join(out_dir, "id_to_shards.jsonl"), "wt") as jf:
            for _id, shards_list in id_to_shards.items():
                jf.write(json.dumps({"id": _id, "shards": shards_list}) + "\n")

        print(f"Wrote aggregation for {config.dataset_name} {n}-gram to {out_dir}")


cfg = AggregateOverlapConfig(
    dedupe_steps=dedupe_steps,
    dataset_name="mmlu",
    ngrams=(10, 15),
    output_path=this_output_path(),
)
aggregate_step = ExecutorStep(
    name=f"train_test_overlap/dolma/aggregate_overlap/{cfg.dataset_name}",
    fn=aggregate_overlap,
    config=cfg,
)
if __name__ == "__main__":

    # run dedupe steps, then aggregation
    executor_main(
        steps=[aggregate_step],
        description="Aggregate n-gram overlap across shards",
    )
