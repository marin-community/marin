# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Take bounded, reproducible row-group samples from pinned HF Parquet shards."""

import argparse
import base64
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from experiments.post_training.task_curation_partitions import assign_partitions
from experiments.post_training.task_curation_prefix_sampling import READ_BLOCK_BYTES, sample_prefix
from experiments.post_training.tasktrove.taskbinary import read_task_binary

TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
NEMO_REVISION = "9643c8103d7bfbc2d7fc4d15991d6739c612ff58"
CONFIGS = {
    "nl2bash": "DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
    "taco": "laion__exp_rpt_taco-v2",
    "codeforces": "laion__codeforces-v3",
    "unitsyn": "DCAgent__exp_rpt_unitsyn-python-v4",
    "calendar": "laion__nemotron-gym-agent-calendar-v2",
    "reasoning_gym": "laion__nemotron-gym-reasoning-gym-v2",
    "all_puzzles": "laion__all-puzzles-v2",
    "knowledge_openqa": "laion__nemotron-gym-knowledge-openqa-v4",
    "science_openqa": "laion__nemotron-gym-science-so-openq-v3",
}
NEXT_CONFIGS = {
    "code_contests": "DCAgent__code-contests-noblock",
    "codenet": "laion__exp_rpt_codenet-python-v4",
    "math_openreasoning": "laion__nemotron-gym-math-openmathreasoning-v2",
    "advanced_calculations": "laion__nemotron-gym-math-advanced-calculations-v4",
    "knowledge_mcqa": "laion__nemotron-gym-knowledge-mcqa-v2",
    "web_search_mcqa": "laion__nemotron-gym-knowledge-web-search-mcqa-v2",
    "qa_abstention": "laion__nemotron-gym-qa-abstention-v4",
    "arc_transductive": "laion__nemotron-gym-arc-agi-transductive-v3",
    "arc_inductive": "laion__nemotron-gym-arc-agi-python-inductive-v2",
    "indirect_injection": "laion__nemotron-gym-agentic-indirect-prompt-injection-v3",
}
SOURCE_CONFIGS = CONFIGS | NEXT_CONFIGS
MAX_SAMPLE_BYTES = 64 * 1024 * 1024


def sample_source(name: str, output: Path, count: int, seed: int, nemo_shard: str) -> dict:
    """Sample across eight row groups, retaining unsupported rows and all archive files."""
    if name == "nemo_actions":
        return sample_nemo(output, count, seed, nemo_shard)
    dataset = "open-thoughts/TaskTrove"
    revision = TASKTROVE_REVISION
    shard = f"{SOURCE_CONFIGS[name]}/tasks.parquet"
    randomizer = random.Random(f"{seed}:{name}")
    rows = []
    with HfFileSystem().open(f"datasets/{dataset}@{revision}/{shard}", block_size=READ_BLOCK_BYTES) as stream:
        parquet = pq.ParquetFile(stream)
        groups = randomizer.sample(range(parquet.num_row_groups), min(8, parquet.num_row_groups))
        population = sum(parquet.metadata.row_group(group).num_rows for group in groups)
        while population < count and len(groups) < parquet.num_row_groups:
            group = randomizer.choice([i for i in range(parquet.num_row_groups) if i not in groups])
            groups.append(group)
            population += parquet.metadata.row_group(group).num_rows
        decoded_bytes = sum(parquet.metadata.row_group(group).total_byte_size for group in groups)
        if decoded_bytes > MAX_SAMPLE_BYTES:
            return sample_prefix(
                name,
                output,
                count,
                seed,
                config=SOURCE_CONFIGS[name],
                revision=revision,
                max_transfer_bytes=MAX_SAMPLE_BYTES,
            )
        offsets = [0]
        for group in range(parquet.num_row_groups):
            offsets.append(offsets[-1] + parquet.metadata.row_group(group).num_rows)
        selected_rows = randomizer.sample(
            [(group, index) for group in groups for index in range(parquet.metadata.row_group(group).num_rows)], count
        )
        for group in groups:
            candidates = parquet.read_row_group(group).to_pylist()
            selected = [index for selected_group, index in selected_rows if selected_group == group]
            for index in selected:
                raw = candidates[index]
                files = read_task_binary(raw["task_binary"]).files
                row = {
                    "path": raw["path"],
                    "instruction": files["instruction.md"].decode(),
                    "files": {path: base64.b64encode(data).decode() for path, data in files.items()},
                    "archive_sha256": hashlib.sha256(raw["task_binary"]).hexdigest(),
                }
                if "tests/verifier_data.json" in files:
                    row["verifier_data"] = json.loads(files["tests/verifier_data.json"])
                row["sample_index"] = offsets[group] + index
                row["sample_row_group"] = group
                # Identical public problems stay in one partition even if archived twice.
                row["sample_group"] = hashlib.sha256(row["instruction"].encode()).hexdigest()
                rows.append(row)
        source_count = parquet.metadata.num_rows
    if len(rows) != count:
        raise ValueError(f"{name}: sampled {len(rows)}, expected {count}; row groups are too uneven")
    assign_partitions(rows, randomizer, "sample_index")
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / "sample.jsonl").write_text(snapshot)
    manifest = {
        "name": name,
        "dataset": dataset,
        "revision": revision,
        "shard": shard,
        "source_rows": source_count,
        "sample_rows": len(rows),
        "seed": seed,
        "row_groups": groups,
        "selected_group_decoded_bytes": decoded_bytes,
        "sampling": (
            "Uniform selected row groups, then uniform rows within their combined pool; not a uniform population sample"
        ),
        "grouping": "SHA256 of public instruction",
        "partitions": {
            partition: sum(row["sample_partition"] == partition for row in rows)
            for partition in ("development", "holdout")
        },
        "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
    }
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    return manifest


def sample_nemo(output: Path, count: int, seed: int, source_file: str) -> dict:
    """Sample complete JSONL records from eight bounded byte ranges, grouped by trajectory."""
    dataset = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
    path = f"datasets/{dataset}@{NEMO_REVISION}/{source_file}"
    filesystem = HfFileSystem()
    size = filesystem.info(path)["size"]
    randomizer = random.Random(f"{seed}:nemo_actions")
    chunk_size = 1024 * 1024
    offsets = sorted(randomizer.sample(range(size - chunk_size), 8))
    candidates = {}
    with filesystem.open(path, block_size=READ_BLOCK_BYTES) as stream:
        for offset in offsets:
            stream.seek(offset)
            chunk = stream.read(chunk_size)
            first, last = chunk.find(b"\n") + 1, chunk.rfind(b"\n")
            cursor = offset + first
            for line in chunk[first:last].splitlines(keepends=True):
                row = json.loads(line)
                row["sample_byte_offset"] = cursor
                row["sample_group"] = str(row["trajectory_id"])
                candidates[cursor] = row
                cursor += len(line)
    rows = randomizer.sample(list(candidates.values()), count)
    assign_partitions(rows, randomizer, "sample_byte_offset")
    directory = output / "nemo_actions"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / "sample.jsonl").write_text(snapshot)
    manifest = {
        "name": "nemo_actions",
        "dataset": dataset,
        "revision": NEMO_REVISION,
        "file": source_file,
        "source_bytes": size,
        "sample_rows": len(rows),
        "seed": seed,
        "byte_ranges": offsets,
        "read_budget_bytes": chunk_size * len(offsets),
        "candidate_rows": len(candidates),
        "sampling": "Eight random 1 MiB windows, then uniform complete records; length-biased, not population-uniform",
        "grouping": "trajectory_id",
        "unique_groups": len({row["sample_group"] for row in rows}),
        "partitions": {
            partition: sum(row["sample_partition"] == partition for row in rows)
            for partition in ("development", "holdout")
        },
        "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
    }
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--nemo-shard")
    parser.add_argument("--source", action="append", choices=(*SOURCE_CONFIGS, "nemo_actions"))
    args = parser.parse_args()
    names = args.source or [*CONFIGS, "nemo_actions"]
    if "nemo_actions" in names and args.nemo_shard is None:
        parser.error("--nemo-shard is required when sampling nemo_actions")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(sample_source, name, args.output, args.count, args.seed, args.nemo_shard) for name in names
        ]
        manifests = [future.result() for future in futures]
    (args.output / "manifest.json").write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
