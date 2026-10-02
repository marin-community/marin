# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded acquisition for direct preference, instruction, and generated sources."""

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path

import numpy as np
import requests

from experiments.post_training.task_curation_direct_sampling import DirectSource, SourceFormat, sample_direct
from experiments.post_training.task_curation_partitions import assign_partitions

NAMES = (
    "hh_harmless_base",
    "hh_helpful_base",
    "hh_helpful_online",
    "hh_helpful_rejection_sampled",
    "kto_mix",
    "nemotron_if",
    "rlvr_ifeval",
    "reasoning_gym_generated",
)
TRANSFER_LIMIT = 16 * 1024 * 1024
BLOCK_BYTES = 65536
GENERATION_SEED = 42


def write_snapshot(name: str, output: Path, rows: list[dict], manifest: dict, seed: int) -> dict:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        row["sample_index"] = index
        row["sample_record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        row["sample_group"] = row["sample_record_sha256"]
    assign_partitions(rows, random.Random(f"{seed}:{name}"), "sample_index")
    snapshot = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / "sample.jsonl").write_text(snapshot)
    manifest.update(
        name=name, sample_rows=len(rows), seed=seed, snapshot_sha256=hashlib.sha256(snapshot.encode()).hexdigest()
    )
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def sample_hh(leaf, output: Path, count: int, seed: int) -> dict:
    path = f"{leaf.CONFIG}/train.jsonl.gz"
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buffer, rows, transferred = b"", [], 0
    url = f"https://huggingface.co/datasets/{leaf.DATASET}/resolve/{leaf.REVISION}/{path}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        for chunk in response.iter_content(BLOCK_BYTES):
            transferred += len(chunk)
            if transferred > TRANSFER_LIMIT:
                raise ValueError("HH source exceeded bounded gzip transfer budget")
            buffer += decompressor.decompress(chunk)
            while b"\n" in buffer and len(rows) < count:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    rows.append(json.loads(line))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"HH selected {len(rows)} rows, expected {count}")
    return write_snapshot(
        leaf.recipe(Path()).name,
        output,
        rows,
        {
            "dataset": leaf.DATASET,
            "revision": leaf.REVISION,
            "config": leaf.CONFIG,
            "split": "train",
            "path": path,
            "transferred_bytes": transferred,
            "transfer_budget_bytes": TRANSFER_LIMIT,
            "sampling": "Ordered selected-component gzip prefix; not population-uniform",
        },
        seed,
    )


def source_json_value(value):
    """Preserve NumPy scalars using the selected SkyRL loader's serialization contract."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Reasoning Gym metadata type {type(value).__name__} is not JSON serializable")


def generate_rows(count: int, seed: int, revision: str) -> dict:
    """Run inside the isolated worker whose import path identifies the acquired pinned code."""
    reasoning = importlib.import_module("reasoning_gym")
    factory = importlib.import_module("reasoning_gym.factory")
    registry = sorted(factory.DATASETS)
    tasks = random.Random(seed).sample(registry, min(count, len(registry)))
    rows_per_task = math.ceil(count / len(tasks))
    rows, controls = [], []
    for task_index, name in enumerate(tasks):
        dataset = reasoning.create_dataset(name, size=rows_per_task, seed=GENERATION_SEED + task_index)
        scorer = reasoning.get_score_answer_fn(name)
        for index in range(rows_per_task):
            if len(rows) == count:
                break
            entry = json.loads(json.dumps(dataset[index], default=source_json_value))
            positive = float(scorer(entry["answer"], entry))
            negative = float(scorer("definitely wrong", entry))
            recorded = {
                "generator_revision": revision,
                "positive": {"candidate": entry["answer"], "reward": positive},
                "negative": {"candidate": "definitely wrong", "reward": negative},
                "execution": "Acquisition-time pinned get_score_answer_fn(task); not a current bound runtime",
            }
            rows.append(
                {
                    "entry": entry,
                    "generation": {
                        "task": name,
                        "seed": GENERATION_SEED + task_index,
                        "index": index,
                        "config": dataclasses.asdict(dataset.config),
                    },
                    "recorded_pinned_generator_controls": recorded,
                }
            )
            controls.append({"task": name, **recorded})
    return {
        "rows": rows,
        "registry_tasks": registry,
        "selected_tasks": tasks,
        "rows_per_task": rows_per_task,
        "generation_seed": GENERATION_SEED,
        "generator_module": reasoning.__file__,
        "recorded_controls": controls,
    }


def sample_generator(leaf, output: Path, count: int, seed: int) -> dict:
    directory = output / leaf.recipe(Path()).name
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "generator.tar.gz"
    transferred = 0
    url = f"https://api.github.com/repos/{leaf.DATASET}/tarball/{leaf.REVISION}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with archive.open("wb") as destination:
            for chunk in response.iter_content(BLOCK_BYTES):
                transferred += len(chunk)
                if transferred > TRANSFER_LIMIT:
                    raise ValueError("Pinned generator archive exceeded transfer budget")
                destination.write(chunk)
    extracted = directory / "generator"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as source:
        root_name = source.getmembers()[0].name.split("/", 1)[0]
        source.extractall(extracted, filter="data")
    generator_root = (extracted / root_name).resolve()
    environment = {
        **os.environ,
        "PYTHONPATH": str(generator_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "MPLCONFIGDIR": str(directory / "matplotlib"),
    }
    worker = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--generate",
            "--count",
            str(count),
            "--seed",
            str(seed),
            "--revision",
            leaf.REVISION,
        ],
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    generated = json.loads(worker.stdout)
    rows = generated.pop("rows")
    manifest = {
        "dataset": leaf.DATASET,
        "revision": leaf.REVISION,
        "config": "generated",
        "split": "generated",
        "transferred_bytes": transferred,
        "transfer_budget_bytes": TRANSFER_LIMIT,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "sampling": "Seeded task-name sample without replacement from pinned sorted registry; " "source-default configs",
        "source_loader": f"MarinSkyRL@{leaf.VERIFIER_REVISION}:" "infra/rl_data/sources.py:generate_reasoning_gym_rows",
        **generated,
    }
    return write_snapshot(leaf.recipe(Path()).name, output, rows, manifest, seed)


def sample_source(name: str, output: Path, count: int, seed: int) -> dict:
    """Acquire one exact pinned source packet; contributor KTO selectors are intentionally absent."""
    if name not in NAMES or count <= 0:
        raise ValueError(f"Unknown source or nonpositive count: {name!r}, {count}")
    leaf = importlib.import_module("taskcompendium.pipeline.datasets." + name)
    if name.startswith("hh_"):
        return sample_hh(leaf, output, count, seed)
    if name == "reasoning_gym_generated":
        return sample_generator(leaf, output, count, seed)
    path = (
        "RL/instruction_following/instruction_following.jsonl"
        if name == "nemotron_if"
        else "data/train-00000-of-00001.parquet"
    )
    source = DirectSource(
        name,
        leaf.DATASET,
        leaf.REVISION,
        "default",
        leaf.SPLIT if name == "nemotron_if" else "train",
        path,
        SourceFormat.JSONL if name == "nemotron_if" else SourceFormat.PARQUET,
    )
    return sample_direct(source, output, count=count, seed=seed, max_transfer_bytes=TRANSFER_LIMIT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=NAMES, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6501)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--revision")
    arguments = parser.parse_args()
    if arguments.generate:
        if arguments.revision is None:
            parser.error("--revision is required for the generation worker")
        print(json.dumps(generate_rows(arguments.count, arguments.seed, arguments.revision)))
    else:
        if arguments.name is None or arguments.output is None:
            parser.error("name and --output are required")
        print(json.dumps(sample_source(arguments.name, arguments.output, arguments.count, arguments.seed), indent=2))
