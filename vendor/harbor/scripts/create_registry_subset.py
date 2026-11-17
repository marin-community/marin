#!/usr/bin/env python3
"""Add a deterministic random subset of an existing registry dataset."""

import argparse
import json
import random
from pathlib import Path

from harbor.models.registry import DatasetSpec, Registry


def datasets_with_random_subset(
    datasets: list[DatasetSpec],
    *,
    source_name: str,
    source_version: str,
    target_version: str,
    count: int,
    seed: int,
    description: str,
) -> list[DatasetSpec]:
    source_matches = [
        dataset
        for dataset in datasets
        if dataset.name == source_name and dataset.version == source_version
    ]
    if len(source_matches) != 1:
        raise ValueError(
            f"expected exactly one source dataset {source_name}@{source_version}, "
            f"found {len(source_matches)}"
        )
    if any(
        dataset.name == source_name and dataset.version == target_version
        for dataset in datasets
    ):
        raise ValueError(
            f"target dataset {source_name}@{target_version} already exists"
        )

    source = source_matches[0]
    tasks = source.tasks
    if count <= 0 or count > len(tasks):
        raise ValueError(f"count must be between 1 and {len(tasks)}, got {count}")

    selected_indices = set(random.Random(seed).sample(range(len(tasks)), count))
    subset = DatasetSpec(
        name=source_name,
        version=target_version,
        description=description,
        tasks=[task for index, task in enumerate(tasks) if index in selected_indices],
    )
    return datasets + [subset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    registry = Registry.from_path(args.registry)
    updated = datasets_with_random_subset(
        registry.datasets,
        source_name=args.source_name,
        source_version=args.source_version,
        target_version=args.target_version,
        count=args.count,
        seed=args.seed,
        description=args.description,
    )
    serialized = [
        dataset.model_dump(mode="json", exclude_defaults=True) for dataset in updated
    ]
    args.registry.write_text(json.dumps(serialized, indent=2) + "\n")


if __name__ == "__main__":
    main()
