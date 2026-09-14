# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample pinned NeMo Gym rows through the Hugging Face Dataset Viewer API."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

COLLECTION_URL = "https://huggingface.co/api/collections/nvidia/nemo-gym"
DATASETS_URL = "https://datasets-server.huggingface.co"
_TRAINING_BLENDS = frozenset(
    {
        "nvidia/Nemotron-3-Nano-RL-Training-Blend",
        "nvidia/Nemotron-RL-Super-Training-Blends",
        "nvidia/Nemotron-RL-Ultra-Training-Blends",
    }
)


def _read_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def _url(endpoint: str, **parameters: str | int) -> str:
    return f"{DATASETS_URL}/{endpoint}?{urlencode(parameters)}"


def _sample_offsets(dataset: str, total: int, samples_per_corpus: int) -> tuple[int, ...]:
    if total < samples_per_corpus:
        raise ValueError(f"{dataset} has only {total} rows")
    seed = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8])
    offsets = {(seed + index * max(1, total // samples_per_corpus)) % total for index in range(samples_per_corpus)}
    while len(offsets) < samples_per_corpus:
        offsets.add(len(offsets))
    return tuple(sorted(offsets))


def _split(dataset: str) -> dict[str, Any]:
    choices = _read_json(_url("splits", dataset=dataset)).get("splits", [])
    if not choices:
        raise ValueError("Dataset Viewer exposed no splits")
    train = next((choice for choice in choices if choice["split"] == "train"), None)
    return train or choices[0]


def _dataset_samples(dataset: str, samples_per_corpus: int) -> dict[str, Any]:
    repository = _read_json(f"https://huggingface.co/api/datasets/{dataset}")
    split = _split(dataset)
    preview = _read_json(_url("first-rows", dataset=dataset, config=split["config"], split=split["split"]))
    size = _read_json(_url("size", dataset=dataset))
    total = next(
        (
            entry.get("num_rows")
            for entry in size.get("size", {}).get("splits", [])
            if entry.get("config") == split["config"] and entry.get("split") == split["split"]
        ),
        None,
    )
    if not isinstance(total, int):
        raise ValueError("Dataset Viewer did not return a row count")
    rows = []
    for offset in _sample_offsets(dataset, total, samples_per_corpus):
        response = _read_json(
            _url("rows", dataset=dataset, config=split["config"], split=split["split"], offset=offset, length=1)
        )
        row = response.get("rows", [])
        if len(row) != 1:
            raise ValueError(f"Dataset Viewer did not return row {offset}")
        rows.append({"offset": offset, "row": row[0]["row"]})
    return {
        "dataset": dataset,
        "revision": repository["sha"],
        "config": split["config"],
        "split": split["split"],
        "features": preview.get("features", []),
        "num_rows": total,
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-corpus", type=int, default=2)
    parser.add_argument("--dataset", action="append", help="Limit sampling to this collection dataset; repeatable")
    args = parser.parse_args()
    if args.samples_per_corpus < 1:
        raise ValueError("samples-per-corpus must be positive")
    collection = _read_json(COLLECTION_URL)
    datasets = [item["id"] for item in collection["items"] if item["id"] not in _TRAINING_BLENDS]
    if args.dataset:
        requested = set(args.dataset)
        unknown = requested - set(datasets)
        if unknown:
            raise ValueError(f"Not NeMo Gym task datasets: {sorted(unknown)}")
        datasets = [dataset for dataset in datasets if dataset in requested]
    args.output.mkdir(parents=True, exist_ok=False)
    successful = []
    unavailable = []
    for dataset in datasets:
        try:
            sample = _dataset_samples(dataset, args.samples_per_corpus)
        except Exception as error:
            unavailable.append({"dataset": dataset, "error": str(error)})
            continue
        destination = args.output / f"{dataset.rsplit('/', 1)[1]}.json"
        destination.write_text(json.dumps(sample, indent=2) + "\n")
        successful.append({"dataset": dataset, "path": destination.name, "revision": sample["revision"]})
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "collection": {"slug": collection["slug"], "last_updated": collection["lastUpdated"]},
                "samples_per_corpus": args.samples_per_corpus,
                "task_datasets": successful,
                "training_blends": sorted(_TRAINING_BLENDS),
                "unavailable": unavailable,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
