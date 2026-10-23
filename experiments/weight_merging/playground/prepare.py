# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download pinned models and freeze the exact small evaluation panel."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from experiments.weight_merging.playground.evaluation import GSM_SMOKE_IDS

MODELS = Path(__file__).with_name("models.json")
MBPP_COUNT = 8
DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google-research-datasets/mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["anchor", "instruct", "coder", "math"])
    parser.add_argument("--panel", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MODELS.read_text())
    paths = {}
    for name in args.models:
        paths[name] = snapshot_download(
            manifest[name]["repo"],
            revision=manifest[name]["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json"],
        )
    (args.output / "model-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    revisions = DATASET_REVISIONS
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=revisions["openai/gsm8k"])
    mbpp = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test", revision=revisions["google-research-datasets/mbpp"]
    )
    items = []
    for index in GSM_SMOKE_IDS if args.panel == "smoke" else range(len(gsm)):
        row = gsm[index]
        items.append(
            {
                "id": f"gsm8k/test/{index}",
                "dataset": "gsm8k",
                "prompt": row["question"],
                "answer": row["answer"].split("####")[-1].strip(),
                "tests": [],
                "setup": "",
            }
        )
    for row in sorted(mbpp, key=lambda r: r["task_id"])[: MBPP_COUNT if args.panel == "smoke" else len(mbpp)]:
        items.append(
            {
                "id": f'mbpp/sanitized/test/{row["task_id"]}',
                "dataset": "mbpp",
                "prompt": row["prompt"],
                "answer": row["code"],
                "tests": row["test_list"],
                "setup": row.get("test_imports", []),
            }
        )
    panel = {"dataset_revisions": revisions, "items": items}
    payload = json.dumps(panel, indent=2) + "\n"
    (args.output / "panel.json").write_text(payload)
    print(
        json.dumps(
            {"panel_sha256": hashlib.sha256(payload.encode()).hexdigest(), "items": len(items), "model_paths": paths},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
