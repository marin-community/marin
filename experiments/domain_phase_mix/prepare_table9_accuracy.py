# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the 20 non-deferred backfill tasks using the original OLMo-Eval formatters.

Run as a module with the pinned OLMo-Eval checkout on PYTHONPATH. OLMo-Eval is a
build-time dependency only; TPU workers consume the frozen JSON artifacts.
"""

import argparse
import dataclasses
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import fsspec
from marin.evaluation.olmo_base_eval.accuracy import CHOICE_BACKFILL_TASKS, DEFERRED_TASKS, GENERATION_BACKFILL_TASKS
from olmo_eval.evals.tasks.common import get_task

CHOICE_TASK_SET = frozenset(CHOICE_BACKFILL_TASKS)


def export_task(name: str) -> tuple[list[dict], dict]:
    """Keep the BPB task's prompt, split and document identity for companion scoring."""
    task = get_task(f"{name}:bpb:olmo3base")
    rows = []
    for index, doc in enumerate(task.instances):
        request = task.format_request(doc)
        if request.continuation_prompts and len(set(request.continuation_prompts)) != 1:
            raise ValueError(f"Per-choice prompts require an explicit adapter: {name}")
        choices = list(request.continuations or ())
        gold = doc.metadata.get("gold_idx", 0)
        if name in CHOICE_TASK_SET:
            if not doc.choices or len(doc.choices) < 2:
                raise ValueError(f"No wrong answers available for {name}/{index}")
            if len(choices) == 1:
                expected = choices[0]
                choices = [" " + c for c in doc.choices]
                if choices[gold] != expected:
                    raise ValueError(f"Gold continuation formatting differs for {name}/{index}")
            if len(choices) != len(doc.choices):
                raise ValueError(f"Incomplete alternatives for {name}/{index}")
        rows.append(
            {
                "task": name,
                "doc_id": index,
                "context": request.prompt,
                "choices": choices if name in CHOICE_TASK_SET else [],
                "gold_index": gold,
                "reference": choices[0 if len(choices) == 1 else gold],
                "metadata": doc.metadata,
            }
        )
    metric = "acc_per_token" if name.startswith("basic_skills_") else "acc_per_char"
    if name in GENERATION_BACKFILL_TASKS:
        metric = "exact_match" if name.startswith("minerva_math_") else "pass@1"
    stops = ["```"] if name == "mbpp" else list(task.config.sampling_params.stop_sequences or ())
    return rows, {
        "count": len(rows),
        "metric": metric,
        "task_spec": f"{name}:bpb:olmo3base",
        "config": dataclasses.asdict(task.config),
        "generation": {"max_gen_toks": 1024, "temperature": 0.0, "seed": 0, "n": 1, "until": stops},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-eval-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.olmo_eval_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.olmo_eval_checkout), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("OLMo-Eval must be a clean pinned checkout")
    with fsspec.open("gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/v2/manifest.json") as handle:
        native = json.load(handle)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "olmo_eval_git_sha": revision,
        "native_request_set": "gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/v2",
        "deferred": list(DEFERRED_TASKS),
        "tasks": {},
    }
    for name in (*CHOICE_BACKFILL_TASKS, *GENERATION_BACKFILL_TASKS):
        rows, spec = export_task(name)
        if len(rows) != native["tasks"][name]:
            raise ValueError(f"Native task count mismatch: {name}: {len(rows)} != {native['tasks'][name]}")
        encoded = gzip.compress(json.dumps(rows, sort_keys=True, default=str).encode(), mtime=0)
        filename = name + ".json.gz"
        (args.output / filename).write_bytes(encoded)
        spec.update({"file": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)})
        manifest["tasks"][name] = spec
        print(f"Exported {name}: {len(rows)} documents", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
