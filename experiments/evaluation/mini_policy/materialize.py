# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize held-out subset manifests and canonical child benchmark configs."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.experiment import read_bytes
from experiments.evaluation.mini_policy.identities import document_identity
from experiments.evaluation.mini_policy.source_alignment import EVALCHEMY, HARBOR

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled"
STUDY = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v6-21x21"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/subset-manifests-20261030-v6-21x21"
TIMING_MODELS = [
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-4-26B-A4B-it",
    "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02",
]


def main():
    global STUDY, DEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=STUDY)
    parser.add_argument("--destination", default=DEST)
    parser.add_argument("--method", action="append")
    args = parser.parse_args()
    STUDY, DEST = args.study, args.destination
    methods = args.method or ["random", "regularized", "mi_mrmr", "mirt_static", "mirt_adaptive"]
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    settings = json.loads(read_bytes(STUDY + "/settings.json"))
    matrix = json.loads(read_bytes(settings["source"] + "/matrix/tasks.json"))
    alignment = json.loads(
        read_bytes(
            "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/source-alignment-20261030-v2/alignment.json"
        )
    )
    if not alignment["all_identities_match"]:
        raise ValueError("Source identity audit is not green")
    sources = {}
    for prefix in [EVALCHEMY, HARBOR]:
        for row in json.loads(read_bytes(prefix + "/summary.json")):
            benchmark = row["benchmark"]
            raw = read_bytes(prefix + "/" + benchmark + ".json")
            index = json.loads(raw)
            if prefix == EVALCHEMY:
                rows = [row for row in index["rows"] if benchmark != "cruxeval" or row["namespace"] == "output"]
                config = yaml.safe_load(index["parent"]["config_text"])
            else:
                rows = [
                    {**row, "task_id": document_identity(benchmark, {}, row["task_name"])}
                    for part in index["resolved"]
                    for row in part["rows"]
                ]
                config = yaml.safe_load(index["config_text"])
            by_task = defaultdict(list)
            for row in rows:
                by_task[row["task_id"]].append(row)
            sources[benchmark] = {
                "kind": "evalchemy" if prefix == EVALCHEMY else "harbor",
                "config": config,
                "rows": by_task,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            }
    profiles = yaml.safe_load((CAMPAIGN / "campaign.yaml").read_text())
    receipts = []
    for model in TIMING_MODELS:
        model_index = settings["models"].index(model)
        fold = next(i for i, split in enumerate(settings["splits"]) if model_index in split["test"])
        slug = model.replace("/", "-")
        for method in methods:
            nested = {}
            for budget in [210, 840]:
                record = json.loads(read_bytes(STUDY + f"/fold-{fold}-{method}-0-{budget}.json"))
                selected = next(row["selected"] for row in record["probe_manifests"] if row["model"] == model)
                if len(selected) != budget or len(set(selected)) != budget:
                    raise ValueError("Invalid task budget")
                nested[budget] = selected
                groups = defaultdict(list)
                for column in selected:
                    task = matrix["tasks"][column]
                    groups[task["benchmark"]].append(task["task_id"])
                directory = output / slug / method / str(budget)
                directory.mkdir(parents=True, exist_ok=True)
                manifests = []
                for benchmark, task_ids in groups.items():
                    source = sources[benchmark]
                    missing = set(task_ids) - source["rows"].keys()
                    if missing:
                        raise ValueError(f"Unknown canonical tasks {benchmark}: {sorted(missing)}")
                    config = json.loads(json.dumps(source["config"]))
                    rows = [row for identity in task_ids for row in source["rows"][identity]]
                    if source["kind"] == "evalchemy":
                        if len(config["tasks"]) != 1:
                            raise ValueError("Expected one canonical Evalchemy task")
                        keys = sorted({row["source_key"] for row in rows})
                        config.setdefault("task_options", {}).setdefault(config["tasks"][0], {})[
                            "selected_source_keys"
                        ] = keys
                        requests = sum(row["request_count"] for row in rows)
                    else:
                        if benchmark == "swebench-verified":
                            config = yaml.safe_load(
                                (
                                    ROOT
                                    / (
                                        "experiments/evaluation/mini_policy/runtime-configs/frozen-target/swebench-ve"
                                        "rified.yaml"
                                    )
                                ).read_text()
                            )
                        info = (profiles.get("harbor_model_info") or {}).get(slug)
                        profile_name = (profiles.get("harbor_eval_harness_profiles") or {}).get(benchmark)
                        profile = (profiles.get("harbor_harness_profiles") or {}).get(profile_name, {})
                        for agent in config["agents"]:
                            kwargs = agent.setdefault("kwargs", {})
                            if info and kwargs.get("model_info"):
                                kwargs["model_info"].update(info)
                            if agent["name"] == profile.get("agent"):
                                kwargs.update(profile.get("kwargs") or {})
                                kwargs.update((profile.get("model_kwargs") or {}).get(slug, {}) or {})
                        if len(config["datasets"]) != 1:
                            raise ValueError("Expected one canonical Harbor dataset")
                        config["datasets"][0]["task_names"] = sorted({row["task_name"] for row in rows})
                        requests = None  # Agent request counts depend on the completed episode.
                    path = directory / f"{benchmark}.yaml"
                    path.write_text(yaml.safe_dump(config, sort_keys=False))
                    manifests.append(
                        {
                            "benchmark": benchmark,
                            "tasks": task_ids,
                            "task_count": len(task_ids),
                            "model_requests": requests,
                            "source_index_sha256": source["source_sha256"],
                            "config": str(path.relative_to(output)),
                            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                receipt = {
                    "model": model,
                    "method": method,
                    "seed": 0,
                    "outer_fold": fold,
                    "budget": budget,
                    "selected_matrix_columns": selected,
                    "benchmark_manifests": manifests,
                    "adaptive_path_replay_only": method.startswith("mirt_adaptive"),
                    "warning": (
                        "Adaptive manifests describe historical paths; fresh adaptive execution must select online."
                        if method.startswith("mirt_adaptive")
                        else None
                    ),
                    "comparison_source": STUDY,
                    "status": "Not launched; model-serving config audit and e2e canary required.",
                }
                (directory / "manifest.json").write_text(json.dumps(receipt, indent=2))
                receipts.append(receipt)
            if nested[840][:210] != nested[210]:
                raise ValueError("Non-nested micro/mini")
    (output / "summary.json").write_text(
        json.dumps(
            [
                {
                    "model": r["model"],
                    "method": r["method"],
                    "tasks": r["budget"],
                    "benchmarks": len(r["benchmark_manifests"]),
                    "adaptive_path_replay_only": r["adaptive_path_replay_only"],
                }
                for r in receipts
            ],
            indent=2,
        )
    )
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        fs, key = filesystem_for(DEST + "/" + str(path.relative_to(output)))
        fs.pipe_file(key, path.read_bytes())
    print(json.dumps({"destination": DEST, "manifests": len(receipts), "models": TIMING_MODELS}), flush=True)


if __name__ == "__main__":
    main()
