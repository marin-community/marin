# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a bounded launch queue for matched full/Micro/Mini timing components."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from experiments.evaluation.mini_policy.launch_timing import COMPONENTS
from experiments.evaluation.mini_policy.materialize import TIMING_MODELS
from experiments.evaluation.mini_policy.timing_materialize import OUTPUT

CAMPAIGN = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled")
FROZEN = Path("experiments/evaluation/mini_policy/runtime-configs/frozen-target")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-root", type=Path, default=OUTPUT)
    parser.add_argument("--reference-root", type=Path, default=OUTPUT)
    parser.add_argument("--methods", nargs="+", default=["full", "random", "regularized", "greedy_cv", "mirt_static"])
    args = parser.parse_args()
    launches = []
    for model in TIMING_MODELS:
        slug = model.replace("/", "-")
        parent = CAMPAIGN / "model-configs" / (slug + ".yaml")
        tau = FROZEN / "model-configs/tau3" / parent.name
        tau.parent.mkdir(exist_ok=True, parents=True)
        config = yaml.safe_load(parent.read_text())
        config["serve"]["max_model_len"] = 131072
        config.setdefault("generation", {})["max_gen_toks"] = 32768
        tau.write_text(yaml.safe_dump(config, sort_keys=False))
        for method in args.methods:
            for budget in ((None,) if method == "full" else (210, 840)):
                arm = "full" if budget is None else f"{method}-{budget}"
                directory = (args.reference_root if method == "full" else args.timing_root) / slug / arm
                selected = json.loads((directory / "manifest.json").read_text())
                for component, names in COMPONENTS.items():
                    available = [row for row in selected["benchmark_manifests"] if row["benchmark"] in names]
                    if not available:
                        continue
                    model_config = parent
                    if component == "tau3":
                        model_config = tau
                    if component == "swebench":
                        model_config = FROZEN / "model-configs/swebench" / parent.name
                    suffix = "harbor" if component == "harbor-standard" else component
                    model_name = {"Qwen/Qwen3.6-35B-A3B": "qwen36", "google/gemma-4-26B-A4B-it": "gemma4"}.get(
                        model, "antidoom"
                    )
                    run_id = f"mini-policy-{model_name}-{arm.replace('_', '-')}-{suffix}-20261030"
                    # The Mini random Olympiad component contains every task. A
                    # completed full arm can be reused only after identity parity.
                    if method == "random" and budget == 840 and component == "olympiad":
                        continue
                    row = {
                        "run_id": run_id,
                        "model": model,
                        "model_config": str(model_config),
                        "component": component,
                        "method": method,
                        "budget": budget,
                        "configs_dir": str(directory),
                        "manifest_sha256": digest(directory / "manifest.json"),
                        "model_config_sha256": digest(model_config),
                        "component_tasks": sum(item["task_count"] for item in available),
                    }
                    if component in {"olympiad", "finance", "sotopia"}:
                        row["finalization_config"] = str(CAMPAIGN / "finalization.yaml")
                    if component == "bixbench":
                        row["judge_model_config"] = str(CAMPAIGN / "judge-model-configs/MiniMaxAI-MiniMax-M3-MXFP8.yaml")
                    if model == TIMING_MODELS[0]:
                        receipt_name = f"{arm}-{suffix}.receipt.json"
                        if component == "nonjudge" and method == "random":
                            receipt_name = f"{arm}-{suffix}-v2.receipt.json"
                        receipt_path = args.output.parent / receipt_name
                        if receipt_path.exists():
                            prior = json.loads(receipt_path.read_text())
                            if prior.get("model") != model:
                                raise ValueError("Existing receipt model mismatch")
                            old_directory = args.reference_root / slug / arm
                            equivalent = False
                            if (old_directory / "manifest.json").exists():
                                old_manifest = json.loads((old_directory / "manifest.json").read_text())
                                old_hashes = {
                                    r["benchmark"]: r["config_sha256"]
                                    for r in old_manifest["benchmark_manifests"]
                                    if r["benchmark"] in names
                                }
                                new_hashes = {r["benchmark"]: r["config_sha256"] for r in available}
                                equivalent = old_hashes == new_hashes
                            if equivalent:
                                row["existing_job"] = prior["job"]
                                row["existing_receipt"] = str(receipt_path)
                            else:
                                row["run_id"] += "-v5"
                    launches.append(row)
    # Start the missing full-reference components first, then complete each
    # model's smaller subsets before allocating additional long references.
    long_reference = {"nonjudge", "harbor-standard", "swebench", "tau3", "bixbench"}
    launches.sort(
        key=lambda row: (
            0 if row["method"] == "full" and row["component"] in long_reference else 1 if row["method"] == "full" else 2,
            row["component"] if row["method"] == "full" else str(row["budget"]),
            TIMING_MODELS.index(row["model"]),
            row["method"],
            row["component"],
        )
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(
        json.dumps(
            {
                "models": TIMING_MODELS,
                "launches": launches,
                "budget_unit": "unique tasks; benchmark-prescribed repeated trials remain grouped",
                "max_in_flight": 20,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"launches": len(launches), "output": str(args.output)}))


if __name__ == "__main__":
    main()
