# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy frozen subset configs into the isolated worktree and derive matched full arms."""

import hashlib
import json
from pathlib import Path

import yaml

from experiments.evaluation.mini_policy.experiment import read_bytes
from experiments.evaluation.mini_policy.materialize import DEST, TIMING_MODELS

OUTPUT = Path("experiments/evaluation/mini_policy/runtime-configs/timing-v6-21x21")
METHODS = ("random", "regularized", "mi_mrmr", "mirt_static")
COUNTS = {
    "NUPA": 5000,
    "aime24": 30,
    "cruxeval": 800,
    "gpqa-diamond": 198,
    "gsm8k-0shot": 1319,
    "humanevalplus": 164,
    "ifbench": 300,
    "math500": 500,
    "mbppplus": 378,
    "mmlu-pro": 12032,
    "mrcr": 1196,
    "triviaqa": 17944,
    "truthfulqa": 817,
    "olympiadbench": 30,
    "bfclparity-pi": 123,
    "bixbench-pi": 205,
    "ds-1000-local": 200,
    "ot-tblite-recovery": 100,
    "swebench-verified": 500,
    "tau3-pi": 375,
    "tb2-recovery": 89,
}


def main():
    for model in TIMING_MODELS:
        slug = model.replace("/", "-")
        for method in METHODS:
            for budget in (210, 840):
                source = f"{DEST}/{slug}/{method}/{budget}"
                raw = read_bytes(source + "/manifest.json")
                manifest = json.loads(raw)
                directory = OUTPUT / slug / f"{method}-{budget}"
                directory.mkdir(parents=True, exist_ok=True)
                for item in manifest["benchmark_manifests"]:
                    content = read_bytes(source + "/" + item["benchmark"] + ".yaml")
                    if hashlib.sha256(content).hexdigest() != item["config_sha256"]:
                        raise ValueError("Frozen benchmark config changed")
                    (directory / (item["benchmark"] + ".yaml")).write_bytes(content)
                manifest["frozen_manifest_source"] = source + "/manifest.json"
                manifest["frozen_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
                (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        parent = OUTPUT / slug / "random-210"
        directory = OUTPUT / slug / "full"
        directory.mkdir(exist_ok=True)
        manifest = json.loads((parent / "manifest.json").read_text())
        for item in manifest["benchmark_manifests"]:
            benchmark = item["benchmark"]
            config = yaml.safe_load((parent / (benchmark + ".yaml")).read_bytes())
            for options in (config.get("task_options") or {}).values():
                options.pop("selected_source_keys", None)
            for dataset in config.get("datasets", []):
                dataset.pop("task_names", None)
            content = yaml.safe_dump(config, sort_keys=False).encode()
            path = directory / (benchmark + ".yaml")
            path.write_bytes(content)
            item.update(
                config=str(path),
                config_sha256=hashlib.sha256(content).hexdigest(),
                tasks=None,
                task_count=COUNTS[benchmark],
            )
        manifest.update(
            method="full",
            budget=None,
            selected_matrix_columns=None,
            status=(
                "Full predictor-pool reference; five excluded predictor benchmarks are "
                "separate full-policy reference components."
            ),
        )
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps({"model": model, "directory": str(OUTPUT / slug), "subset_manifests": 8, "full_benchmarks": 21}),
            flush=True,
        )


if __name__ == "__main__":
    main()
