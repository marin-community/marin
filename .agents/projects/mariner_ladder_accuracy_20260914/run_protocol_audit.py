# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run a region-local, model-free old/new harness protocol comparison."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import fsspec
from marin.evaluation.eval_dataset_cache import MANIFEST_FILE, load_eval_datasets_from_gcs

OLD = "git+https://github.com/stanford-crfm/lm-evaluation-harness.git@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab"
NEW = "git+https://github.com/EleutherAI/lm-evaluation-harness.git@f7d0b116146bd616b59d0b991549120f197369a4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert args.output.startswith("gs://marin-us-east5/")
    config = json.loads(args.config.read_text())
    with fsspec.open(config["cache_uri"] + "/" + MANIFEST_FILE, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == config["cache_manifest_sha256"]
    cache = "/tmp/mariner-protocol-cache"
    manifest = load_eval_datasets_from_gcs(config["cache_uri"], cache)
    assert manifest is not None and manifest.supports_full_offline_task_loading()
    env = dict(
        os.environ,
        HF_HOME=cache,
        HF_DATASETS_CACHE=cache + "/datasets",
        HF_HUB_CACHE=cache + "/hub",
        HUGGINGFACE_HUB_CACHE=cache + "/hub",
        HF_MODULES_CACHE=cache + "/modules",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
    )
    env["PYTHONPATH"] = str(Path("lib/levanter/src").resolve())
    reports = {}
    for label, pin in (("old", OLD), ("new", NEW)):
        output = Path(f"/tmp/protocol-{label}.json")
        venv = Path(f"/tmp/protocol-{label}-venv")
        subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True)
        python = str(venv / "bin/python")
        site_packages = venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        # Reuse the exact locked parent runtime; shadow only the harness and TF4 import boundary.
        (site_packages / "marin_parent.pth").write_text(sysconfig.get_path("purelib") + "\n")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                f"lm-eval@{pin}",
                "transformers==4.57.6",
                "huggingface-hub==0.36.2",
                "numexpr==2.14.1",
                "tqdm-multiprocess==0.0.11",
            ],
            check=True,
        )
        command = [
            python,
            str(Path(__file__).with_name("audit_upstream_protocol.py")),
            "--config",
            str(args.config),
            "--output",
            str(output),
        ]
        if label == "new":
            command.append("--upstream")
        subprocess.run(command, env=env, check=True)
        reports[label] = json.loads(output.read_text())
        with fsspec.open(f"{args.output}/{label}.json", "wb") as handle:
            handle.write(output.read_bytes())
    differences = {
        name: {"old": reports["old"][name], "new": reports["new"].get(name)}
        for name in reports["old"]
        if reports["old"][name] != reports["new"].get(name)
    }
    result = {
        "passed": not differences,
        "differences": differences,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "task_only_transformers": "4.57.6",
        "inference": "synthetic deterministic responses, no model",
    }
    with fsspec.open(f"{args.output}/comparison.json", "wt") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)
    if differences:
        raise RuntimeError("Protocol differs; do not release accuracy retries")


if __name__ == "__main__":
    main()
