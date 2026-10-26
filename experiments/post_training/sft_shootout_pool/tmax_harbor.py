# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert TMax-SFT-16.5K environments to Harbor tasks with TMax's own converter, minus leaking environments.

Uses hamishivi/tmax@6d3d606c rl_data/scripts/analyze/convert_to_harbor.py unchanged, imported from a checkout of that
revision. One normalization is applied to a staged copy of each task before conversion: a null task.json base_image
becomes "" (the converter calls .replace on it; null means a legacy, self-contained container.def). Environments
whose build creates hidden verifier assets (tmax_leaks) are not converted.

Usage:
    python -m experiments.post_training.sft_shootout_pool.tmax_harbor \
        --tasks-zip tasks.zip --tmax-repo <checkout> --output <dir>
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from experiments.post_training.sft_shootout_pool.tmax_leaks import answer_paths

TMAX_REVISION = "6d3d606c0f9de6ceec9481f9ccc793936818fef0"
CONVERTER = "rl_data/scripts/analyze/convert_to_harbor.py"


def load_converter(repository: Path):
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    if head.stdout.strip() != TMAX_REVISION:
        raise ValueError(f"tmax checkout is at {head.stdout.strip()}, expected {TMAX_REVISION}")
    spec = importlib.util.spec_from_file_location("convert_to_harbor", repository / CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(tasks_zip: Path, repository: Path, output: Path) -> dict:
    converter = load_converter(repository)
    base_post, base_env = converter.load_intricate_base(converter.DEFAULT_BASE_INTRICATE_DEF)
    with zipfile.ZipFile(tasks_zip) as archive:
        leaks = answer_paths(archive)
        excluded = sorted(task for task, paths in leaks.items() if paths["hidden"])
        tasks = sorted({name.split("/", 1)[0] for name in archive.namelist() if name.endswith("/task.json")})
        normalized, converted, failed = [], [], []
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            for task in tasks:
                if task in excluded:
                    continue
                members = [
                    name for name in archive.namelist() if name.startswith(f"{task}/") and "/solutions/" not in name
                ]
                archive.extractall(staging, members)
                meta_path = staging / task / "task.json"
                meta = json.loads(meta_path.read_text())
                if "base_image" in meta and meta["base_image"] is None:
                    meta["base_image"] = ""
                    meta_path.write_text(json.dumps(meta))
                    normalized.append(task)
                name = converter.convert_task(
                    staging / task,
                    output,
                    intricate_base_post=base_post,
                    intricate_env=base_env,
                    intricate_base_image_tag=None,
                )
                (converted if name else failed).append(task)
                shutil.rmtree(staging / task)
    return {
        "converter": f"hamishivi/tmax@{TMAX_REVISION}/{CONVERTER}",
        "source_tasks": len(tasks),
        "excluded_hidden_verifier_assets": excluded,
        "visible_verifier_assets": sorted(
            task for task, paths in leaks.items() if paths["visible"] and not paths["hidden"]
        ),
        "normalized_null_base_image": normalized,
        "converted": converted,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-zip", type=Path, required=True)
    parser.add_argument("--tmax-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    report = convert(args.tasks_zip, args.tmax_repo, args.output)
    (args.output.parent / f"{args.output.name}-conversion.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in report.items()}))


if __name__ == "__main__":
    main()
