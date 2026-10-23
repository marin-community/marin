# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the historical runtime in a new, isolated checkout; launch no jobs."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path, help="Local hero-bpe-v16384 archive; required only for native data/control"
    )
    parser.add_argument("--gpu-count", type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    destination = args.checkout.resolve()
    if destination.exists():
        raise ValueError(f"Refusing to replace existing checkout: {destination}")
    reference = json.loads((ROOT / "candidate11-reference.json").read_text())
    audit = json.loads((ROOT / "records/data-manifest.json").read_text())
    if args.tokenizer is not None:
        for name, expected in audit["tokenizer_file_sha256"].items():
            if hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Tokenizer fingerprint differs: {name}")
    repository = ROOT.parents[2]
    subprocess.run(
        ["git", "-C", str(repository), "worktree", "add", "--detach", str(destination), reference["source_revision"]],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "apply", "--unidiff-zero", "--check", str(ROOT / "runtime.patch")], check=True
    )
    subprocess.run(["git", "-C", str(destination), "apply", "--unidiff-zero", str(ROOT / "runtime.patch")], check=True)
    target = destination / ROOT.relative_to(repository)
    shutil.copytree(
        ROOT,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", "tokenizers", "data-native-audit", "gpu-probe-manifest.json"
        ),
    )
    (target / "data-native-audit").mkdir()
    shutil.copyfile(ROOT / "records/data-manifest.json", target / "data-native-audit/manifest.json")
    if args.tokenizer is not None:
        tokenizer = target / "tokenizers/hero-bpe-v16384"
        tokenizer.mkdir(parents=True)
        for name in audit["tokenizer_file_sha256"]:
            shutil.copyfile(args.tokenizer / name, tokenizer / name)
    files = [
        *target.glob("*.py"),
        target / "candidate11-reference.json",
        target / "data-native-audit/manifest.json",
    ]
    if args.tokenizer is not None:
        files.extend((target / "tokenizers/hero-bpe-v16384").glob("*.json"))
    manifest = {
        "source_revision": reference["source_revision"],
        "runtime_patch_sha256": (
            hashlib.sha256(subprocess.check_output(["git", "-C", str(destination), "diff"])).hexdigest()
        ),
        "model_source_sha256": (
            hashlib.sha256((destination / "experiments/grug/fast_track/model.py").read_bytes()).hexdigest()
        ),
        "experiment_files": {
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
        },
        "cluster": "cw-rno2a",
        "priority": "interactive",
        "gpu_count": args.gpu_count,
    }
    (target / "gpu-probe-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
