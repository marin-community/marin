# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and freeze pool P for the SFT shootout.

One job, all inputs pinned:
- TMax: the 2,200 environments of allenai/TMax-SFT-16.5K tasks.zip, converted with hamishivi/tmax's Harbor converter
  (tmax_harbor); the 181 environments with hidden verifier assets are excluded.
- SWE-smith: DCAgent's Harbor packages, outcome-blind stratified sample by repo (sample), rebuilt without git history
  (build_swesmith, issue #10212).
- Held-out: 5% per family, carrying the LR-sweep set's held-out ids.
- Decontamination against SWE-bench Verified, Terminal-Bench 2.1 and TBLite 2.0 (decon).

Writes to the output prefix: tasks.jsonl (one row per pool task), excluded.jsonl, decon.jsonl,
tmax-packages.parquet and swesmith-packages.parquet (columns path, task_binary: one gzipped Harbor task folder per
row), swesmith-packages.jsonl (per-package restore records) and manifest.json with every input pin and file hash.

Usage: python -m experiments.post_training.sft_shootout_pool.pool --heldout <LR-sweep heldout-ids.json> --output <prefix>
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_pool import decon
from experiments.post_training.sft_shootout_pool.build_swesmith import PACKAGES, fix_package, upstream_packages
from experiments.post_training.sft_shootout_pool.pool_io import (
    MANIFEST,
    SWESMITH_PACKAGES,
    jsonl_bytes,
    parquet_bytes,
    swesmith_revision,
    write_pool,
)
from experiments.post_training.sft_shootout_pool.sample import SEED, SWEBENCH_VERIFIED_REPOS, heldout, sample_swesmith
from experiments.post_training.sft_shootout_pool.tmax_harbor import CONVERTER, TMAX_REVISION, convert

TMAX_ENVS = ("allenai/TMax-SFT-16.5K", "848e218270672080a92a94791c028ee43ae33387", "tasks.zip")
ALLKIMI = ("laion/calibforge-relay-traces", "66feab4d24bfc37af4c38d95a6a26187b5648308", "kimi_trials.txt")
TMAX_REVISION_STRING = f"{TMAX_ENVS[0]}@{TMAX_ENVS[1]}+hamishivi/tmax@{TMAX_REVISION[:8]}"


def folder_tarball(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                info = archive.gettarinfo(str(path), arcname=str(path.relative_to(folder)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    return buffer.getvalue()


def allkimi_trials() -> bytes:
    """Download allkimi's pinned kimi_trials.txt."""
    return Path(hf_hub_download(ALLKIMI[0], ALLKIMI[2], repo_type="dataset", revision=ALLKIMI[1])).read_bytes()


def trial_tasks(trials: bytes) -> set[str]:
    """Task ids with at least one trial in kimi_trials.txt (one '<task>__<trial>' per line)."""
    return {line.strip().rsplit("__", 1)[0] for line in trials.decode().splitlines() if line.strip()}


class TmaxConversion(NamedTuple):
    # convert()'s report: converted, failed, excluded and visible-asset task lists, counts and the converter pin.
    report: dict
    tasks: list[str]
    rows: list[dict]
    # The converted Harbor task folders, one per task.
    folders: Path


def convert_tmax(work: Path) -> TmaxConversion:
    """Convert the pinned TMax environments to Harbor tasks under `work` with the pinned hamishivi/tmax converter."""
    tasks_zip = Path(hf_hub_download(TMAX_ENVS[0], TMAX_ENVS[2], repo_type="dataset", revision=TMAX_ENVS[1]))
    tmax_repo = work / "tmax"
    subprocess.run(["git", "clone", "-q", "https://github.com/hamishivi/tmax.git", str(tmax_repo)], check=True)
    subprocess.run(["git", "-C", str(tmax_repo), "checkout", "-q", TMAX_REVISION], check=True)
    converted = work / "tmax-harbor"
    converted.mkdir()
    conversion = convert(tasks_zip, tmax_repo, converted)
    if conversion["failed"]:
        raise ValueError(f"TMax conversion failed for {conversion['failed'][:5]}")
    tasks = sorted(conversion["converted"])
    rows = [{"path": task, "task_binary": folder_tarball(converted / task)} for task in tasks]
    return TmaxConversion(conversion, tasks, rows, converted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", required=True, help="the LR-sweep set's heldout-ids.json (local path or URI)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if (output / MANIFEST).exists():
        raise ValueError("The pool prefix already holds a manifest; use a fresh prefix")
    carried_bytes = StoragePath(args.heldout).read_bytes()
    carried = json.loads(carried_bytes)

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        tmax = convert_tmax(work)
        conversion, tmax_tasks, converted = tmax.report, tmax.tasks, tmax.folders

        # SWE-smith
        upstream = upstream_packages()
        configs, repo_of = upstream.configs, upstream.repo_of
        sample = sample_swesmith(repo_of, set(carried["swesmith"]))
        swesmith_tasks = sample.tasks
        mirrors = work / "mirrors"
        mirrors.mkdir()
        swesmith_rows, package_records = [], []
        for task in swesmith_tasks:
            fixed = fix_package(upstream.binaries[task], mirrors)
            swesmith_rows.append({"path": task, "task_binary": fixed.binary})
            package_records.append({"task_id": task} | fixed.record)

        # Held-out
        allkimi_bytes = allkimi_trials()
        allkimi_tasks = trial_tasks(allkimi_bytes)
        tmax_heldout = set(heldout(tmax_tasks, set(carried["tmax"]), set()))
        swesmith_heldout = set(heldout(swesmith_tasks, set(carried["swesmith"]), allkimi_tasks))

        # Decontamination
        evals = decon.eval_documents(work)
        swesmith_parquet = work / "swesmith.parquet"
        swesmith_parquet.write_bytes(parquet_bytes(swesmith_rows))
        documents = decon.tmax_documents(converted, tmax_tasks) + decon.swesmith_documents(
            swesmith_parquet, set(swesmith_tasks)
        )
        marks = decon.run(documents, evals, work / "decon")

        tmax_revision = TMAX_REVISION_STRING
        files = {
            "tmax-packages.parquet": parquet_bytes(tmax.rows),
            SWESMITH_PACKAGES: swesmith_parquet.read_bytes(),
            "swesmith-packages.jsonl": jsonl_bytes(package_records),
        }
        swesmith_package_revision = swesmith_revision(args.output, files)
        rows = []
        for task in tmax_tasks:
            rows.append(
                {
                    "task_id": task,
                    "family": "tmax",
                    "heldout": task in tmax_heldout,
                    "repo": None,
                    "instance_id": None,
                    "leak_risk": "visible_verifier_asset" if task in conversion["visible_verifier_assets"] else None,
                    "decon": marks[task],
                    "package_source": f"{TMAX_ENVS[0]}/{TMAX_ENVS[2]}",
                    "package_revision": tmax_revision,
                }
            )
        for task in swesmith_tasks:
            rows.append(
                {
                    "task_id": task,
                    "family": "swesmith",
                    "heldout": task in swesmith_heldout,
                    "repo": repo_of[task],
                    "instance_id": configs[task]["instance_id"],
                    "leak_risk": None,
                    "decon": marks[task],
                    "package_source": f"{PACKAGES[0]}@{PACKAGES[1]} rebuilt for #10212",
                    "package_revision": swesmith_package_revision,
                }
            )
        excluded = [
            {"task_id": task, "family": "tmax", "leak_risk": "hidden_verifier_assets"}
            for task in conversion["excluded_hidden_verifier_assets"]
        ]
        files["tasks.jsonl"] = jsonl_bytes(rows)
        files["excluded.jsonl"] = jsonl_bytes(excluded)
        contaminated = [row["task_id"] for row in rows if row["decon"]["contaminated"]]
        manifest = {
            "task": 206,
            "seed": SEED,
            "families": {
                "tmax": {
                    "pool_tasks": len(tmax_tasks),
                    "heldout_tasks": len(tmax_heldout),
                    "source_tasks": conversion["source_tasks"],
                    "excluded_hidden_verifier_assets": len(excluded),
                    "visible_verifier_assets_kept": len(conversion["visible_verifier_assets"]),
                    "normalized_null_base_image": conversion["normalized_null_base_image"],
                    "package_revision": tmax_revision,
                    "converter": conversion["converter"],
                },
                "swesmith": {
                    "pool_tasks": len(swesmith_tasks),
                    "heldout_tasks": len(swesmith_heldout),
                    "source_packages": len(upstream.binaries),
                    "swebench_verified_repo_packages_dropped": sample.verified_dropped,
                    "swebench_verified_repos": sorted(SWEBENCH_VERIFIED_REPOS),
                    "repo_allocation": sample.allocation,
                    "package_revision": swesmith_package_revision,
                },
            },
            "inputs": {
                "tmax_envs": "@".join(TMAX_ENVS[:2]) + "/" + TMAX_ENVS[2],
                "tmax_converter": f"hamishivi/tmax@{TMAX_REVISION}/{CONVERTER}",
                "swesmith_packages": f"{PACKAGES[0]}@{PACKAGES[1]}/tasks.parquet",
                "swesmith_mirrors": "https://github.com/swesmith/<repo> (bare clones at build time)",
                "allkimi_trials": {
                    "source": "@".join(ALLKIMI[:2]) + "/" + ALLKIMI[2],
                    "sha256": hashlib.sha256(allkimi_bytes).hexdigest(),
                },
                "task202_heldout": {"uri": args.heldout, "sha256": hashlib.sha256(carried_bytes).hexdigest()},
                "decon_evals": {
                    "swebench_verified": "@".join(decon.SWEBENCH_VERIFIED),
                    "terminal_bench_2_1": "@".join(decon.TERMINAL_BENCH_21),
                    "tblite_2_0": "@".join(decon.TBLITE),
                    "documents": {name: len(docs) for name, docs in evals.items()},
                },
            },
            "selection": {
                "tmax": "every converted TMax-SFT-16.5K environment except those with hidden verifier assets",
                "swesmith": (
                    "drop SWE-bench Verified repos; allocate tasks across repos in proportion to packages (largest "
                    "remainder); random within repo; no model's pass/fail is used. The LR-sweep set's held-out "
                    "SWE-smith tasks are included first and count toward their repo's allocation (they were Kimi "
                    "passes, so these ids are not outcome-blind)"
                ),
                "heldout": (
                    "round(5%) per family; the LR-sweep set's held-out ids carried; the rest drawn with the seed from "
                    "tasks not in allkimi's kimi_trials.txt"
                ),
            },
            "decon": {
                "method": "marin.datakit.decon.decon_to_parquet, 13-gram, overlap 0.5, min 2 matched features",
                "pool_document": "instruction + verifier test files",
                "contaminated_tasks": contaminated,
                "near_threshold_tasks": {
                    row["task_id"]: round(row["decon"]["max_overlap"], 3)
                    for row in rows
                    if not row["decon"]["contaminated"] and row["decon"]["max_overlap"] >= 0.2
                },
            },
            "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        output.mkdirs()
        write_pool(output, files, manifest)
        summary = {key: value for key, value in manifest.items() if key not in ("inputs", "selection")}
        summary["families"]["swesmith"].pop("repo_allocation")
        print("POOL_BUILT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
