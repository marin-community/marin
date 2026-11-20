# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fidelity-gate trials: the condensed SWE-rebench V2.1 env against the upstream digest image.

Reads the sample manifest produced by ``swe_rebench_fidelity_gate`` and runs, per sampled
row, nop + oracle LOCAL Harbor trials on two hardened packages that differ only in their
environment (same SWE_REBENCH adapter, same vendored upstream grading bytes, same contract
family):

- ``upstream``: the row's own digest-pinned image with no task-start setup;
- ``condensed``: the repair's shared base image plus the declared task-start setup.

Outcome comparison per row: reward equality per phase, plus the upstream grade.py per-test
verdicts when the trial captures them. Local trials follow the scale-lego protected-QA
pattern (the hardened image layer moves into the pre-agent setup; the declared image is
consumed digest-pinned without a build). Scope: fidelity gate only — the full-population
oracle/nop qualification still runs through the standard pipeline.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.shared_source import harden_shared_source
from taskcompendium.integrity import SharedDependencyState

from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    WORKSPACE,
    dockerfile,
    family_config,
    row_base,
    setup_command,
    swe_rebench_profile,
    task_toml,
    to_workspace,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream import task_files
from experiments.post_training.task_curation.difficulty.runtime import worker_limits
from experiments.post_training.task_curation.scale_lego_protected_qa_v1 import (
    ATTEMPT_TIMEOUT,
    adapt_for_local,
    run_trial,
    trial_config,
)
from experiments.post_training.task_curation.swe_rebench_fidelity_gate import load_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("swe-rebench-fidelity-trials")


def _flatten(data: dict) -> dict:
    flat = dict(data)
    install_config = data.get("install_config") or {}
    flat["test_cmd"] = install_config.get("test_cmd")
    flat["log_parser"] = install_config.get("log_parser")
    flat["install"] = install_config.get("install") or []
    return flat


def row_variant_files(row: dict, variant: str, base_image: str | None) -> dict[str, bytes]:
    """One sampled row's hardened package for the declared variant."""
    flat = _flatten(row)
    base = (row.get("install_config") or {}).get("base_image_name") or groups.V1_PYTHON_BASE
    group = groups.GROUP_BY_BASE[base]
    upstream_files = task_files.task_files(flat)
    profile = swe_rebench_profile()
    if variant == "upstream":
        # The upstream renderer's own task.toml (digest-pinned image, no setup), with the
        # family config keys added so the same profile hardens both variants. The vendored
        # renderer declares only docker_image; the hardening contract requires a Dockerfile,
        # so the variant carries the minimal FROM for the row's pinned image.
        files = {name: content.encode() for name, content in upstream_files.items()}
        files["environment/Dockerfile"] = f"FROM {row['image_name']}\n".encode()
    elif variant == "condensed":
        setup = setup_command(flat, base, group)
        files = {
            "task.toml": task_toml(flat, base, group, setup).encode(),
            "environment/Dockerfile": dockerfile(group, base_image).encode(),
        }
        for name, content in upstream_files.items():
            if name == "task.toml":
                continue
            if name.startswith("solution/"):
                # The condensed variant's solve script runs in the uniform workspace; the
                # upstream variant keeps the row's original paths (its image bakes them).
                content = to_workspace(content, flat["workdir"])
            files[name] = content.encode()
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    config = family_config(flat)
    if variant == "upstream":
        # The upstream digest image bakes the row's repo at its own workdir: the upstream
        # variant keeps the row's original paths (the /testbed rewrite belongs to the
        # condensed consolidation; ruling 2026-11-20).
        config["workdir"] = flat["workdir"]
        config["test_cmd"] = flat["test_cmd"]
    files["tests/config.json"] = json.dumps(config, indent=1, ensure_ascii=False).encode() + b"\n"
    harden_shared_source(files, profile=profile, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    return files


def write_package(root: Path, name: str, files: dict[str, bytes], row: dict, variant: str, image: str) -> Path:
    package = root / "packages" / name
    if package.exists():
        raise FileExistsError(package)
    package.mkdir(parents=True)
    workdir = row["workdir"] if variant == "upstream" else WORKSPACE
    adapt_for_local(files, workdir, image)
    for relative, content in sorted(files.items()):
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {
        "task_id": row["instance_id"],
        "variant": variant,
        "declared_image": image,
        "upstream_revision": groups.UPSTREAM_REVISION,
        "files": {
            relative: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for relative, content in sorted(files.items())
        },
    }
    (root / "packages" / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return package


def trial(root: Path, row: dict, variant: str, phase: str, image: str, limits: dict) -> dict:
    name = f"{row['instance_id']}-{variant}"
    package = write_package(root, name, row_variant_files(row, variant, image), row, variant, image)
    attempt = root / "trials" / f"{name}-{phase}"
    attempt.mkdir(parents=True, exist_ok=True)
    agent = {"name": "nop"} if phase == "nop" else {"name": "oracle"}
    config = trial_config(package, attempt / "jobs", agent, f"{name}-{phase}", "local", limits)
    config_path = attempt / "trial.json"
    config_path.write_text(json.dumps(config))
    summary = asyncio.run(
        asyncio.wait_for(_run_logged(config_path, attempt / "harbor.log"), timeout=ATTEMPT_TIMEOUT + 900)
    )
    LOGGER.info("[%s/%s] reward=%s", name, phase, summary["rewards"].get("reward"))
    return summary


async def _run_logged(config_path: Path, log_path: Path) -> dict:
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return await run_trial(config_path, {})


def compare(row: dict, upstream: dict, condensed: dict) -> dict:
    def outcomes(summary: dict) -> dict:
        return {
            "nop_reward": summary["nop"]["rewards"].get("reward"),
            "oracle_reward": summary["oracle"]["rewards"].get("reward"),
        }

    up, con = outcomes(upstream), outcomes(condensed)
    return {
        "task_id": row["instance_id"],
        "group": groups.GROUP_BY_BASE[row_base(row)].group_id,
        "upstream": up,
        "condensed": con,
        "match": up == con,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-uri", required=True, help="URI of the fidelity sample manifest")
    parser.add_argument("--bindings-uri", required=True, help="URI of the base-image bindings manifest")
    parser.add_argument("--dataset-uri", required=True, help="URI of the staged pinned V2.1 shard directory")
    parser.add_argument("--output-dir", required=True, help="Directory (local or s3://) for trial records and verdicts")
    args = parser.parse_args()

    root = (
        Path(os.environ["IRIS_OUTPUT_DIR"]) / "swe-rebench-fidelity-gate-v1"
        if os.environ.get("IRIS_OUTPUT_DIR")
        else Path(args.output_dir) / "gate"
    )
    limits_source = os.environ.get("GATE_LIMITS")
    try:
        limits = json.loads(limits_source) if limits_source else worker_limits(root)
    except Exception as error:
        # The disposable-container cgroup delegation is a sandboxing nicety, not the variable
        # under test; the gate compares two environments under the same backend settings. The
        # sandbox BINARIES (bwrap/skopeo/umoci/e2fsprogs) are still required, so provision them
        # here with the non-interactive dpkg options the worker helper does not set.
        LOGGER.warning("worker cgroup provisioning unavailable (%s); trials run under the job's container limits", error)
        os.environ["DEBIAN_FRONTEND"] = "noninteractive"
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(
            [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "-o",
                "Dpkg::Options::=--force-confdef",
                "-o",
                "Dpkg::Options::=--force-confold",
                "bubblewrap",
                "e2fsprogs",
                "util-linux",
                "umoci",
                "skopeo",
            ],
            check=True,
        )
        limits = {"disk_root": str(root / "sandbox-disks")}
    manifest = json.loads(StoragePath(args.sample_uri).read_text())
    bindings = json.loads(StoragePath(args.bindings_uri).read_text())
    # Stage the pinned shards to local disk with fsutil (the CLI resolves the regional store
    # credentials; fsspec-backed parquet reads segfaulted in this container). The job then
    # reads plain local files with pyarrow.
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for index in range(6):
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "fsutil",
                "cp",
                f"{args.dataset_uri}/train-{index:05d}-of-00006.parquet",
                str(data_dir),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    rows_by_id = {}
    for row in load_rows(data_dir):
        rows_by_id[row["instance_id"]] = row
    missing = {
        instance_id
        for members in manifest["groups"].values()
        for instance_id in members
        if instance_id not in rows_by_id
    }
    if missing:
        raise ValueError(f"Sample references rows absent from the loaded population: {sorted(missing)[:5]}")
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for group_id, members in sorted(manifest["groups"].items()):
            image = bindings["images"][group_id]
            for instance_id in members:
                row = rows_by_id[instance_id]
                for variant, phase in (
                    ("upstream", "nop"),
                    ("upstream", "oracle"),
                    ("condensed", "nop"),
                    ("condensed", "oracle"),
                ):
                    futures[(instance_id, variant, phase)] = pool.submit(trial, root, row, variant, phase, image, limits)
        results = {}
        for key, future in futures.items():
            results[key] = future.result()
        for instance_id in sorted({key[0] for key in results}):
            upstream = {
                "nop": results[(instance_id, "upstream", "nop")],
                "oracle": results[(instance_id, "upstream", "oracle")],
            }
            condensed = {
                "nop": results[(instance_id, "condensed", "nop")],
                "oracle": results[(instance_id, "condensed", "oracle")],
            }
            records.append(compare(rows_by_id[instance_id], upstream, condensed))
    (root / "fidelity-records.json").write_text(json.dumps(records, indent=2) + "\n")
    by_group: dict[str, dict[str, int]] = {}
    for record in records:
        verdict = by_group.setdefault(record["group"], {"tasks": 0, "match": 0, "mismatch": 0})
        verdict["tasks"] += 1
        verdict["match" if record["match"] else "mismatch"] += 1
    (root / "fidelity-verdicts.json").write_text(json.dumps(by_group, indent=2) + "\n")
    LOGGER.info("fidelity verdicts per group: %s", json.dumps(by_group))
    mismatches = [record for record in records if not record["match"]]
    if mismatches:
        (root / "fidelity-mismatches.json").write_text(json.dumps(mismatches, indent=2) + "\n")
        raise SystemExit(f"{len(mismatches)} rows mismatch; per-group ruling required")


if __name__ == "__main__":
    main()
