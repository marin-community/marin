# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the forge cohort's physical nop controls over its calibrated consumed packages.

A workspace-graded task needs an unchanged-package production nop trial after its
calibrated oracle: the control consumes the exact package the oracle consumed, runs
the declared preparation with no agent, and must not be credited. This driver walks
the calibrated task map, selects the tasks whose negative-control applicability is
``workspace``, drives one ``harbor.cli.curation_control`` request per task against
the last calibrated attempt's package, and archives every control's output::

    uv run python -m experiments.post_training.task_curation.forge_nop_wave \\
        --calibrated-root /tmp/calibrated --cohort /tmp/cohort.parquet \\
        --output-root /tmp/nop-wave --archive-root s3://.../nop-wave-v1 \\
        --controls /tmp/controls.jsonl --concurrency 32
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.adapter import TASK_SPEC_FILE
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.oracle_filter import negative_control_applicability

from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.negative_controls import negative_control_from_archive
from experiments.post_training.task_curation.oracle_filter import row_task

HARBOR_PROJECT = Path("vendor/harbor").resolve()
NOP_AGENT = "nop"


def calibrated_packages(calibrated_root: Path) -> dict[str, tuple[Path, dict | None, dict | None]]:
    """Each task's last calibrated consumed package and the verifier allocation its trial ran."""
    packages: dict[str, tuple[Path, dict | None, dict | None]] = {}
    for task_dir in sorted(calibrated_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for attempt in sorted(task_dir.glob("attempt-*")):
            # The consumed package is the attempt's task directory: it carries task.toml.
            for package in attempt.iterdir():
                if package.is_dir() and (package / "task.toml").is_file():
                    result = attempt / "jobs" / "oracle" / "result.json"
                    measured = None
                    runtime = None
                    if result.is_file():
                        document = json.loads(result.read_text())
                        config_task = (document.get("config") or {}).get("task") or {}
                        measured = config_task.get("verifier_execution_requirements")
                        runtime = config_task.get("source_runtime")
                    packages[task_dir.name] = (package, measured, runtime)
    return packages


def measured_requirements(package: Path) -> ExecutionRequirements:
    """The consumed package's projected actor budget."""
    document = tomllib.loads((package / "task.toml").read_text())
    environment = document.get("environment") or {}
    return ExecutionRequirements(
        cpus=environment["cpus"],
        memory_mb=environment["memory_mb"],
        storage_mb=environment["storage_mb"],
        agent_timeout_sec=document["agent"]["timeout_sec"],
        verifier_timeout_sec=document["verifier"]["timeout_sec"],
    )


def control_request(
    task_id: str,
    entry: dict,
    package: Path,
    measured_verifier: dict | None,
    source_runtime: dict | None,
    output_root: Path,
) -> dict:
    """One curation_control request: the frozen consumed package with no agent.

    The verifier allocation and the original runtime must match what the oracle
    trial consumed, so both travel from that trial's own config into this
    control's.
    """
    requirements = measured_requirements(package)
    task = TaskSpec.model_validate_json((package / TASK_SPEC_FILE).read_text())
    if task.id != task_id or task_spec_sha256(task) != entry["task_sha256"]:
        raise ValueError(f"Calibrated package differs from its source task: {task_id}")
    consumed = calibrated_task_spec(
        task,
        requirements,
        ExecutionRequirements.model_validate(measured_verifier) if measured_verifier is not None else None,
        (package / "task.toml").read_bytes(),
    )
    return {
        "trial": {
            "task": {
                "path": str(package),
                **({"verifier_execution_requirements": measured_verifier} if measured_verifier is not None else {}),
                **({"source_runtime": source_runtime} if source_runtime is not None else {}),
            },
            "agent": {"name": NOP_AGENT},
            "environment": {"type": "daytona", "kwargs": {"provisioning": "declarative"}},
            "trials_dir": str(output_root / "trials"),
            "trial_name": "nop",
        },
        "source": {
            "task_id": task_id,
            "source_task_sha256": entry["task_sha256"],
            "source_task_checksum": entry["source_task_checksum"],
            "task_sha256": task_spec_sha256(consumed),
            "runtime_sha256": entry.get("runtime_sha256"),
            "task_checksum": entry["task_checksum"],
            "execution_requirements": requirements.model_dump(mode="json"),
            "verifier_execution_requirements": measured_verifier,
        },
        "output_root": str(output_root),
        "native_setup": None,
    }


def run_control(request: dict, output_root: Path, archive_root: str) -> dict:
    """Drive one curation_control request, archive its output, and bind its receipt."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            str(HARBOR_PROJECT),
            "--frozen",
            "--extra",
            "runtime",
            "python",
            "-m",
            "harbor.cli.curation_control",
        ],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nop control failed for {request['source']['task_id']}: {result.stderr[-800:]}")
    task_id = request["source"]["task_id"]
    task_hash = canonical_sha256(task_id)
    archive_uri = str(StoragePath(archive_root) / f"{task_hash}.tar.gz")
    buffer = io.BytesIO()
    # The archive root must be the output directory's own name: the control manifest
    # records its absolute trial result path, and the binding locates the trial by
    # finding the root name inside it.
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        bundle.add(output_root, arcname=output_root.name)
    archive_bytes = buffer.getvalue()
    StoragePath(archive_uri).write_bytes(archive_bytes)
    bound = negative_control_from_archive(
        ContentEvidenceReference(uri=archive_uri, sha256=hashlib.sha256(archive_bytes).hexdigest()),
        f"{output_root.name}/negative-control.json",
        source_task_sha256=request["source"]["source_task_sha256"],
        source_task_checksum=request["source"]["source_task_checksum"],
        execution_requirements=ExecutionRequirements.model_validate(request["source"]["execution_requirements"]),
    )
    (output_root / "receipt.json").write_text(json.dumps(bound.model_dump(mode="json"), indent=1))
    return {"task_id": task_id, "receipt": bound.model_dump(mode="json")}


def wave(
    calibrated_root: Path,
    cohort: Path,
    output_root: Path,
    archive_root: str,
    controls_path: Path,
    concurrency: int,
) -> dict:
    """Run one nop per workspace-graded calibrated task and archive the outputs."""
    task_map = json.loads((calibrated_root / "task-map.json").read_text())
    packages = calibrated_packages(calibrated_root)
    rows = {row["path"]: row for row in pq.read_table(cohort).to_pylist()}
    selected = {}
    for task_id, entry in task_map.items():
        row = rows.get(task_id)
        task = row_task(row) if row is not None else None
        if task is None:
            continue
        if negative_control_applicability(task).value != "workspace":
            continue
        selected_hash = canonical_sha256(task_id)
        selected_pair = packages.get(selected_hash)
        if selected_pair is None:
            raise ValueError(f"No calibrated package for a workspace-graded task: {task_id}")
        selected[task_id] = (entry, selected_pair)
    output_root.mkdir(parents=True, exist_ok=False)
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    with controls_path.open("w") as controls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for task_id, (entry, (package, measured_verifier, source_runtime)) in selected.items():
                task_output = output_root / canonical_sha256(task_id) / "nop-control"
                request = control_request(task_id, entry, package, measured_verifier, source_runtime, task_output)
                futures[pool.submit(run_control, request, task_output, archive_root)] = task_id
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                controls.write(json.dumps(record) + "\n")
    return {"selected": len(selected), "controls": str(controls_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrated-root", type=Path, required=True, help="the staged calibration output root")
    parser.add_argument("--cohort", type=Path, required=True, help="the calibrated cohort parquet")
    parser.add_argument("--output-root", type=Path, required=True, help="fresh root receiving each control's artifacts")
    parser.add_argument("--archive-root", required=True, help="object-storage root receiving the control archives")
    parser.add_argument("--controls", type=Path, required=True, help="the controls.jsonl the collector binds")
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    print(
        json.dumps(
            wave(
                args.calibrated_root, args.cohort, args.output_root, args.archive_root, args.controls, args.concurrency
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
