# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the repaired MiMo release's physical NoP controls on the LOCAL trial path.

Stage g proves the declared grader end to end: the pinned image starts, the task's
agent setup command captures the base ref, the NoP agent does nothing, and the
root-delivery grader quiesces, delivers the private verifier files, runs
``setup.sh`` then ``test.sh`` verbatim, and writes a fail-closed zero reward. Each
control consumes the sha-verified release rows through the maintained trial-package
writer, runs one ``harbor.cli.curation_control`` request in Harbor's locked runtime,
and binds its archived receipt through the maintained negative-control binding.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from finestore.rl.curation import ContentEvidenceReference
from huggingface_hub import hf_hub_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.adapter import declared_execution_limits, write_harbor_task
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.mimo.repair import ARCHIVE_PREFIX
from experiments.post_training.task_curation.negative_controls import negative_control_from_archive
from experiments.post_training.task_curation.oracle_filter import row_task

logger = logging.getLogger(__name__)

RELEASE_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-pinned-images-20261124-a-v1"
RELEASE_REVISION = "729925160824efba40d672c10ee4311169dc1a8b"
EXPECTED_TASKS_PARQUET_SHA256 = "e320474a633072e760e171227c928fd2de6cadde1813771d95ceeeed7d80c05e"
SAMPLE_SEED = 0
NOP_AGENT = "nop"
HARBOR_PROJECT = Path("vendor/harbor").resolve()
TRIAL_STORAGE_MB = 10240
"""The trial disk envelope: the source declares no storage (undeclared, preserved in
the runtime binding), and the trial package still needs a concrete budget shape."""


def release_rows(workdir: Path) -> list[tuple[TaskSpec, dict, str]]:
    """The sha-verified release rows as (task, runtime, encoded spec) triples."""
    path = Path(hf_hub_download(RELEASE_REPO, "tasks.parquet", repo_type="dataset", revision=RELEASE_REVISION))
    digest = file_sha256(StoragePath(str(path)))
    if digest != EXPECTED_TASKS_PARQUET_SHA256:
        raise ValueError(f"The release rows differ from the conversion summary: {digest}")
    rows = []
    import pyarrow.parquet as pq  # noqa: PLC0415

    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            task = row_task(row)
            if task is None:
                raise ValueError("Release row carries no TaskSpec")
            rows.append((task, json.loads(row["runtime"]) if row.get("runtime") else {}, row["task_spec"]))
    return rows


def seeded_control_tasks(rows: list[tuple[TaskSpec, dict, str]], sample: int) -> list[tuple[TaskSpec, dict, str]]:
    """The standard seed-zero order, so the controls review the frozen panel's own tasks."""
    _count, ordered = seeded_sample(
        rows, size=min(sample, len(rows)), key=lambda row: seeded_order(row[0].id, SAMPLE_SEED)
    )
    return ordered


def original_package_checksum(task: TaskSpec, workdir: Path) -> str:
    """The archived original package's checksum, materialized from its provenance files."""
    from dirhash import dirhash  # noqa: PLC0415

    original = workdir / "original-packages" / canonical_sha256(task.id)
    if original.exists():
        shutil.rmtree(original)
    original.mkdir(parents=True)
    for resource in task.resources.verifier:
        if not resource.path.startswith(ARCHIVE_PREFIX):
            raise ValueError(f"Repaired task carries a non-archived verifier file: {task.id} {resource.path}")
        target = original / resource.path.removeprefix(ARCHIVE_PREFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
    return dirhash(str(original), "sha256")


def trial_package_checksum(directory: Path) -> str:
    """Harbor's deterministic package checksum over the trial package."""
    from dirhash import dirhash  # noqa: PLC0415

    return dirhash(str(directory), "sha256")


def local_environment_kwargs(workdir: Path) -> dict:
    """The LOCAL factory options: the trial disk pool and the GHCR pull authfile."""
    kwargs: dict = {"disk_root": str(workdir / "disk")}
    authfile = get_secret("GHCR_REGISTRY_AUTHFILE")
    if authfile:
        path = Path(tempfile.gettempdir()) / f"mimo-ghcr-authfile-{os.getpid()}.json"
        path.write_text(authfile)
        path.chmod(0o600)
        kwargs["registry_authfile"] = str(path)
    return kwargs


def verifier_shape(requirements: ExecutionRequirements, runtime: dict) -> ExecutionRequirements:
    """The trial's derived verifier allocation: the initial shape bounded by the declared machine."""
    from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS  # noqa: PLC0415

    machine = (runtime or {}).get("verifier_machine") or {}
    limits = {name: machine[name] for name in ("cpus", "memory_mb", "storage_mb") if machine.get(name) is not None}
    return INITIAL_REQUIREMENTS.model_copy(
        update={
            **limits,
            "agent_timeout_sec": requirements.agent_timeout_sec,
            "verifier_timeout_sec": requirements.verifier_timeout_sec,
        }
    )


def control_request(
    task: TaskSpec,
    runtime: dict,
    package: Path,
    requirements: ExecutionRequirements,
    task_checksum: str,
    source_task_checksum: str,
    output_root: Path,
    environment_kwargs: dict,
) -> dict:
    """One curation_control request: the repaired package with no agent on the LOCAL path."""
    return {
        "trial": {
            "task": {
                "path": str(package),
                "source_runtime": (
                    {
                        "task_sha256": task_spec_sha256(task),
                        "runtime_sha256": canonical_sha256(runtime) if runtime else None,
                        "runtime": runtime,
                    }
                    if runtime
                    else None
                ),
            },
            "agent": {"name": NOP_AGENT},
            "environment": {"type": "local", "kwargs": environment_kwargs},
            "trials_dir": str(output_root / "trials"),
            "trial_name": "nop",
        },
        "source": {
            "task_id": task.id,
            "source_task_sha256": task_spec_sha256(task),
            "source_task_checksum": source_task_checksum,
            "task_sha256": task_spec_sha256(task),
            "runtime_sha256": canonical_sha256(runtime) if runtime else None,
            "task_checksum": task_checksum,
            "execution_requirements": requirements.model_dump(mode="json"),
            "verifier_execution_requirements": verifier_shape(requirements, runtime).model_dump(mode="json"),
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
    if bound.score != 0.0 or bound.verifier_exit_code != 0 or bound.annotation is not None:
        raise ValueError(
            f"nop control for {task_id} did not fail closed: score={bound.score} "
            f"exit={bound.verifier_exit_code} annotation={bound.annotation}"
        )
    return {"task_id": task_id, "receipt": bound.model_dump(mode="json"), "score": bound.score}


def run_controls(output_root: str, workdir: Path, sample: int, concurrency: int) -> dict:
    """Run the NoP controls over the seeded release sample and bind every receipt."""
    root = StoragePath(output_root)
    workdir.mkdir(parents=True, exist_ok=True)
    environment_kwargs = local_environment_kwargs(workdir)
    rows = release_rows(workdir)
    selected = seeded_control_tasks(rows, sample)
    logger.info(
        "Selected %d control tasks of %d release rows; local kwargs: %s",
        len(selected),
        len(rows),
        sorted(environment_kwargs),
    )
    archive_root = str(root / "archives")
    receipts = {}
    failures = {}

    def drive(entry: tuple[TaskSpec, dict, str]) -> dict:
        task, runtime, _encoded = entry
        requirements = ExecutionRequirements.model_validate(
            {**declared_execution_limits(task), "storage_mb": TRIAL_STORAGE_MB}
        )
        package = workdir / "packages" / canonical_sha256(task.id)
        if package.exists():
            shutil.rmtree(package)
        write_harbor_task(task, package, requirements)
        task_checksum = trial_package_checksum(package)
        source_task_checksum = original_package_checksum(task, workdir)
        with tempfile.TemporaryDirectory(prefix="mimo-nop-") as scratch:
            control_root = Path(scratch) / "control"
            request = control_request(
                task,
                runtime,
                package,
                requirements,
                task_checksum,
                source_task_checksum,
                control_root,
                environment_kwargs,
            )
            return run_control(request, control_root, archive_root)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {pool.submit(drive, entry): entry[0].id for entry in selected}
        for future in as_completed(pending):
            task_id = pending[future]
            try:
                receipts[task_id] = future.result()
            except Exception as error:  # one failed control fails the stage below
                failures[task_id] = f"{type(error).__name__}: {error}"

    if failures:
        raise ValueError(f"NoP controls failed for {len(failures)} tasks: {failures}")
    summary = {
        "source_id": "FWeindel:validated-tasks/mimo_v1",
        "release": {"repo": RELEASE_REPO, "revision": RELEASE_REVISION},
        "kind": "nop",
        "backend": "local",
        "local_kwargs": sorted(environment_kwargs),
        "selected": len(selected),
        "completed": len(receipts),
        "archive_root": archive_root,
        "receipts": {task_id: row["receipt"] for task_id, row in receipts.items()},
    }
    (root / "controls-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("NoP controls complete: %d/%d", len(receipts), len(selected))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--workdir", default="/workspace/mimo-controls")
    args = parser.parse_args()
    configure_coreweave_s3()
    run_controls(args.output_root, Path(args.workdir), args.sample, args.concurrency)


if __name__ == "__main__":
    main()
