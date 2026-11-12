# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard the SWE bug/reference CPU admission across privileged Iris Docker workers."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fray.client import wait_all
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.rpc import job_pb2
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.timing import Duration
from taskcompendium.models import TaskSpec

from experiments.post_training.long_loop.admission_index import finalize_admission
from experiments.post_training.long_loop.execution_policy import DEFAULT_TASK_TIMEOUT_SECONDS
from experiments.post_training.long_loop.pi_rl_data import SOURCE, image_context_digest
from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.swe_admission import build_image, run_controls
from experiments.post_training.long_loop.swe_verified_data import CURRENT_GRADER, RECIPE_VERSION, RUNTIME, prepare_task


def publish_directory(directory: Path, destination: str) -> None:
    """Publish bounded task evidence and a content inventory after every task."""
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            copy(str(path), str(StoragePath(destination) / relative))
            inventory[relative] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    (StoragePath(destination) / "evidence-inventory.json").write_text(json.dumps(inventory, indent=2))


def run_shard(
    shard_uri: str, shard_sha256: str, output: str, setup_timeout: int, verifier_timeout: int, harbor_timeout: int
) -> None:
    """Execute the same admission controls as local Docker, on a CPU-only Iris pod."""
    configure_coreweave_s3()
    # Docker owns mounts below this path. The task pod reclaims scratch after
    # exit; Python rmtree must not obscure a build error with an EBUSY cleanup.
    with tempfile.TemporaryDirectory(prefix="swe-admission-", delete=False) as scratch:
        root = Path(scratch)
        source = root / "tasks.parquet"
        copy(shard_uri, str(source))
        if file_digest(source) != shard_sha256:
            raise ValueError("Admission shard changed after submission")
        with (root / "docker-install.log").open("wb") as log:
            subprocess.run(
                ["apt-get", "update"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=DEFAULT_TASK_TIMEOUT_SECONDS,
            )
            subprocess.run(
                ["apt-get", "install", "-y", "docker.io", "slirp4netns"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=DEFAULT_TASK_TIMEOUT_SECONDS,
            )
        socket = root / "docker.sock"
        os.environ["DOCKER_HOST"] = f"unix://{socket}"
        with (root / "dockerd.log").open("wb") as log:
            daemon = subprocess.Popen(
                [
                    "dockerd",
                    "--bridge=none",
                    "--iptables=false",
                    "--ip-forward=false",
                    "--ip-masq=false",
                    "--host",
                    os.environ["DOCKER_HOST"],
                    "--data-root",
                    str(root / "docker-data"),
                    "--exec-root",
                    str(root / "docker-exec"),
                    "--pidfile",
                    str(root / "dockerd.pid"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                ready_deadline = time.monotonic() + DEFAULT_TASK_TIMEOUT_SECONDS
                while subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
                    if daemon.poll() is not None or time.monotonic() > ready_deadline:
                        raise RuntimeError("The private in-pod Docker daemon did not become ready")
                    time.sleep(1)
                rows = pq.read_table(source).to_pylist()
                prepared = [
                    prepare_task(
                        TaskSpec.model_validate_json(row["task_spec"]),
                        setup_timeout=setup_timeout,
                        verifier_timeout=verifier_timeout,
                        harbor_verifier_timeout=harbor_timeout,
                        dependency_constraints="",
                        profile=CURRENT_GRADER,
                    )
                    for row in rows
                ]
                contexts = {image_context_digest(item.task) for item in prepared}
                if len(contexts) != 1:
                    raise ValueError("An admission shard must preserve one original image context")
                image = build_image(prepared[0], root, network="host")
                (StoragePath(output) / "build-ready.json").write_text(json.dumps({"image": image, "at": time.time()}))
                results = []
                # Iris may give the worker its node's network namespace. Task
                # containers use isolated namespaces with userspace forwarding.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    futures = {
                        pool.submit(
                            run_controls,
                            item,
                            image,
                            root / "tasks" / hashlib.sha256(item.task.id.encode()).hexdigest(),
                            network="slirp4netns",
                        ): item
                        for item in prepared
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        key = hashlib.sha256(result["task_id"].encode()).hexdigest()
                        publish_directory(root / "tasks" / key, str(StoragePath(output) / "tasks" / key))
                        results.append(
                            {"task_id": result["task_id"], "status": result["status"], "elapsed": result["elapsed"]}
                        )
                        (StoragePath(output) / "progress.json").write_text(json.dumps(results, indent=2))
                        print(json.dumps(results[-1]), flush=True)
                (StoragePath(output) / "complete.json").write_text(
                    json.dumps({"shard_sha256": shard_sha256, "results": results}, indent=2)
                )
            finally:
                for build_log in root.glob("images/*/logs/*.json"):
                    copy(
                        str(build_log),
                        str(StoragePath(output) / "build-logs" / build_log.parent.parent.name / build_log.name),
                    )
                daemon.terminate()
                try:
                    daemon.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait()
                copy(str(root / "dockerd.log"), str(StoragePath(output) / "dockerd.log"))
                copy(str(root / "docker-install.log"), str(StoragePath(output) / "docker-install.log"))


def admission_job(name: str, entrypoint: Entrypoint) -> JobRequest:
    """Construct the actual CPU worker request, including the native Fray profile."""
    return JobRequest(
        name=name,
        entrypoint=entrypoint,
        resources=ResourceConfig.with_cpu(
            cpu=2,
            ram="6g",
            disk="96g",
            container_profile=job_pb2.ContainerProfile.Name(job_pb2.CONTAINER_PROFILE_PRIVILEGED),
        ),
        environment=create_environment(sync_packages=["marin-core"]),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        timeout=Duration.from_seconds(DEFAULT_TASK_TIMEOUT_SECONDS),
        max_retries_failure=0,
        max_retries_preemption=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--job-prefix", required=True)
    parser.add_argument("--tasks-per-shard", type=int, required=True)
    parser.add_argument("--setup-timeout", type=int, required=True)
    parser.add_argument("--verifier-timeout", type=int, required=True)
    parser.add_argument("--harbor-verifier-timeout", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    source_prefix, output = StoragePath(args.input_prefix), StoragePath(args.output_prefix)
    groups = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="swe-plan-") as scratch:
        root = Path(scratch)
        copy(str(source_prefix / "manifest.json"), str(root / "manifest.json"))
        if file_digest(root / "manifest.json") != args.manifest_sha256:
            raise ValueError("The pinned source manifest changed")
        manifest = json.loads((root / "manifest.json").read_text())
        runtime_uri = str(output / "grader.py")
        StoragePath(runtime_uri).write_bytes(CURRENT_GRADER.runtime)
        profiles = {file_digest(RUNTIME): {"uri": runtime_uri, "version": RECIPE_VERSION}}
        failures = []
        for name in ("agentic-train.parquet", "validation.parquet"):
            local = root / name
            copy(str(source_prefix / name), str(local))
            if file_digest(local) != manifest["files"][name]["sha256"]:
                raise ValueError("The pinned source parquet changed")
            for batch in pq.ParquetFile(local).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    task = TaskSpec.model_validate_json(row["task_spec"])
                    if task.source.dataset != SOURCE:
                        continue
                    try:
                        prepared = prepare_task(
                            task,
                            setup_timeout=args.setup_timeout,
                            verifier_timeout=args.verifier_timeout,
                            harbor_verifier_timeout=args.harbor_verifier_timeout,
                            dependency_constraints="",
                            profile=CURRENT_GRADER,
                        )
                    except ValueError as error:
                        failures.append({"task_id": task.id, "reason": str(error)})
                    else:
                        groups[(image_context_digest(prepared.task), prepared.pins.repository)].append(row)
        (output / "input-exclusions.json").write_text(json.dumps(failures, indent=2))
        plan = []
        handles = []
        client = current_client()
        for group, rows in sorted(groups.items()):
            for start in range(0, len(rows), args.tasks_per_shard):
                selected = rows[start : start + args.tasks_per_shard]
                index = len(plan)
                local = root / f"shard-{index:04d}.parquet"
                pq.write_table(pa.Table.from_pylist(selected), local, compression="zstd")
                shard_uri = str(output / "inputs" / local.name)
                copy(str(local), shard_uri)
                shard_hash = file_digest(local)
                request = admission_job(
                    f"{args.job_prefix}-{index:04d}",
                    Entrypoint.from_callable(
                        run_shard,
                        args=[
                            shard_uri,
                            shard_hash,
                            str(output / "shards" / f"{index:04d}"),
                            args.setup_timeout,
                            args.verifier_timeout,
                            args.harbor_verifier_timeout,
                        ],
                    ),
                )
                intent = {
                    "index": index,
                    "context": group[0],
                    "repository": group[1],
                    "rows": len(selected),
                    "task_ids": [json.loads(row["task_spec"])["id"] for row in selected],
                    "input": shard_uri,
                    "sha256": shard_hash,
                    "job_name": request.name,
                }
                (output / "intents" / f"{index:04d}.json").write_text(json.dumps(intent, indent=2))
                job = client.submit(request, adopt_existing=False)
                handles.append(job)
                plan.append({**intent, "job_id": job.job_id})
                (output / "plan.json").write_text(
                    json.dumps(
                        {
                            "manifest_sha256": args.manifest_sha256,
                            "runtime_sha256": file_digest(RUNTIME),
                            "shards": plan,
                            "submitted": False,
                        },
                        indent=2,
                    )
                )
                print(json.dumps({"job_id": job.job_id, "rows": len(selected)}), flush=True)
        (output / "plan.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": args.manifest_sha256,
                    "runtime_sha256": file_digest(RUNTIME),
                    "shards": plan,
                    "submitted": True,
                },
                indent=2,
            )
        )
        # The parent stays alive to retain its child jobs; it never retries a
        # failed shard or silently replaces its receipts.
        statuses = wait_all(handles, raise_on_failure=False)
        (output / "terminal.json").write_text(
            json.dumps({entry["job_id"]: status for entry, status in zip(plan, statuses, strict=True)}, indent=2)
        )
        if any(status != "succeeded" for status in statuses):
            raise RuntimeError("Some CPU admission shards failed; inspect terminal.json before recovery")
        finalize_admission(
            [args.output_prefix], args.output_prefix, workers=32, profiles=profiles, replacement_sources=()
        )


if __name__ == "__main__":
    main()
