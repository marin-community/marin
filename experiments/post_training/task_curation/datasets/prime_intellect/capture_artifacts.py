# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Multi-SWE's image-resident private grading files before task conversion."""

import argparse
import asyncio
import hashlib
import io
import json
import logging
import tarfile
import uuid
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from finestore.rl.licenses import LicenseReceipt, require_license
from iris.cli.connect import connect_controller
from iris.cluster.client.job_info import get_job_info
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.rpc import job_pb2
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.cluster_config import data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy

from experiments.post_training.task_curation.datasets.prime_intellect.capture_models import (
    CaptureConfig,
    CapturedArtifacts,
    CaptureInput,
    ImageBinding,
)
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.local import artifact_prefix, local_client

LOGGER = logging.getLogger(__name__)
DATASET = "PrimeIntellect/Multi-SWE-RL-Verified"
REVISION = "64b3ebf62750b1373e26106cfc658a1861809d0f"
SOURCE_ID = "PrimeIntellect:" + DATASET
SOURCE_TASKS = 2232
TASKSET_REVISION = "b41d8759070715dc102000e06fba0d73ebce9955"
REMOTE_ARCHIVE = "/tmp/multiswe_grading_artifacts.tar.gz"
CAPTURE_COMMAND = """set -euo pipefail
test -f /home/fix-run.sh
test -f /home/test.patch
rm -f /tmp/multiswe_grading_artifacts.tar.gz
manifest="$(mktemp)"
find /home -maxdepth 1 -type f \\( -name '*.sh' -o -name 'test.patch' \\) \
  ! -name 'fix.patch' ! -name 'create_fix_patch.sh' ! -name 'extract_multiswe_fix_patch.sh' \
  -printf '%P\\0' > "$manifest"
tar --null -C /home -czf /tmp/multiswe_grading_artifacts.tar.gz -T "$manifest"
rm -f "$manifest"
sha256sum /tmp/multiswe_grading_artifacts.tar.gz
"""


def archive_members(data: bytes) -> list[dict]:
    members = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in {"", ".", ".."} or "/" in member.name:
                raise ValueError(f"Unexpected private grading archive member: {member.name}")
            content = archive.extractfile(member)
            assert content is not None
            members.append(
                {
                    "name": member.name,
                    "mode": member.mode,
                    "mtime": member.mtime,
                    "bytes": member.size,
                    "sha256": hashlib.sha256(content.read()).hexdigest(),
                }
            )
    names = [member["name"] for member in members]
    if len(names) != len(set(names)) or not {"fix-run.sh", "test.patch"}.issubset(names):
        raise ValueError("Private grading archive lacks its source harness or contains repeated members")
    return members


async def capture_one(binding: ImageBinding, config: CaptureConfig, factory: IrisMachineFactory) -> dict:
    output = StoragePath(config.output_path)
    key = hashlib.sha256(binding.pinned_reference.encode()).hexdigest()
    receipt_path = output / "receipts" / f"{key}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        data = StoragePath(receipt["archive_uri"]).read_bytes()
        if receipt["binding"] != binding.model_dump(mode="json"):
            raise ValueError("Completed capture belongs to another image/task binding")
        if hashlib.sha256(data).hexdigest() != receipt["archive_sha256"] or archive_members(data) != receipt["members"]:
            raise ValueError("Completed private capture bytes differ from their receipt")
        return receipt
    machine = await factory.create(
        MachineSpec(
            source=RegistryImage(binding.pinned_reference),
            workdir="/home",
            cpus=1,
            memory_mb=2048,
            # Metadata acquisition requests a job budget without claiming task storage enforcement.
            storage_mb=None,
            network=NetworkPolicy.DENY,
        )
    )
    try:
        result = await machine.run(Command(("bash", "-c", CAPTURE_COMMAND), timeout=180))
        if result.exit_code != 0 or result.stdout_truncated:
            diagnostic = output / "failed-commands" / uuid.uuid4().hex
            (diagnostic / "stdout.bin").write_bytes(result.stdout, auto_mkdir=True)
            (diagnostic / "stderr.bin").write_bytes(result.stderr)
            raise RuntimeError(f"Private artifact capture failed: exit {result.exit_code}; raw output at {diagnostic}")
        expected = result.stdout.decode().strip().split()[0]
        with TemporaryDirectory(prefix="multiswe-private-capture-") as scratch:
            local_archive = Path(scratch) / "grading.tar.gz"
            await machine.download(REMOTE_ARCHIVE, local_archive)
            data = local_archive.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError("Private archive transfer differs from the source-machine SHA256")
        members = archive_members(data)
        destination = output / "archives" / f"{digest}.tar.gz"
        receipt = {
            "binding": binding.model_dump(mode="json"),
            "task_id": str(machine.task.task_id),
            "cluster": config.cluster,
            "profile": "SANDBOX",
            "job_storage_request_mb": factory.disk_mb,
            "execution_qualification": False,
            "archive_uri": str(destination),
            "archive_sha256": digest,
            "archive_bytes": len(data),
            "members": members,
            "taskset_revision": TASKSET_REVISION,
        }
        destination.write_bytes(data, auto_mkdir=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", auto_mkdir=True)
        LOGGER.info("Captured %s: %d bytes, %d members", binding.pinned_reference, len(data), len(members))
        return receipt
    finally:
        await machine.close()


async def capture_all(inputs: CaptureInput, config: CaptureConfig) -> list[dict]:
    job = get_job_info()
    if job is None or job.controller_address is None:
        raise ValueError("Full image capture requires a trusted regional Iris CPU parent")
    factory = IrisMachineFactory(
        controller_url=job.controller_address,
        job_ttl=259200,
        scheduling_timeout=900,
        disk_mb=10240,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        secret_env=None,
    )
    semaphore = asyncio.Semaphore(config.concurrency)

    async def capture(binding: ImageBinding) -> dict:
        async with semaphore:
            try:
                return await capture_one(binding, config, factory)
            except Exception as error:
                path = StoragePath(config.output_path) / "failures" / f"{uuid.uuid4().hex}.json"
                path.write_text(
                    json.dumps({"binding": binding.model_dump(mode="json"), "error": str(error)}) + "\n",
                    auto_mkdir=True,
                )
                LOGGER.exception("Private capture failed for %s", binding.pinned_reference)
                raise

    async with asyncio.timeout(259200):
        results = await asyncio.gather(*(capture(binding) for binding in inputs.images), return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} source-image captures failed; retained partial evidence") from failures[0]
    return cast(list[dict], results)


def capture_artifacts(config: CaptureConfig) -> CapturedArtifacts:
    require_license(LicenseReceipt.model_validate_json(config.license_receipt_json), SOURCE_ID, "train", ban=None)
    job = get_job_info()
    output = StoragePath(config.output_path)
    bucket = data_buckets().get(output.bucket)
    if job is None or not job.controller_address or output.scheme != "s3" or bucket is None:
        raise ValueError("Full capture must write directly from a regional Iris CPU job to its same-region S3 bucket")
    endpoint = connect_controller(controller_url=job.controller_address)
    client = RemoteClusterClient(
        endpoint.url,
        interceptors=endpoint.credentials.interceptors() if endpoint.credentials is not None else (),
    )
    task = client.get_task_status(job.task_id)
    attempts = [attempt for attempt in task.attempts if attempt.attempt_id == job.attempt_id]
    if task.cluster != "local" or len(attempts) != 1 or not attempts[0].node_name or not attempts[0].pod_name:
        raise ValueError("Capture placement requires the current local task attempt's pod and node")
    attempt = attempts[0]
    backends = [
        backend
        for backend in client.list_backends()
        if backend.kind == "kubernetes"
        and any(node.name == attempt.node_name for node in backend.detail.kubernetes.nodes)
    ]
    if len(backends) != 1:
        raise ValueError("Capture placement must resolve to one local controller backend")
    backend = backends[0]
    regions = tuple(backend.advertised_attributes["region"].values)
    if (
        backend.name != config.cluster
        or len(regions) != 1
        or bucket.signing_region is None
        or regions[0].casefold() != bucket.signing_region.casefold()
    ):
        raise ValueError("Controller-advertised capture region does not match the requested cluster and S3 bucket")
    placement = {
        "task_id": job.task_id.to_wire(),
        "attempt_id": job.attempt_id,
        "attempt_uid": attempt.attempt_uid,
        "pod_name": attempt.pod_name,
        "node_name": attempt.node_name,
        "backend_id": backend.backend_id,
        "backend_name": backend.name,
        "region": regions[0],
        "bucket": output.bucket,
        "bucket_region": bucket.signing_region,
        "evidence": "Current task and backend records from its submitting Iris controller",
    }
    placement_key = hashlib.sha256(f"{job.task_id.to_wire()}:{attempt.attempt_uid}".encode()).hexdigest()
    placement_path = output / "placements" / placement_key / "placement.json"
    encoded_placement = json.dumps(placement, indent=2) + "\n"
    if placement_path.exists() and placement_path.read_text() != encoded_placement:
        raise ValueError("Capture attempt already has different placement evidence")
    if not placement_path.exists():
        placement_path.write_text(encoded_placement, auto_mkdir=True)
    if not (output / "placement.json").exists():
        (output / "placement.json").write_text(encoded_placement, auto_mkdir=True)
    raw = StoragePath(config.input_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.input_sha256:
        raise ValueError("Frozen source image index changed")
    inputs = CaptureInput.model_validate_json(raw)
    if (inputs.source_id, inputs.dataset_id, inputs.revision, inputs.taskset_revision) != (
        SOURCE_ID,
        DATASET,
        REVISION,
        TASKSET_REVISION,
    ):
        raise ValueError("Image index belongs to another source/taskset revision")
    references = [binding.pinned_reference for binding in inputs.images]
    task_ids = [task_id for binding in inputs.images for task_id in binding.task_ids]
    if len(references) != len(set(references)) or len(task_ids) != len(set(task_ids)) or len(task_ids) != SOURCE_TASKS:
        raise ValueError("Source image index repeats images/tasks or does not cover the complete pinned source")
    captures = asyncio.run(capture_all(inputs, config))
    manifest = json.dumps(captures, indent=2).encode() + b"\n"
    destination = StoragePath(config.output_path) / "captures.json"
    destination.write_bytes(manifest, auto_mkdir=True)
    return CapturedArtifacts(
        source_id=SOURCE_ID,
        dataset_id=DATASET,
        revision=REVISION,
        input_sha256=config.input_sha256,
        images=len(captures),
        tasks=len(task_ids),
        manifest_uri=str(destination),
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        placement_uri=str(placement_path),
    )


def output_config(config: CaptureConfig, context: StepContext) -> CaptureConfig:
    return replace(config, output_path=context.output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--images-sha256", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--license-receipts", type=Path, default=LICENSE_RECEIPT_INDEX)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    receipt = read_license_receipts(args.license_receipts).get(SOURCE_ID)
    require_license(receipt, SOURCE_ID, "train", ban=None)
    assert receipt is not None
    if args.concurrency <= 0:
        raise ValueError("Capture concurrency must be positive")
    configure_coreweave_s3()
    config = CaptureConfig(
        args.images, args.images_sha256, args.cluster, args.concurrency, receipt.model_dump_json(), ""
    )
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/atlas/prime-intellect/multiswe-private-artifacts",
            version=args.artifact_version,
            artifact_type=CapturedArtifacts,
            run=capture_artifacts,
            build_config=partial(output_config, config),
        )
        if not args.run:
            args.output.write_text(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}) + "\n")
            return
        with local_client():
            result = run(step)[0]
        args.output.write_text(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
