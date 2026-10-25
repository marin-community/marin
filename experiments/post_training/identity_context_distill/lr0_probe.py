# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind and submit the #310 LR0 precision probe: one finite 4 x H100x8 RNO2a job, no retries.

``lr0_worker`` runs on every replica. It builds the source-derived recipe with the measured router policy and
calls ``precision_probe.run_precision_probe`` with the reviewed callbacks:

- source audit: ``precision_source_audit.audit_precision_source`` (502 tensors) -> ``audits/source-tensor-hashes.json``
- master audit: ``precision_audit.audit_initialized_masters`` at steps 0 and 1 -> ``audits/master-step-<n>.json``
- pre-update gate: ``precision_gate.make_pre_update_gate`` -> ``precision-gate/``
- export audit: ``precision_audit.audit_precision_export`` -> ``hf/lr0-step-0/precision-audit.json``

Process 0 writes ``lr0-result.json`` only after the probe returns. Any failure raises and fails the job; a gate
deadline exits with status 4. The probe never reads a dataset: LR0 uses its own synthetic batch.

``main`` prints the job plan; with ``--run`` it submits. The launch owner runs the capacity guard first and passes its
record (four wholly idle cw-rno2a H100x8 nodes, with its observation time) as ``--capacity-record``. ``--run`` writes
``--launch-receipt`` before the submit and again with the Iris job id after it.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.tracker import NoopConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import s3_credential_options
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.precision_gate import PrecisionGateInputs, make_pre_update_gate
from experiments.post_training.identity_context_distill.precision_audit import (
    audit_initialized_masters,
    audit_precision_export,
)
from experiments.post_training.identity_context_distill.precision_probe import run_precision_probe
from experiments.post_training.identity_context_distill.precision_source_audit import audit_precision_source
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
from experiments.post_training.identity_context_distill.sft import BASE_MODEL, identity_sft_config

T310 = "s3://marin-rno2a/marin/experiments/snowball-identity/t310"
INPUTS = f"{T310}/inputs-20261024-0252"
RUNTIME_RECEIPT_SHA256 = "33b9a1682121332fe0b5d9d8e90c2ab409eff6a83fd98d40715523efac144191"
SOURCE_FILE_HASHES = {
    "config.json": "c7e1e7ac08e7b8b712d2728596765d2a5f954a7372ba1090cf87709e5fa9fbd5",
    "generation_config.json": "fef404c3c02c0c5120a18a55820caac75222b507f27e1aef0e4cd482ba887d73",
    "chat_template.jinja": "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de",
    "model.safetensors.index.json": "7737eed0ad5faa748e709ae9fb7dfb1baded1e5a18b2d811cf13f69776b4869d",
    "tokenizer.json": "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45",
    "tokenizer_config.json": "7c11be094d34bd4bb5cf704584fb71b1164220bcfcc5963321bc6ec03d7ec853",
}
SEED = 310
REPLICAS = 4
MAX_TIMEOUT_SECONDS = 10800
PRIORITY_BANDS = {
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
}
RESOURCES = ResourceConfig.with_gpu(
    "H100", count=8, cpu=64, ram="768GB", disk="1TB", target_cluster="cw-rno2a", replicas=REPLICAS
)


def check_pod_storage() -> None:
    """Every rank, before any read: plain ``StoragePath`` needs the pod's endpoint and the AWS-named key pair."""
    missing = [name for name in ("AWS_ENDPOINT_URL", "FSSPEC_S3") if not os.environ.get(name)]
    credentials = s3_credential_options()
    if not (credentials.get("key") and credentials.get("secret")):
        missing.append("AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
    if missing:
        raise RuntimeError(f"Pod storage settings missing: {missing}")


def approved_receipt(receipt_bytes: bytes) -> dict:
    if hashlib.sha256(receipt_bytes).hexdigest() != RUNTIME_RECEIPT_SHA256:
        raise ValueError("Runtime receipt differs from the reviewed derived receipt")
    return json.loads(receipt_bytes)


def gate_inputs(run_root: str, routing_path: str) -> PrecisionGateInputs:
    return PrecisionGateInputs(
        positions=f"{INPUTS}/positions.pt",
        vllm=f"{INPUTS}/vllm-positions-20261024-0252.json",
        identity=f"{INPUTS}/identity-4569cb3a.jsonl",
        tokenizer_source=BASE_MODEL,
        torch_routing=routing_path.rsplit("/", 1)[0],
        output=f"{run_root}/precision-gate",
        batch=8,
        pad_multiple=64,
        deadline_seconds=1800,
    )


def lr0_worker(run_root: str, run_id: str, receipt_bytes: bytes) -> None:
    """Run on every replica; ``receipt_bytes`` are the reviewed runtime receipt's exact bytes."""
    check_pod_storage()
    runtime_receipt = approved_receipt(receipt_bytes)
    reference = runtime_receipt["routing_reference"]
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    unused = f"{run_root}/no-dataset-lr0-uses-a-synthetic-batch"
    base = identity_sft_config(
        dataset_uri=unused,
        cache_uri=unused,
        output_uri=run_root,
        run_id=run_id,
        seed=SEED,
        source_config=json.loads(config_bytes),
        source_generation_config=json.loads(generation_bytes),
    )
    config, observed = configure_precision_runtime(
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
    )
    if observed != runtime_receipt:
        raise ValueError("Derived runtime recipe differs from the reviewed receipt")
    config = replace(config, trainer=replace(config.trainer, tracker=NoopConfig()))
    source_receipt = f"{run_root}/audits/source-tensor-hashes.json"

    def master_audit(state, completed_updates: int) -> None:
        if completed_updates not in (0, 1):
            raise ValueError(f"Unexpected LR0 master audit step {completed_updates}")
        receipt = f"{run_root}/audits/master-step-{completed_updates}.json"
        audit_initialized_masters(state.model.to_state_dict(), source_receipt, receipt)

    result = run_precision_probe(
        config,
        expected_source_receipt=runtime_receipt["source_metadata"],
        expected_runtime_receipt=runtime_receipt,
        source_audit=lambda root: audit_precision_source(root, SOURCE_FILE_HASHES, source_receipt),
        master_audit=master_audit,
        pre_update_gate=make_pre_update_gate(gate_inputs(run_root, reference["path"])),
        export_audit=lambda source, export: audit_precision_export(source, export, SOURCE_FILE_HASHES),
    )
    # Collective: every rank calls it; process 0 writes.
    run_on_export_writer(lambda: write_export_json(StoragePath(run_root) / "lr0-result.json", result))


def plan(job_name: str, run_root: str, timeout_seconds: int, routing_path: str, priority: str) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    return {
        "job": job_name,
        "commit": head,
        "clean": not dirty,
        "run_root": run_root,
        "gate": vars(gate_inputs(run_root, routing_path)),
        "runtime_receipt_sha256": RUNTIME_RECEIPT_SHA256,
        "region": RESOURCES.target_cluster,
        "replicas": REPLICAS,
        "gpus": 8 * REPLICAS,
        "resources": {"cpu": RESOURCES.cpu, "ram": RESOURCES.ram, "disk": RESOURCES.disk},
        "priority": priority,
        "ports": ["jax"],
        "secret_behavior": "isolate: CW_KEY_ID/SECRET, also as AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY",
        "retries_failure": 0,
        "retries_preemption": 0,
        "timeout_seconds": timeout_seconds,
        "h100_hour_ceiling": 8 * REPLICAS * timeout_seconds / 3600,
    }


def submit(job_name: str, run_root: str, run_id: str, receipt_bytes: bytes, timeout_seconds: int, priority: str) -> str:
    key, secret = os.environ["CW_KEY_ID"], os.environ["CW_KEY_SECRET"]
    # StoragePath reads the AWS-named pair; bucket-routed filesystems read the CW names.
    secrets = {"CW_KEY_ID": key, "CW_KEY_SECRET": secret, "AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    environment = EnvironmentSpec(extras=["gpu"], secrets=secrets, secret_behavior=SecretBehavior.ISOLATE)
    with connect_controller(cluster_name="marin") as info:
        with IrisClient.remote(info.url, workspace=Path.cwd(), credentials=info.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(lr0_worker, run_root, run_id, receipt_bytes),
                name=job_name,
                resources=convert_resources(RESOURCES),
                environment=environment,
                # Iris allocates the JAX coordinator port and exports it as IRIS_PORT_JAX on every replica.
                ports=["jax"],
                constraints=convert_constraints(RESOURCES) or None,
                coscheduling=resolve_coscheduling(RESOURCES, REPLICAS),
                replicas=REPLICAS,
                timeout=Duration.from_seconds(timeout_seconds),
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_band=PRIORITY_BANDS[priority],
            )
            return str(job.job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-root", required=True, help="fresh s3://marin-rno2a/ prefix; must not exist")
    parser.add_argument("--runtime-receipt", required=True, help="local runtime-recipe-f37aec3.json")
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--allow-long-timeout", action="store_true", help=f"permit more than {MAX_TIMEOUT_SECONDS} s")
    parser.add_argument("--expect-commit", help="the reviewed worker pin; required with --run, must equal HEAD")
    parser.add_argument("--priority", choices=sorted(PRIORITY_BANDS), default="interactive")
    parser.add_argument("--capacity-record", help="the launch owner's capacity guard JSON; required with --run")
    parser.add_argument("--launch-receipt", help="local JSON receipt to write; required with --run")
    parser.add_argument("--run", action="store_true", help="submit; without it only print the plan")
    args = parser.parse_args()
    run_root = args.run_root.rstrip("/")
    if not run_root.startswith("s3://marin-rno2a/"):
        raise SystemExit("run root must be on s3://marin-rno2a/")
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be a positive number of seconds")
    if args.timeout_seconds > MAX_TIMEOUT_SECONDS and not args.allow_long_timeout:
        raise SystemExit(f"timeout above {MAX_TIMEOUT_SECONDS} s needs --allow-long-timeout")
    receipt_bytes = Path(args.runtime_receipt).read_bytes()
    receipt = approved_receipt(receipt_bytes)
    job_plan = plan(args.job_name, run_root, args.timeout_seconds, receipt["routing_reference"]["path"], args.priority)
    print(json.dumps(job_plan, indent=2))
    if not args.run:
        return
    if not job_plan["clean"]:
        raise SystemExit("worktree is not clean")
    if not args.capacity_record or not args.launch_receipt or not args.expect_commit:
        raise SystemExit("--run needs --capacity-record, --launch-receipt and --expect-commit")
    if job_plan["commit"] != args.expect_commit:
        raise SystemExit(f"HEAD {job_plan['commit']} is not the reviewed pin {args.expect_commit}")
    fs, root = filesystem_for(run_root)
    if fs.exists(root):
        raise SystemExit(f"run root already exists: {run_root}")
    capacity = Path(args.capacity_record).read_bytes()
    launch = {
        **job_plan,
        "state": "pre-submit",
        "submit_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "capacity_record": {
            "path": args.capacity_record,
            "sha256": hashlib.sha256(capacity).hexdigest(),
            "content": json.loads(capacity),
        },
    }
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=2) + "\n")
    job_id = submit(args.job_name, run_root, args.job_name, receipt_bytes, args.timeout_seconds, args.priority)
    launch.update(state="submitted", iris_job_id=job_id)
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps({"job": job_id, "status": "submitted"}))


if __name__ == "__main__":
    main()
