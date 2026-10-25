# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind and submit the #310 LR0 precision probe: one finite 4 x H100x8 RNO2a job, no retries.

``lr0_worker`` runs on every replica. It builds the source-derived recipe with the measured router policy and
calls ``precision_probe.run_precision_probe`` with the reviewed callbacks:

- source audit: ``precision_source_audit.audit_precision_source`` (502 tensors) -> ``audits/source-tensor-hashes.json``
- master audit: ``precision_audit.audit_initialized_masters`` at steps 0 and 1 -> ``audits/master-step-<n>.json``
- pre-update gate: ``precision_gate.make_pre_update_gate`` -> ``precision-gate/``
- post-update memory: every rank's devices after the update -> ``lr0-result.json`` ``device_memory.after_update``
  and stage ``updated`` of ``precision-gate/ranks/<process>.json``
- export audit: ``precision_audit.audit_precision_export`` -> ``hf/lr0-step-0/precision-audit.json``

Process 0 writes ``lr0-result.json`` only after the probe returns. Any failure raises and fails the job; a gate
deadline exits with status 4. The probe never reads a dataset: LR0 uses its own synthetic batch.

The runtime receipt is bound by ``--runtime-receipt-sha256``: the launcher and every worker check the receipt's
bytes against it, and the worker re-derives the recipe (capacity factor and FP32 combine from the receipt) and
requires it to equal the receipt.

``main`` prints the job plan; with ``--run`` it submits. The launch owner runs the capacity guard first and passes its
record (four wholly idle cw-rno2a H100x8 nodes, with its observation time) as ``--capacity-record``. ``--run`` writes
``--launch-receipt`` before the submit and again with the Iris job id after it.

``--io-smoke`` runs no job. From the submit host, under a scratch root that must not exist, it calls the worker's own
storage functions on real ``s3://marin-rno2a`` objects (the source shard opener, the receipt writer, the gate's
readers and writers), deletes only what it wrote, and writes ``--smoke-receipt``. With ``--emulate-pod-env`` it first
sets the task pod's storage variables from the CW keys, against the external endpoint; the pod's node-local endpoint
is not exercised.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import fsspec
import jax
import s3fs
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker import NoopConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import fsspec_s3_conf, s3_credential_options
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.levanter_positions import save_positions
from experiments.interp.snowball_identity.logit_check import read, stored_sequences, write_json
from experiments.interp.snowball_identity.precision_gate import (
    ROUTE_POLICIES,
    PrecisionGateInputs,
    make_pre_update_gate,
)
from experiments.interp.snowball_identity.prompts import load_tokenizer
from experiments.post_training.identity_context_distill import precision_source_audit
from experiments.post_training.identity_context_distill.precision_audit import (
    audit_initialized_masters,
    audit_precision_export,
)
from experiments.post_training.identity_context_distill.precision_probe import run_precision_probe
from experiments.post_training.identity_context_distill.precision_source_audit import audit_precision_source
from experiments.post_training.identity_context_distill.runtime_recipe import (
    configure_precision_runtime,
    require_xla_precision,
    xla_precision_env,
    xla_precision_in_process,
)
from experiments.post_training.identity_context_distill.sft import BASE_MODEL, identity_sft_config

logger = logging.getLogger(__name__)

T310 = "s3://marin-rno2a/marin/experiments/snowball-identity/t310"
INPUTS = f"{T310}/inputs-20261024-0252"
FLOOR = f"{T310}/diag-torch-floor-r1"
VLLM_CAPTURE = f"{T310}/vllm-route-capture-r1"
"""The torch floor job: G1's ``receipt.json`` and ``batch1/top6.pt``, each checked against its pinned sha256."""
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
"""Iris replicas = H100x8 hosts = JAX slices; the factory places context across them, as in the #302 SFT."""
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


def approved_receipt(receipt_bytes: bytes, sha256: str) -> dict:
    """The reviewed runtime receipt: its bytes must hash to the sha256 the launch owner passed."""
    actual = hashlib.sha256(receipt_bytes).hexdigest()
    if actual != sha256:
        raise ValueError(f"Runtime receipt sha256 {actual} differs from the reviewed receipt {sha256}")
    return json.loads(receipt_bytes)


def gate_inputs(run_root: str, routing_path: str, route_policy: str) -> PrecisionGateInputs:
    return PrecisionGateInputs(
        positions=f"{INPUTS}/positions.pt",
        vllm=f"{INPUTS}/vllm-positions-20261024-0252.json",
        identity=f"{INPUTS}/identity-4569cb3a.jsonl",
        tokenizer_source=BASE_MODEL,
        torch_routing=routing_path.rsplit("/", 1)[0],
        floor_receipt=f"{FLOOR}/receipt.json",
        torch_top6=f"{FLOOR}/batch1/top6.pt",
        torch_batched_top6=f"{FLOOR}/batched/top6.pt",
        route_policy=route_policy,
        vllm_capture=VLLM_CAPTURE,
        output=f"{run_root}/precision-gate",
        batch=8,
        pad_multiple=64,
        deadline_seconds=1800,
    )


def base_config(
    run_root: str, run_id: str, config_bytes: bytes, generation_bytes: bytes, receipt: dict
) -> TrainLmConfig:
    """The source-derived recipe with the receipt's runtime settings (capacity factor, FP32 combine, FP32
    activations, attention implementation); a receipt below schema 6 or lacking any of them is refused."""
    if receipt.get("schema_version") != 6:
        raise ValueError(f"Runtime receipt schema {receipt.get('schema_version')!r}; this probe needs 6")
    runtime = receipt["runtime_model_config"]
    names = ("moe_capacity_factor", "router_combine_fp32", "fp32_activations", "attention_implementation")
    missing = [name for name in names if name not in runtime]
    if missing:
        raise ValueError(f"Runtime receipt lacks {missing}")
    unused = f"{run_root}/no-dataset-lr0-uses-a-synthetic-batch"
    return identity_sft_config(
        dataset_uri=unused,
        cache_uri=unused,
        output_uri=run_root,
        run_id=run_id,
        seed=SEED,
        source_config=json.loads(config_bytes),
        source_generation_config=json.loads(generation_bytes),
        hosts=REPLICAS,
        moe_capacity_factor=runtime["moe_capacity_factor"],
        router_combine_fp32=runtime["router_combine_fp32"],
        fp32_activations=runtime["fp32_activations"],
        attention_implementation=runtime["attention_implementation"],
    )


def mesh_layout(mesh: MeshConfig) -> dict:
    return {
        "axes": dict(mesh.axes),
        "dcn_axes": dict(mesh.dcn_axes),
        "compute_mapping": dict(mesh.compute_mapping),
        "param_mapping": dict(mesh.param_mapping),
        "expected_num_slices": REPLICAS,
    }


def lr0_worker(run_root: str, run_id: str, receipt_bytes: bytes, receipt_sha256: str, route_policy: str) -> None:
    """Run on every replica; ``receipt_bytes`` are the reviewed runtime receipt's exact bytes."""
    runtime_receipt = approved_receipt(receipt_bytes, receipt_sha256)
    # Before anything touches a device: XLA reads XLA_FLAGS once, when its backend starts.
    require_xla_precision(runtime_receipt["xla_precision"])
    check_pod_storage()
    reference = runtime_receipt["routing_reference"]
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    base = base_config(run_root, run_id, config_bytes, generation_bytes, runtime_receipt)
    config, observed = configure_precision_runtime(
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
        xla_precision=xla_precision_in_process(),
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

    gate = gate_inputs(run_root, reference["path"], route_policy)

    def post_update_receipt(memory: list[dict]) -> None:
        """Every rank, after the update: its devices' memory in its rank receipt, stage ``updated`` (best effort)."""
        rank = jax.process_index()
        rows = [row for row in memory if row["process"] == rank]
        try:
            write_json(
                prefix_join(gate.output, f"ranks/{rank}.json"),
                {"process": rank, "stage": "updated", "device_memory_after_update": rows},
            )
        except Exception:
            logger.exception("rank receipt updated not written")

    result = run_precision_probe(
        config,
        expected_source_receipt=runtime_receipt["source_metadata"],
        expected_runtime_receipt=runtime_receipt,
        source_audit=lambda root: audit_precision_source(root, SOURCE_FILE_HASHES, source_receipt),
        master_audit=master_audit,
        pre_update_gate=make_pre_update_gate(gate),
        post_update_receipt=post_update_receipt,
        export_audit=lambda source, export: audit_precision_export(
            source, export, SOURCE_FILE_HASHES, receipt_path=str(StoragePath(export) / "precision-audit.json")
        ),
    )
    # Collective: every rank calls it; process 0 writes.
    run_on_export_writer(lambda: write_export_json(StoragePath(run_root) / "lr0-result.json", result))


def plan(
    job_name: str,
    run_root: str,
    timeout_seconds: int,
    routing_path: str,
    priority: str,
    mesh: MeshConfig,
    receipt_sha256: str,
    xla_precision: str,
    route_policy: str,
) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    return {
        "job": job_name,
        "commit": head,
        "clean": not dirty,
        "run_root": run_root,
        "gate": vars(gate_inputs(run_root, routing_path, route_policy)),
        "route_policy": route_policy,
        "mesh": mesh_layout(mesh),
        "runtime_receipt_sha256": receipt_sha256,
        "xla_precision": xla_precision,
        "env_vars": xla_precision_env(xla_precision),
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


def submit(
    job_name: str,
    run_root: str,
    run_id: str,
    receipt_bytes: bytes,
    receipt_sha256: str,
    route_policy: str,
    timeout_seconds: int,
    priority: str,
    xla_precision: str,
) -> str:
    entrypoint = Entrypoint.from_callable(lr0_worker, run_root, run_id, receipt_bytes, receipt_sha256, route_policy)
    return submit_entrypoint(job_name, entrypoint, timeout_seconds, priority)


def submit_entrypoint(job_name: str, entrypoint: Entrypoint, timeout_seconds: int, priority: str) -> str:
    """One finite 4 x H100x8 RNO2a job with no retries; shared by the LR0 and SFT launchers."""
    key, secret = os.environ["CW_KEY_ID"], os.environ["CW_KEY_SECRET"]
    # StoragePath reads the AWS-named pair; bucket-routed filesystems read the CW names.
    secrets = {"CW_KEY_ID": key, "CW_KEY_SECRET": secret, "AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    environment = EnvironmentSpec(
        extras=["gpu"],
        secrets=secrets,
        secret_behavior=SecretBehavior.ISOLATE,
        env_vars=xla_precision_env(xla_precision),
    )
    with connect_controller(cluster_name="marin") as info:
        with IrisClient.remote(info.url, workspace=Path.cwd(), credentials=info.credentials) as client:
            job = client.submit(
                entrypoint=entrypoint,
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


EXTERNAL_ENDPOINT = "https://cwobject.com"
SMOKE_BYTES = 1024 * 1024


def emulate_pod_env(endpoint: str = EXTERNAL_ENDPOINT) -> None:
    """The task pod's storage variables (k8s ``_build_s3_task_env``) and the job's AWS-named secrets, from CW keys."""
    os.environ["AWS_ENDPOINT_URL"] = endpoint
    os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = "auto"
    os.environ["FSSPEC_S3"] = json.dumps(fsspec_s3_conf(endpoint))
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CW_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CW_KEY_SECRET"]
    # A pod has these before Python starts; fsspec reads FSSPEC_S3 at import, so reload it here.
    fsspec.config.conf.clear()
    fsspec.config.set_conf_env(fsspec.config.conf)
    s3fs.S3FileSystem.clear_instance_cache()


def io_smoke(scratch_root: str, routing: dict) -> dict:
    """Call the worker's storage functions on real objects; every step runs and reports, and all must pass."""
    smoke = f"{scratch_root}/io-smoke"
    written: list[str] = []
    steps: dict[str, dict] = {}

    def step(name, action):
        try:
            steps[name] = {"ok": True, **(action() or {})}
        except Exception as exc:
            steps[name] = {"ok": False, "error": repr(exc)[:500]}

    def source_config():
        data = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != SOURCE_FILE_HASHES["config.json"]:
            raise ValueError("source config.json differs from the pinned hash")
        return {"bytes": len(data)}

    def source_shard():
        index = json.loads((StoragePath(BASE_MODEL) / "model.safetensors.index.json").read_bytes())
        shard = sorted(set(index["weight_map"].values()))[0]
        with precision_source_audit.open_audit_stream(str(StoragePath(BASE_MODEL) / shard)) as stream:
            data = stream.read(SMOKE_BYTES)
        if len(data) != SMOKE_BYTES:
            raise ValueError(f"short read from {shard}: {len(data)} bytes")
        return {"shard": shard, "bytes": len(data)}

    def receipt_writer():
        path = f"{smoke}/receipt.json"
        value = {"io_smoke": True}
        written.append(path)
        write_export_json(StoragePath(path), value)
        if json.loads(StoragePath(path).read_bytes()) != value:
            raise ValueError("receipt read back differs")
        return {"path": path}

    def gate_writers():
        paths = [f"{smoke}/precision-gate/ranks/0.json", f"{smoke}/precision-gate/positions.pt"]
        written.extend(paths)
        write_json(paths[0], {"io_smoke": True})
        save_positions({"io-smoke": {"tokens": [1, 2]}}, paths[1])
        if json.loads(read(paths[0])) != {"io_smoke": True} or not read(paths[1]):
            raise ValueError("gate outputs read back differ")
        return {"paths": paths}

    def gate_readers():
        data = read(routing["path"])
        if hashlib.sha256(data).hexdigest() != routing["sha256"]:
            raise ValueError("routing reference differs from the receipt's hash")
        sequences = stored_sequences(f"{INPUTS}/positions.pt")
        load_tokenizer(BASE_MODEL)
        return {"routing_bytes": len(data), "stored_sequences": len(sequences)}

    def cleanup():
        fs, _ = filesystem_for(smoke)
        for path in written:
            _, local = filesystem_for(path)
            if fs.exists(local):
                fs.rm_file(local)
        left = [path for path in written if fs.exists(filesystem_for(path)[1])]
        if left:
            raise ValueError(f"io-smoke objects left behind: {left}")
        return {"deleted": len(written)}

    step("pod_storage", check_pod_storage)
    step("source_config", source_config)
    step("source_shard_first_mb", source_shard)
    step("receipt_writer", receipt_writer)
    step("gate_writers", gate_writers)
    step("gate_readers", gate_readers)
    step("cleanup", cleanup)
    return {"ok": all(result["ok"] for result in steps.values()), "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-root", required=True, help="fresh s3://marin-rno2a/ prefix; must not exist")
    parser.add_argument("--runtime-receipt", required=True, help="local reviewed runtime receipt JSON")
    parser.add_argument("--runtime-receipt-sha256", required=True, help="the reviewed receipt's sha256")
    parser.add_argument(
        "--route-policy",
        choices=ROUTE_POLICIES,
        required=True,
        help="the routes proxy gate: G1 (torch), G1a (vLLM) or report_g1b_l0 (G1b's L0 line)",
    )
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--allow-long-timeout", action="store_true", help=f"permit more than {MAX_TIMEOUT_SECONDS} s")
    parser.add_argument("--expect-commit", help="the reviewed worker pin; required with --run, must equal HEAD")
    parser.add_argument("--priority", choices=sorted(PRIORITY_BANDS), default="interactive")
    parser.add_argument("--capacity-record", help="the launch owner's capacity guard JSON; required with --run")
    parser.add_argument("--launch-receipt", help="local JSON receipt to write; required with --run")
    parser.add_argument("--run", action="store_true", help="submit; without it only print the plan")
    parser.add_argument("--io-smoke", action="store_true", help="run the storage smoke under --run-root; no job")
    parser.add_argument("--emulate-pod-env", action="store_true", help="with --io-smoke: set the pod storage env")
    parser.add_argument("--smoke-receipt", help="local JSON receipt for --io-smoke")
    args = parser.parse_args()
    run_root = args.run_root.rstrip("/")
    if not run_root.startswith("s3://marin-rno2a/"):
        raise SystemExit("run root must be on s3://marin-rno2a/")
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be a positive number of seconds")
    if args.timeout_seconds > MAX_TIMEOUT_SECONDS and not args.allow_long_timeout:
        raise SystemExit(f"timeout above {MAX_TIMEOUT_SECONDS} s needs --allow-long-timeout")
    receipt_bytes = Path(args.runtime_receipt).read_bytes()
    receipt = approved_receipt(receipt_bytes, args.runtime_receipt_sha256)
    # The recorded layout comes from the factory itself, built from the pinned source metadata. The submit host
    # reads through the bucket-routed filesystem: plain StoragePath needs the pod's S3 settings.
    source_fs, source = filesystem_for(BASE_MODEL)
    config_bytes = source_fs.cat_file(prefix_join(source, "config.json"))
    generation_bytes = source_fs.cat_file(prefix_join(source, "generation_config.json"))
    if hashlib.sha256(config_bytes).hexdigest() != SOURCE_FILE_HASHES["config.json"]:
        raise SystemExit("source config.json differs from the pinned hash")
    if hashlib.sha256(generation_bytes).hexdigest() != SOURCE_FILE_HASHES["generation_config.json"]:
        raise SystemExit("source generation_config.json differs from the pinned hash")
    config = base_config(run_root, args.job_name, config_bytes, generation_bytes, receipt)
    routing = receipt["routing_reference"]["path"]
    job_plan = plan(
        args.job_name,
        run_root,
        args.timeout_seconds,
        routing,
        args.priority,
        config.trainer.mesh,
        args.runtime_receipt_sha256,
        receipt["xla_precision"],
        args.route_policy,
    )
    print(json.dumps(job_plan, indent=2))
    if args.io_smoke:
        if args.run or not args.smoke_receipt or not args.expect_commit:
            raise SystemExit("--io-smoke needs --smoke-receipt and --expect-commit, and no --run")
        if job_plan["commit"] != args.expect_commit or not job_plan["clean"]:
            raise SystemExit(f"io-smoke must run on a clean checkout of {args.expect_commit}")
        fs, root = filesystem_for(run_root)
        if fs.exists(root):
            raise SystemExit(f"smoke root already exists: {run_root}")
        if args.emulate_pod_env:
            emulate_pod_env()
        result = io_smoke(run_root, receipt["routing_reference"])
        smoke_receipt = {
            "commit": job_plan["commit"],
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "scratch_root": run_root,
            "endpoint": os.environ.get("AWS_ENDPOINT_URL"),
            "pod_env_emulated": args.emulate_pod_env,
            "not_exercised": "the task pod's node-local endpoint and its secret-file delivery",
            **result,
        }
        Path(args.smoke_receipt).write_text(json.dumps(smoke_receipt, indent=2) + "\n")
        print(json.dumps(smoke_receipt, indent=2))
        raise SystemExit(0 if result["ok"] else 1)
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
    job_id = submit(
        args.job_name,
        run_root,
        args.job_name,
        receipt_bytes,
        args.runtime_receipt_sha256,
        args.route_policy,
        args.timeout_seconds,
        args.priority,
        receipt["xla_precision"],
    )
    launch.update(state="submitted", iris_job_id=job_id)
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps({"job": job_id, "status": "submitted"}))


if __name__ == "__main__":
    main()
