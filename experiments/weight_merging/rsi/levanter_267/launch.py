# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded single-slice Levanter Snowball smoke; dry run unless --submit is given."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.actor.resolver import ResolveResult
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.cluster.worker.env_probe import _probe_tpu_metadata
from rigging.timing import Duration

from experiments.weight_merging.rsi.levanter_267.runtime import run


def _resolve(resolver, name):
    try:
        return resolver.resolve(name)
    except ConnectError as exc:
        if exc.code not in (Code.NOT_FOUND, Code.UNAVAILABLE):
            raise
        return ResolveResult(name=name)


def run_host(mode, model, output, item, max_seq_len, source_commit, max_seqs):
    info = get_job_info()
    ctx = iris_ctx()
    slice_id, variant, _, worker_id, _ = _probe_tpu_metadata()
    if not slice_id or not worker_id or variant != "v4-32":
        raise RuntimeError(f"Wrong TPU identity: {slice_id}, {variant}, {worker_id}")
    registration = ctx.registry.register(
        f"levanter267-identity-{info.task_index}",
        info.advertise_host,
        metadata={"slice": slice_id, "worker": worker_id},
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            identities = [_resolve(ctx.resolver, f"levanter267-identity-{i}") for i in range(4)]
            if all(not value.is_empty for value in identities):
                break
            time.sleep(2)
        else:
            raise TimeoutError("Four-host slice-identity preflight timed out")
        metadata = [entry.first().metadata for entry in identities]
        if {m["slice"] for m in metadata} != {slice_id} or len({m["worker"] for m in metadata}) != 4:
            raise RuntimeError(f"Gang is not one slice with four distinct hosts: {metadata}")
        print("SINGLE_SLICE_VERIFIED", json.dumps(metadata), flush=True)
        run(mode, model, output, item, max_seq_len, source_commit, max_seqs)
    finally:
        ctx.registry.unregister(registration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--mode", choices=("tiny", "numerics", "serve"), default="tiny")
    parser.add_argument(
        "--model",
        default="gs://marin-us-central2/scratch/karan/rsi-merge/parents/S--open-athena_Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02@50fe2f488aee/",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--item", type=Path)
    parser.add_argument("--oracle", type=Path, help="Pinned item303 oracle artifact for hash/token validation")
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--max-seqs", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--runtime-minutes", type=int, default=90)
    parser.add_argument("--name", default="levanter267-v4-32-tiny")
    args = parser.parse_args()
    if not 1 <= args.runtime_minutes <= 1200:
        parser.error("Runtime must be bounded to at most twenty hours")
    if args.mode != "tiny" and (not args.item or not args.oracle):
        parser.error("The numerics item and pinned oracle are required for full-model runs")
    # Reads and outputs stay in the reserved slice's region. No implicit cross-region mirror.
    if not args.model.startswith("gs://marin-us-central2/") or not args.output.startswith("gs://marin-us-central2/"):
        parser.error("This launcher only permits us-central2 model and output paths")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.submit and subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode:
        parser.error("Commit tracked source changes before submitting a reproducible probe")
    item = json.loads(args.item.read_text()) if args.item else None
    if item is not None and args.oracle is not None:
        oracle = json.loads(args.oracle.read_text())
        if str(oracle["id"]) != str(item["id"]):
            parser.error("Oracle and numerical item IDs differ")
        item = dict(
            item,
            expected_input_sha256=oracle["input_sha256"],
            oracle_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
            expected_token_ids=[row["token_id"] for row in oracle["tokens"]],
        )
    resource = ResourceConfig.with_tpu(
        "v4-32",
        cpu=32,
        ram="400g",
        disk="80g",
        regions=["us-central2"],
        zone="us-central2-b",
        preemptible=False,
    )
    print(
        json.dumps(
            dict(
                source_commit=source_commit,
                name=args.name,
                mode=args.mode,
                hosts=4,
                expert=16,
                model_parallel=1,
                max_seq_len=args.max_seq_len,
                max_seqs=args.max_seqs,
                max_pages=((args.max_seq_len + 127) // 128 + 2) * args.max_seqs,
                runtime_minutes=args.runtime_minutes,
                model=args.model,
                output=args.output,
            ),
            indent=2,
        )
    )
    print("COSCHEDULING", resolve_coscheduling(resource, resource.replicas))
    print("PORTS", ["jax", "http"])
    if not args.submit:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=args.name,
                entrypoint=Entrypoint.from_callable(
                    run_host, args.mode, args.model, args.output, item, args.max_seq_len, source_commit, args.max_seqs
                ),
                resources=convert_resources(resource),
                constraints=convert_constraints(resource),
                replicas=resource.replicas,
                coscheduling=resolve_coscheduling(resource, resource.replicas),
                ports=["jax", "http"],
                environment=EnvironmentSpec(extras=("tpu",)),
                timeout=Duration.from_seconds(args.runtime_minutes * 60),
                scheduling_timeout=Duration.from_seconds(14400),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            print(job)


if __name__ == "__main__":
    main()
