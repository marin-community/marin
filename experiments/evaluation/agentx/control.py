# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit/cancel Iris coordinators and deliver their completed results to ingestion."""

import argparse
import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client import resolve_job_user
from iris.cluster.constraints import Constraint, ConstraintOp, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.evaluation.agentx.artifacts import copy_file, read_record, unpack, write_record
from experiments.evaluation.agentx.coordinator import COORDINATOR_TIMEOUT, recover_results, run_coordinator
from experiments.evaluation.agentx.recipe import load_series


def action_output(name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError("Action output must be one line")
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write(f"{name}={value}\n")


def source_run() -> dict:
    if os.environ["GITHUB_ACTIONS"] != "true":
        raise ValueError("Submission provenance must come from its actual GitHub Actions run")
    return {
        "kind": "github-actions",
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "github_head_sha": os.environ["GITHUB_SHA"],
        "event": os.environ["GITHUB_EVENT_NAME"],
    }


def submission_uri(root: str, run_id: int, attempt: int, profile: str) -> str:
    return prefix_join(root, f"submissions/{run_id}-{attempt}-{profile}.json")


def coordinator_plan(
    source: dict, profile: str, cluster: str, priority: str, root: str, concurrency: int | None
) -> dict:
    """Freeze the Action identity and requested grid; Iris resolves the recipe."""
    series = load_series()
    name = f"agentx-{source['run_id']}-{source['run_attempt']}-{profile}"
    return {
        "name": name,
        "job_id": str(JobName.root(resolve_job_user(None), name)),
        "source": source,
        "profile": profile,
        "cluster": cluster if profile == "h100" else series["profiles"][profile]["cluster"],
        "priority": priority,
        "root": root,
        "retention_uri": prefix_join(root, f"runs/{source['run_id']}/{source['run_attempt']}/{profile}"),
        "concurrencies": [
            value
            for value in series["profiles"][profile]["concurrencies"]
            if concurrency is None or value == concurrency
        ],
    }


def submit_coordinator(client: IrisClient, plan: dict):
    return client.submit(
        entrypoint=Entrypoint.from_callable(run_coordinator, plan),
        name=plan["name"],
        resources=ResourceSpec(cpu=2, memory="32GB", disk="64GB"),
        environment=EnvironmentSpec(sync_packages=["marin-core"], extras=["cpu"]),
        constraints=[
            Constraint.create(key="cluster", op=ConstraintOp.EQ, value=plan["cluster"]),
            preemptible_constraint(False),
        ],
        priority_band={"batch": job_pb2.PRIORITY_BAND_BATCH, "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE}[
            plan["priority"]
        ],
        scheduling_timeout=Duration.from_seconds(load_series()["scheduling_timeout"]),
        timeout=Duration.from_seconds(COORDINATOR_TIMEOUT),
        max_retries_failure=0,
        max_retries_preemption=0,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
        task_image=load_series()["task_image"],
    )


def pending_result(root: str) -> str | None:
    """Select completed output awaiting artifact delivery; never inspect or schedule jobs."""
    filesystem, prefix = filesystem_for(prefix_join(root, "completed"))
    if not filesystem.exists(prefix):
        return None
    for path in sorted(filesystem.ls(prefix, detail=False)):
        if path.endswith(".json"):
            uri = filesystem.unstrip_protocol(path)
            marker = prefix_join(root, f"exported/{Path(path).name}")
            marker_fs, marker_path = filesystem_for(marker)
            if not marker_fs.exists(marker_path):
                return uri
    return None


def materialize(result_uri: str, output: Path) -> dict:
    result = read_record(result_uri)
    output.mkdir(parents=True, exist_ok=False)
    for index, point in enumerate(result["points"]):
        receipt = point["receipt"]
        if receipt["status"] != "valid":
            continue
        archive = output.parent / f"point-{index}.tar.zst"
        copy_file(receipt["artifacts_uri"], str(archive))
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != receipt["artifacts_sha256"]:
            raise ValueError("Retained processed artifact checksum changed")
        unpack(archive, output)
    write_record(str(output / "agentx-source.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    commands = {
        name: subparsers.add_parser(name) for name in ("submit", "observe", "cancel", "recover", "export", "ack")
    }
    for command in commands.values():
        command.add_argument("--root", required=True)
    for name in ("submit", "observe", "cancel", "recover"):
        commands[name].add_argument("--profile", choices=["h100", "gb200", "both"], default="both")
    for name in ("observe", "cancel", "recover"):
        commands[name].add_argument("--run-id", type=int, required=True)
        commands[name].add_argument("--run-attempt", type=int, default=1)
    commands["submit"].add_argument("--cluster", choices=["cw-rno2a", "cw-us-east-02a"], default="cw-rno2a")
    commands["submit"].add_argument("--priority", choices=["batch", "interactive"], default="batch")
    commands["submit"].add_argument("--concurrency", type=int, choices=[8, 32])
    commands["submit"].add_argument("--work", type=Path)
    commands["observe"].add_argument("--observe-seconds", type=int, default=300)
    commands["export"].add_argument("--work", type=Path, required=True)
    commands["export"].add_argument("--result-uri")
    commands["ack"].add_argument("--result-uri", required=True)
    commands["ack"].add_argument("--artifact-url", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.operation == "ack":
        write_record(
            prefix_join(args.root, f"exported/{Path(args.result_uri).name}"), {"artifact_url": args.artifact_url}
        )
        return
    if args.operation == "export":
        uri = args.result_uri or pending_result(args.root)
        if uri is None:
            return
        result = materialize(uri, args.work / "import")
        action_output("result_uri", uri)
        action_output("bundle", str(args.work / "import"))
        action_output("source_run_id", str(result["source"]["run_id"]))
        action_output("source_run_attempt", str(result["source"]["run_attempt"]))
        return
    profiles = ["h100", "gb200"] if args.profile == "both" else [args.profile]
    source = source_run() if args.operation == "submit" else None
    run_id = source["run_id"] if source is not None else args.run_id
    attempt = source["run_attempt"] if source is not None else args.run_attempt
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            for profile in profiles:
                uri = submission_uri(args.root, run_id, attempt, profile)
                if source is not None:
                    try:
                        plan = read_record(uri)
                    except FileNotFoundError:
                        plan = coordinator_plan(
                            source, profile, args.cluster, args.priority, args.root, args.concurrency
                        )
                        write_record(uri, plan)
                    if args.work is not None:
                        write_record(str(args.work / f"{profile}-submission.json"), plan)
                    job = submit_coordinator(client, plan)
                    if str(job.job_id) != plan["job_id"]:
                        raise ValueError("Iris job identity differs from the retained submission")
                    print(
                        f"Submitted {job.job_id}; {job.job_id.dashboard_url(endpoint.url)}; recovery descriptor: {uri}",
                        flush=True,
                    )
                    continue
                plan = read_record(uri)
                job = client.job(JobName.from_string(plan["job_id"]))
                if args.operation == "cancel":
                    job.cancel()
                elif args.operation == "observe":
                    try:
                        job.wait(timeout=args.observe_seconds, stream_logs=True)
                    except TimeoutError:
                        print(
                            f"Observation ended; Iris continues at {job.job_id.dashboard_url(endpoint.url)}. "
                            f"Reattach: uv run iris --cluster=marin job logs {job.job_id} --follow",
                            flush=True,
                        )
                elif args.operation == "recover":
                    with tempfile.TemporaryDirectory(prefix="agentx-recovery-") as directory:
                        work = Path(directory)
                        print(recover_results(client, plan, work))


if __name__ == "__main__":
    main()
