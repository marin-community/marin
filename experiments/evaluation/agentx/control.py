# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded GitHub Actions entry points for submission, collection and cancellation."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.agentx.batch import advance_batch, cancel_batch, create_batch, read_record, write_record
from experiments.evaluation.agentx.collect import collect_terminal, copy_file, unpack


def action_output(name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError("Action output must be one line")
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


def checkpoint_uri(root: str, run_id: int, attempt: int, phase: str = "active") -> str:
    return prefix_join(root, f"states/{phase}/{run_id}-{attempt}.json")


def select_checkpoint(root: str, run_id: int | None, attempt: int) -> tuple[str, dict] | None:
    if run_id is not None:
        for phase in ("active", "complete"):
            uri = checkpoint_uri(root, run_id, attempt, phase)
            try:
                return uri, read_record(uri)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("No saved plan for this GitHub run and attempt")
    filesystem, prefix = filesystem_for(prefix_join(root, "states/active"))
    try:
        paths = filesystem.ls(prefix, detail=False)
    except FileNotFoundError:
        return None
    if len(paths) > 100:
        raise RuntimeError("More than 100 active descriptors; inspect the backlog before submitting more work")
    plans = [
        (filesystem.unstrip_protocol(path), read_record(filesystem.unstrip_protocol(path)))
        for path in paths
        if path.endswith(".json")
    ]
    return min(plans, key=lambda item: item[1]["created_unix"]) if plans else None


def github_attempt(source: dict) -> dict:
    path = f"repos/{source['repository']}/actions/runs/{source['run_id']}/attempts/{source['run_attempt']}"
    return json.loads(subprocess.check_output(["gh", "api", path], text=True, timeout=60))


def materialize(batch: dict, output: Path) -> None:
    """Recreate a complete import bundle even if an earlier collector lost its upload step."""
    output.mkdir(parents=True, exist_ok=True)
    for index, point in enumerate(batch["points"]):
        receipt = point.get("receipt")
        if receipt is None or receipt["status"] != "valid":
            continue
        archive = output.parent / f"point-{index}.tar.zst"
        copy_file(receipt["artifacts_uri"], str(archive))
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != receipt["artifacts_sha256"]:
            raise ValueError("Retained processed artifact checksum changed")
        unpack(archive, output)
    write_record(
        str(output / "agentx-source.json"),
        {
            "source": batch["source"],
            "plan_sha256": batch["plan_sha256"],
            "status": batch["status"],
            "points": batch["points"],
            "collector_run_id": int(os.environ["GITHUB_RUN_ID"]),
            "collector_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        },
    )


def acknowledge(checkpoint: str, artifact_url: str) -> None:
    batch = read_record(checkpoint)
    batch["last_collection"] = {
        "artifact_url": artifact_url,
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
    }
    write_record(checkpoint, batch)
    if batch["status"] == "complete" or (
        batch["status"] == "cancelled"
        and all(point["state"] in {"unsubmitted", "cancelled", "collected", "failed"} for point in batch["points"])
    ):
        destination = checkpoint.replace("/states/active/", "/states/complete/")
        if destination != checkpoint:
            copy_file(checkpoint, destination)
            filesystem, path = filesystem_for(checkpoint)
            filesystem.rm(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["submit", "select", "advance", "cancel", "ack"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--profile", choices=["h100", "gb200", "both"], default="both")
    parser.add_argument("--cluster", choices=["cw-rno2a", "cw-us-east-02a"], default="cw-rno2a")
    parser.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--concurrency", type=int, choices=[8, 32], help="One point from the fixed grid; omit for both")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-attempt", type=int, default=1)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--processor", type=Path)
    parser.add_argument("--artifact-url")
    args = parser.parse_args()
    if args.operation == "select":
        selected = select_checkpoint(args.root, args.run_id, args.run_attempt)
        if selected:
            checkpoint, batch = selected
            action_output("checkpoint", checkpoint)
            action_output("head_sha", batch["source"]["head_sha"])
            action_output("source_run_id", str(batch["source"]["run_id"]))
            action_output("source_run_attempt", str(batch["source"]["run_attempt"]))
        return
    if args.operation == "ack":
        acknowledge(args.checkpoint, args.artifact_url)
        return
    if args.operation == "submit":
        source = source_run()
        checkpoint = checkpoint_uri(args.root, source["run_id"], source["run_attempt"])
        try:
            batch = read_record(checkpoint)
        except FileNotFoundError:
            profiles = ["h100", "gb200"] if args.profile == "both" else [args.profile]
            retention = prefix_join(args.root, f"runs/{source['run_id']}/{source['run_attempt']}")
            batch = create_batch(
                source, profiles, args.cluster, args.priority, "runner", retention, time.time(), args.concurrency
            )
            write_record(checkpoint, batch)
        action_output("checkpoint", checkpoint)
        oldest = select_checkpoint(args.root, None, 1)
        if oldest is not None and oldest[0] != checkpoint:
            print("Saved this batch behind the existing active batch; no second GPU allocation was submitted")
            return
    else:
        checkpoint = args.checkpoint or checkpoint_uri(args.root, args.run_id, args.run_attempt)
        batch = read_record(checkpoint)
        oldest = select_checkpoint(args.root, None, 1)
        if (
            args.operation == "advance"
            and batch["status"] == "active"
            and oldest is not None
            and oldest[0] != checkpoint
        ):
            print("Another active batch precedes this one; no second GPU allocation was submitted")
            return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:

            def collect(status, manifest, retention):
                if args.work is None or args.processor is None:
                    return None
                return collect_terminal(
                    status, manifest, retention, args.work, args.processor, time.time(), client=client
                )

            if args.operation == "cancel" or github_attempt(batch["source"])["conclusion"] == "cancelled":
                cancel_batch(client, batch, checkpoint, collect)
            else:
                head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                if head != batch["source"]["head_sha"]:
                    raise ValueError(
                        "Collector must check out the saved source commit before it can submit another point"
                    )

                advance_batch(client, batch, checkpoint, collect, time.time())
    if args.operation == "submit":
        return
    if args.work is not None:
        materialize(batch, args.work / "import")
        action_output("bundle", str(args.work / "import"))
        action_output("source_run_id", str(batch["source"]["run_id"]))
        action_output("source_run_attempt", str(batch["source"]["run_attempt"]))
    print(
        json.dumps(
            {
                "status": batch["status"],
                "points": [{"job_id": p["job_id"], "state": p["state"]} for p in batch["points"]],
            }
        )
    )
    if any(point["state"] == "failed" for point in batch["points"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
