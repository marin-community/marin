# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One regional child job: run a bounded attempt batch or a concrete repair."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret

from experiments.post_training.task_curation.difficulty.campaign import campaign_record
from experiments.post_training.task_curation.difficulty.ledger import (
    LEDGER_VERSION,
    RESULTS_DIRECTORY,
    record_action_failure,
)
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, action_id, load_action
from experiments.post_training.task_curation.difficulty.remediate import repair_action

REGISTRY_AUTH_SECRET = "GHCR_REGISTRY_AUTHFILE"
"""Skopeo authfile JSON for private ghcr images; the controller bundle carries it."""
TPU_ATTEMPT_WORKERS = 4
"""Concurrent tasks per regional batch, validated on a frozen 20-task panel."""


def attempt_command(
    action: AttemptBatch,
    manifest: str,
    ledger: str,
    output: str,
    local_root: Path,
    registry_auth: bool,
    serving_pool: str | None = None,
) -> list[str]:
    """The runtime command for one attempt batch, with the batch's registry credential."""
    queues = {(source.queue_uri, source.queue_sha256) for source in action.sources}
    if len(queues) != 1:
        raise ValueError("A batch must use one frozen panel queue")
    uri, sha = queues.pop()
    command = [
        "python",
        "-m",
        "experiments.post_training.task_curation.difficulty.runtime",
        "--solver",
        action.solver.value,
        "--sources",
        uri,
        "--sources-sha256",
        sha,
        "--output-root",
        output,
        "--local-root",
        str(local_root),
        "--workers",
        "16",
        "--attempts-per-task",
        str(max(n for source in action.sources for _, _, n in source.tasks)),
        "--priority",
        "interactive",
        "--daytona-provisioning",
        "declarative",
        "--ledger",
        ledger,
        "--action-manifest",
        manifest,
    ]
    for source in action.sources:
        command.extend(["--source-id", source.source_id])
    if serving_pool is not None:
        command.extend(["--solver-pool", serving_pool])
    if registry_auth:
        # Private actor pins (ghcr.io/marin-community) need the skopeo authfile;
        # public pulls ignore it, so a bundle without the credential stays anonymous.
        command.extend(["--registry-auth-secret", REGISTRY_AUTH_SECRET])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--serving-pool")
    args = parser.parse_args()
    configure_coreweave_s3()
    action = load_action(args.manifest)
    key = action_id(action)
    try:
        if isinstance(action, AttemptBatch):
            campaign = campaign_record(args.ledger)
            output = (
                args.ledger.removesuffix("/ledger") + "/runs/" + key
                if campaign
                else str(StoragePath(marin_prefix()) / "rl-data" / "atlas" / "difficulty-reconciler" / "runs" / key)
            )
            # The batch's local root must live on the node's NVMe /cache mount: the
            # container root and /tmp are memory-backed on CoreWeave bare metal, so
            # pulled images and sandbox disks there are unreclaimable cgroup memory.
            local_root = (
                Path("/var/lib/iris-cache/atlas-difficulty/actions" if campaign else "/cache/atlas-difficulty/actions")
                / key
                / "runtime"
            )
            command = attempt_command(
                action,
                args.manifest,
                args.ledger,
                output,
                local_root,
                registry_auth=get_secret(REGISTRY_AUTH_SECRET) is not None,
                serving_pool=args.serving_pool,
            )
            if campaign:
                command[command.index("--workers") + 1] = str(TPU_ATTEMPT_WORKERS)
            subprocess.run(command, check=True)
            result = {"status": "completed", "run_root": output, "local_root": str(local_root)}
        else:
            with tempfile.TemporaryDirectory(prefix="difficulty-action-") as directory:
                result = repair_action(action, Path(directory), args.ledger)
    except Exception as error:
        record_action_failure(args.ledger, args.manifest, error)
        raise
    target = StoragePath(args.ledger) / LEDGER_VERSION / RESULTS_DIRECTORY / f"{key}.json"
    target.parent.mkdirs()
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
