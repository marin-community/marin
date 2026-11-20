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

from experiments.post_training.task_curation.difficulty.ledger import LEDGER_VERSION, digest
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, action_from_payload
from experiments.post_training.task_curation.difficulty.remediate import repair_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest = StoragePath(args.manifest)
    payload = json.loads(manifest.read_text())
    key = digest(payload)
    if manifest.name != key + ".json":
        raise ValueError("Action manifest changed after planning")
    action = action_from_payload(payload)
    with tempfile.TemporaryDirectory(prefix="difficulty-action-") as directory:
        local = Path(directory)
        if isinstance(action, AttemptBatch):
            queues = {(source.queue_uri, source.queue_sha256) for source in action.sources}
            if len(queues) != 1:
                raise ValueError("A batch must use one frozen panel queue")
            uri, sha = queues.pop()
            output = str(StoragePath(marin_prefix()) / "rl-data" / "atlas" / "difficulty-reconciler" / "runs" / key)
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
                str(local / "runtime"),
                "--workers",
                "22",
                "--attempts-per-task",
                "3",
                "--priority",
                "interactive",
                "--daytona-provisioning",
                "declarative",
                "--ledger",
                args.ledger,
                "--action-manifest",
                args.manifest,
            ]
            for source in action.sources:
                command.extend(["--source-id", source.source_id])
            subprocess.run(command, check=True)
            result = {"status": "completed", "run_root": output}
        else:
            result = repair_action(action, local)
    target = StoragePath(args.ledger) / LEDGER_VERSION / "results" / f"{key}.json"
    target.parent.mkdirs()
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
