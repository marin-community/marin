# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare SWE Harbor packages for the pinned Pi agent and non-root workspace."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.long_loop.eval_runtime.pi_image import PiPathScope, with_pi_runtime
from experiments.post_training.long_loop.rl_dataset_admission import file_digest

AGENT_USER = "longloop-agent"
AGENT_HOME = f"/home/{AGENT_USER}"
SOURCE = "long-loop-smartie-swesmith"
NONROOT_LAYER = f"""
# Agent commands use Harbor's agent.user; provider administration remains root.
RUN useradd --create-home --user-group --uid 10001 --shell /bin/bash {AGENT_USER} \\
    && mkdir -p /testbed /logs/agent /output \\
    && chown -R {AGENT_USER}:{AGENT_USER} /testbed /logs/agent /output {AGENT_HOME} \\
    && chmod 0755 /testbed \\
    && chown root:root / && chmod 0755 / \\
    && TASK_PURELIB="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \\
    && TASK_PLATLIB="$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')" \\
    && TASK_SCRIPTS="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))')" \\
    && chgrp -R {AGENT_USER} "$TASK_PURELIB" "$TASK_PLATLIB" \\
    && chmod -R g+rwX "$TASK_PURELIB" "$TASK_PLATLIB" \\
    && chgrp {AGENT_USER} "$TASK_SCRIPTS" && chmod g+w,+t "$TASK_SCRIPTS" \\
    && ln -s /opt/long-loop-node/node "$TASK_SCRIPTS/node" \\
    && ln -s /opt/long-loop-pi/node_modules/.bin/pi "$TASK_SCRIPTS/pi"
"""


def pi_nonroot_task(task: TaskSpec) -> TaskSpec:
    """Keep task identity and grading, adding Pi and workspace ownership to SWE images."""
    if task.source.dataset != SOURCE:
        return task
    prepared = with_pi_runtime(task, path_scope=PiPathScope.AGENT)
    files = {resource.path: resource for resource in prepared.resources.verifier}
    config = tomllib.loads(resource_bytes(files["task.toml"]).decode())
    agent = config.setdefault("agent", {})
    agent["user"] = AGENT_USER
    agent.setdefault("env", {})["HOME"] = AGENT_HOME
    dockerfile = resource_bytes(files["environment/Dockerfile"]).decode() + NONROOT_LAYER
    replacements = {"task.toml": tomli_w.dumps(config).encode(), "environment/Dockerfile": dockerfile.encode()}
    resources = tuple(r for r in prepared.resources.verifier if r.path not in replacements) + tuple(
        inline_resource(path, content) for path, content in replacements.items()
    )
    return prepared.model_copy(update={"resources": prepared.resources.model_copy(update={"verifier": resources})})


def image_context_digest(task: TaskSpec) -> str:
    digest = hashlib.sha256()
    for resource in sorted(task.resources.verifier, key=lambda r: r.path):
        if resource.path.startswith("environment/"):
            digest.update(resource.path.encode() + b"\0" + resource_bytes(resource) + b"\0")
    return digest.hexdigest()


def prepare_file(source: Path, destination: Path, identities) -> dict:
    count = 0
    changed = 0
    images = Counter()
    with pq.ParquetWriter(destination, pq.read_schema(source), compression="zstd") as writer:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
            rows = batch.to_pylist()
            for row in rows:
                original = row["task_spec"]
                task = TaskSpec.model_validate_json(original)
                prepared = pi_nonroot_task(task)
                if prepared is not task:
                    assert (prepared.id, prepared.context, prepared.grader, prepared.source) == (
                        task.id,
                        task.context,
                        task.grader,
                        task.source,
                    )
                    row["task_spec"] = prepared.model_dump_json()
                    changed += 1
                    images[image_context_digest(prepared)] += 1
                identities.write(
                    json.dumps(
                        {
                            "row": count,
                            "task_id": task.id,
                            "input_sha256": hashlib.sha256(original.encode()).hexdigest(),
                            "output_sha256": hashlib.sha256(row["task_spec"].encode()).hexdigest(),
                            "changed": prepared is not task,
                        }
                    )
                    + "\n"
                )
                count += 1
            writer.write_table(pa.Table.from_pylist(rows, schema=batch.schema))
    return {"rows": count, "changed": changed, "image_contexts": dict(images), "sha256": file_digest(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    args = parser.parse_args()
    if file_digest(args.input / "manifest.json") != args.input_manifest_sha256:
        raise ValueError("The pinned input manifest changed")
    manifest = json.loads((args.input / "manifest.json").read_text())
    for name, pin in manifest["files"].items():
        if file_digest(args.input / name) != pin["sha256"]:
            raise ValueError(f"Input data changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ("agentic-train.parquet", "validation.parquet"):
        with (args.output / f"{name}.identities.jsonl").open("w") as identities:
            files[name] = prepare_file(args.input / name, args.output / name, identities)
    for name, info in files.items():
        info["identities_sha256"] = file_digest(args.output / f"{name}.identities.jsonl")
    if files["agentic-train.parquet"]["changed"] != manifest["agentic_train"]:
        raise ValueError("Every agentic training row must receive the execution overlay")
    if files["validation.parquet"]["changed"] != manifest["validation_composition"]["swesmith"]:
        raise ValueError("Only SWE validation rows may change")
    receipt = {
        **manifest,
        "execution_overlay_version": "pi-task-python-shared-v3",
        "upstream_manifest_sha256": file_digest(args.input / "manifest.json"),
        "marin_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy": (
            "Pi 0.87.0, UID10001; original task Python shared for installations; original verifier environment/user"
        ),
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
