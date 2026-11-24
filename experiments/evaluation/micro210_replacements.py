# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic same-family replacements for gate-broken Micro210 tasks.

Three rows fail their oracle/NoP build gate in the verified in-region environment and
their task environments are the defect (#12159-style: the task's own oracle or its
image-baked dependencies, not the grading code):

- ``tb2-recovery/terminal-bench/mcmc-sampling-stan`` (R package source builds fail),
- ``ot-tblite-recovery/pdf-table-parsing`` (its solve installs ``camelot`` but the
  verifier imports it before the install lands),
- ``humanevalplus/HumanEval/130`` (oracle and NoP both zero in-region; passes elsewhere).

Each gets one same-family replacement chosen deterministically: seed 20261124, shuffled
family pool, first candidate not already in the policy whose gate grades oracle 1.0 and
NoP 0.0 on every repetition. The swap and each removed task's reason are recorded in the
policy provenance.
"""

import base64
import json
import random
import re
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

REPLACEMENT_SEED = 20261124

TB2_REGISTRY = ("harbor-framework/terminal-bench-1", "original-tasks")
TB2_IMAGE_TAG = "20251031"
OT_TBLITE_DATASET = ("DCAgent/dev_set_v2", "377118ff3031c934")
HUMANEVALPLUS_DATASET = ("evalplus/humanevalplus", "v0.1.10")

BROKEN_TASKS = {
    "tb2-recovery/terminal-bench/mcmc-sampling-stan": "the task's oracle solve fails building R packages from source in the verified environment",
    "ot-tblite-recovery/pdf-table-parsing": "the task's verifier imports camelot before the solve's install lands",
    "humanevalplus/HumanEval/130": "oracle and NoP both zero in the verified in-region environment",
}


def _row_spec(row: dict) -> dict:
    return json.loads(row["task_spec"])


def _verifier_files(spec: dict) -> dict[str, bytes]:
    return {
        resource["path"]: base64.b64decode(resource["source"]["content_base64"])
        for resource in spec["resources"]["verifier"]
    }


def policy_task_ids(rows: list[dict]) -> set[str]:
    return {row["source_task_id"] for row in rows}


@dataclass(frozen=True)
class CandidateRow:
    """One same-family replacement row built from a pinned upstream pool."""

    benchmark: str
    source_task_id: str
    task_id: str
    task_spec: str
    task_rollout_spec: str


def _shuffled(names: list[str], skip: set[str]) -> list[str]:
    """Deterministic candidate order: fixed seed, sorted names, policy members dropped."""
    ordered = sorted(name for name in names if name not in skip)
    random.Random(REPLACEMENT_SEED).shuffle(ordered)
    return ordered


def humanevalplus_candidate(source_task_id: str, template_row: dict) -> CandidateRow:
    """Build one humanevalplus row from the evalplus pool in the frozen row's shape."""
    from huggingface_hub import hf_hub_download

    dataset, revision = HUMANEVALPLUS_DATASET
    table = pq.read_table(
        hf_hub_download(dataset, repo_type="dataset", revision=revision, filename="data/test-00000-of-00001-5973903632b82d40.parquet")
    )
    problems = {row["task_id"]: row for row in table.to_pylist()}
    record = problems[source_task_id]
    spec = _row_spec(template_row)
    files = _verifier_files(spec)
    config = json.loads(files["config.json"])
    doc = config["spec"]["doc"]
    doc.update(
        prompt=record["prompt"],
        canonical_solution=record["canonical_solution"],
        entry_point=record["entry_point"],
        test=record["test"],
        task_id=record["task_id"],
    )
    reference = config["spec"]["grader"]["reference"]
    reference["problem"] = {
        "prompt": record["prompt"],
        "canonical_solution": record["canonical_solution"],
        "entry_point": record["entry_point"],
        "test": record["test"],
        "task_id": record["task_id"],
    }
    files["config.json"] = json.dumps(config, allow_nan=False, sort_keys=True, indent=1).encode()
    new_spec = dict(spec)
    task_id = f"humanevalplus/{record['task_id']}"
    new_spec["id"] = task_id
    new_spec["labels"] = {
        **spec["labels"],
        "micro210_replacement_source": json.dumps({"dataset": dataset, "revision": revision, "task": record["task_id"]}),
    }
    new_spec["resources"] = {
        **spec["resources"],
        "verifier": [
            {
                "path": resource["path"],
                "source": {
                    "kind": "inline_file",
                    "content_base64": base64.b64encode(files[resource["path"]]).decode(),
                },
            }
            for resource in spec["resources"]["verifier"]
        ],
    }
    return CandidateRow(
        benchmark="humanevalplus",
        source_task_id=record["task_id"],
        task_id=task_id,
        task_spec=json.dumps(new_spec, sort_keys=True),
        task_rollout_spec=template_row["task_rollout_spec"],
    )


def ot_tblite_candidate(source_task_id: str, template_row: dict, package_files: dict[str, bytes]) -> CandidateRow:
    """Build one ot-tblite row from the DCAgent pool with the Pi runtime injected."""
    dataset, revision = OT_TBLITE_DATASET
    spec = _row_spec(template_row)
    frozen = _verifier_files(spec)
    pi_manifests = {
        path: payload for path, payload in frozen.items() if path.startswith("environment/long-loop-pi/")
    }
    verifier = [
        {"path": "instruction.md", "content": package_files["instruction.md"]},
        {"path": "solution/solve.sh", "content": package_files["solution/solve.sh"]},
        {"path": "tests/test.sh", "content": package_files["tests/test.sh"]},
    ]
    verifier += [
        {"path": f"tests/{path}", "content": payload}
        for path, payload in package_files.items()
        if path.startswith("tests/") and path != "tests/test.sh"
    ]
    verifier += [
        {"path": f"environment/{path}", "content": payload}
        for path, payload in package_files.items()
        if path.startswith("environment/") and not path.startswith("environment/long-loop-pi")
    ]
    verifier.append({"path": "environment/Dockerfile", "content": _pi_dockerfile(package_files["environment/Dockerfile"].decode())})
    verifier += [{"path": path, "content": payload} for path, payload in pi_manifests.items()]
    toml = package_files["task.toml"].decode()
    if "[verifier]" not in toml:
        toml = toml.replace("[environment]", '[verifier]\ntimeout_sec = 1800.0\n\n[environment]')
    verifier.append({"path": "task.toml", "content": toml.encode()})
    resources = [
        {"path": entry["path"], "source": {"kind": "inline_file", "content_base64": base64.b64encode(entry["content"]).decode()}}
        for entry in verifier
    ]
    task_id = f"ot-tblite-recovery/{source_task_id}"
    new_spec = dict(spec)
    new_spec["id"] = task_id
    new_spec["labels"] = {**spec["labels"], "micro210_replacement_source": json.dumps({"dataset": dataset, "revision": revision})}
    new_spec["resources"] = {**spec["resources"], "verifier": resources}
    return CandidateRow(
        benchmark="ot-tblite-recovery",
        source_task_id=source_task_id,
        task_id=task_id,
        task_spec=json.dumps(new_spec, sort_keys=True),
        task_rollout_spec=template_row["task_rollout_spec"],
    )


def tb2_candidate(source_task_id: str, template_row: dict, package_files: dict[str, bytes]) -> CandidateRow:
    """Build one tb2-recovery row from the terminal-bench registry at the pinned tag."""
    spec = _row_spec(template_row)
    frozen = _verifier_files(spec)
    pi_manifests = {path: payload for path, payload in frozen.items() if path.startswith("environment/long-loop-pi/")}
    verifier = [{"path": "instruction.md", "content": package_files["instruction.md"]}]
    verifier += [
        {"path": f"environment/{path}", "content": payload}
        for path, payload in package_files.items()
        if path.startswith(("environment/task-deps/", "environment/data/", "environment/logs/", "environment/scaffold/"))
    ]
    verifier += [{"path": "tests/test.sh", "content": package_files["tests/test.sh"]}]
    verifier += [
        {"path": f"tests/{path}", "content": payload}
        for path, payload in package_files.items()
        if path.startswith("tests/") and path != "tests/test.sh"
    ]
    verifier += [{"path": "solution/solve.sh", "content": package_files["solution/solve.sh"]}]
    verifier.append({"path": "environment/Dockerfile", "content": _pi_dockerfile_tb2(source_task_id)})
    verifier += [{"path": path, "content": payload} for path, payload in pi_manifests.items()]
    toml = package_files["task.toml"].decode()
    if "[verifier]" not in toml:
        toml = toml.replace("[environment]", '[verifier]\ntimeout_sec = 1800.0\n\n[environment]')
    verifier.append({"path": "task.toml", "content": toml.encode()})
    resources = [
        {"path": entry["path"], "source": {"kind": "inline_file", "content_base64": base64.b64encode(entry["content"]).decode()}}
        for entry in verifier
    ]
    task_id = f"tb2-recovery/terminal-bench/{source_task_id}"
    new_spec = dict(spec)
    new_spec["id"] = task_id
    new_spec["labels"] = {
        **spec["labels"],
        "micro210_replacement_source": json.dumps(
            {"registry": TB2_REGISTRY[0], "task": source_task_id, "image": f"alexgshaw/{source_task_id}:{TB2_IMAGE_TAG}"}
        ),
    }
    new_spec["resources"] = {**spec["resources"], "verifier": resources}
    return CandidateRow(
        benchmark="tb2-recovery",
        source_task_id=source_task_id,
        task_id=task_id,
        task_spec=json.dumps(new_spec, sort_keys=True),
        task_rollout_spec=template_row["task_rollout_spec"],
    )


def _pi_dockerfile(upstream_dockerfile: str) -> str:
    """Inject the Pi runtime stage into an upstream tblite Dockerfile."""
    stage = (
        "FROM node@sha256:5ec23b7c622b00547753bc6ffd78085dd868a68aeb41c8e0a1418f033e13d96d AS long_loop_pi_runtime\n"
        "WORKDIR /opt/long-loop-pi\n"
        "COPY long-loop-pi/package.json long-loop-pi/package-lock.json ./\n"
        "RUN npm ci --omit=dev --no-audit --no-fund\n"
    )
    tail = (
        "\nCOPY --from=long_loop_pi_runtime /usr/local/bin/node /opt/long-loop-node/node\n"
        "COPY --from=long_loop_pi_runtime /opt/long-loop-pi /opt/long-loop-pi\n"
        "ENV PATH=/opt/long-loop-node:/opt/long-loop-pi/node_modules/.bin:${PATH}\n"
        "RUN PATH=/opt/long-loop-node:/opt/long-loop-pi/node_modules/.bin:${PATH} bash -c 'test \"$(pi --version)\" = \"0.87.0\"'\n"
    )
    return stage + upstream_dockerfile.rstrip("\n") + "\n" + tail


def _pi_dockerfile_tb2(source_task_id: str) -> str:
    """The tb2 Dockerfile: Pi stage on the task's prebuilt registry image."""
    stage = (
        "FROM node@sha256:5ec23b7c622b00547753bc6ffd78085dd868a68aeb41c8e0a1418f033e13d96d AS long_loop_pi_runtime\n"
        "WORKDIR /opt/long-loop-pi\n"
        "COPY long-loop-pi/package.json long-loop-pi/package-lock.json ./\n"
        "RUN npm ci --omit=dev --no-audit --no-fund\n"
    )
    tail = (
        "\nCOPY --from=long_loop_pi_runtime /usr/local/bin/node /opt/long-loop-node/node\n"
        "COPY --from=long_loop_pi_runtime /opt/long-loop-pi /opt/long-loop-pi\n"
        "ENV PATH=/opt/long-loop-node:/opt/long-loop-pi/node_modules/.bin:${PATH}\n"
        "RUN PATH=/opt/long-loop-node:/opt/long-loop-pi/node_modules/.bin:${PATH} bash -c 'test \"$(pi --version)\" = \"0.87.0\"'\n"
    )
    return stage + f"FROM alexgshaw/{source_task_id}:{TB2_IMAGE_TAG}\n" + tail
