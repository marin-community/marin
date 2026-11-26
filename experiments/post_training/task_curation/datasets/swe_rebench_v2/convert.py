# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench V2.1 rows as native Harbor packages behind the shared-source grader.

The repair (rulings recorded in /Users/benfeuer/Documents/experiments/active/swe-repair/STATUS.md):
build from the original author's re-verified release ``ibragim-bad/SWE-rebench-V2.1`` instead of the
DCAgent ``nebius/SWE-rebench-V2`` repackaging, condense the per-task upstream images into ten shared
base images, and grade with the source's own Harbor verifier behind the SWE_REBENCH shared-source
contract.

Per task the converter emits:

- the vendored upstream renderer's files (instruction.md, tests/, solution/) byte-identical to the
  tasks the upstream publishes on the Harbor Hub;
- a rewritten ``task.toml``: the shared base image replaces the per-task digest-pinned image, and the
  upstream image-build step (``combine.Dockerfile.j2``: clone, check out ``base_commit``, drop the
  remote, run the install commands with ``|| true``) becomes the declared task-start setup;
- ``tests/config.json`` extended with the family keys the shared-source contract reads, keeping
  ``FAIL_TO_PASS``/``PASS_TO_PASS`` verbatim so the upstream ``grade.py`` runs unchanged.

``harden_shared_source`` then rewrites the package into the protected SHARED contract; the vendored
upstream ``test.sh``/``grade.py`` bytes remain the sole selection authority (reward channel unchanged).
"""

import base64
import hashlib
import json
import shlex
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import tomlkit
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect
from taskcompendium.harbor import shared_source
from taskcompendium.integrity import SharedDependencyState
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow

from experiments.post_training.task_curation.datasets.native_harbor import native_task
from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream import task_files
from experiments.post_training.task_curation.datasets.swe_rebench_v2.workspace import (
    WORKSPACE,
    to_workspace,
)

HERE = Path(__file__).parent
VERSION = "2026.11.20.2"

IMAGE_BINDINGS_INPUT = "base_images"
IMAGE_BINDINGS_FILENAME = "base-image-bindings.json"

# Optional staged input: the fidelity gate's typed admission receipts (instance_id →
# {"receipt": ..., "evidence": {...}}). A receipt documents a proven, coordinator-ruled case
# where the converted environment's outcomes differ from the upstream digest image's in a
# direction that keeps the row valid RL data — the row is admitted WITH that receipt, never
# silently.
ADMISSION_RECEIPTS_INPUT = "admission_receipts"


class AdmissionReceipt(StrEnum):
    """Typed admission receipts carried on admitted rows whose fidelity comparison diverged.

    ``UPSTREAM_IMAGE_DEFECT``: the row's NoP scores 0 on both variants and its oracle passes on
    the converted shared-base environment while the upstream digest image's oracle FAILS — the
    task spec environment is the RL data, so a working oracle + failing NoP is a valid task
    (coordinator ruling 2026-11-18). The receipt records the upstream oracle failure evidence.
    """

    UPSTREAM_IMAGE_DEFECT = "upstream_image_defect"


def admission_receipt_labels(instance_id: str | None, receipts) -> dict[str, str]:
    """Labels for a row carrying a typed admission receipt; empty when none applies.

    Unknown receipt values fail fast (a malformed receipts input must not silently admit).
    """
    if receipts is None or not instance_id:
        return {}
    staged = json.loads((StoragePath(str(receipts)) / "admission-receipts.json").read_text())
    entry = staged.get(instance_id)
    if not entry:
        return {}
    receipt = AdmissionReceipt(entry["receipt"])
    evidence = entry.get("evidence") or {}
    return {
        "swe_rebench_admission_receipt": receipt.value,
        "swe_rebench_admission_receipt_evidence": json.dumps(evidence, sort_keys=True),
    }


# The upstream per-task image build (combine.Dockerfile.j2) ran exactly this block as a Docker
# RUN step; the repair moves it verbatim into the task-start setup.
CLONE_BLOCK = (
    "mkdir -p {workdir}\n"
    "git clone -o origin https://github.com/{repo} {workdir}\n"
    "chmod -R 777 {workdir}\n"
    "cd {workdir}\n"
    "git reset --hard {base_commit}\n"
    "git remote remove origin\n"
)

# The upstream task.toml declarations (6 CPUs / 24,576 MB / 32,768 MB storage) are recorded
# verbatim below as the SOURCE declaration. The USER DECISION (2026-11-25 ~5 AM EST) approves
# the declared-limits reduction: the repaired package EMITS Daytona's per-sandbox cap
# (4 CPUs / 8,192 MB / 10,240 MB) with the typed deviation record DECLARED_LIMITS_DEVIATION
# (reason: the upstream spec exceeds the Daytona 4/8/10 per-sandbox cap; evidence: the
# fidelity gate ran the full population at 4 CPUs, where every admitted row passed its
# NoP 0.0 + oracle 1.0 trials). The upstream values are never emitted as execution limits.
UPSTREAM_AGENT_TIMEOUT_SEC = 3000.0
UPSTREAM_VERIFIER_TIMEOUT_SEC = 2400.0
UPSTREAM_SETUP_TIMEOUT_SEC = 1800.0
UPSTREAM_BUILD_TIMEOUT_SEC = 1800.0
UPSTREAM_CPUS = 6
UPSTREAM_MEMORY_MB = 24576
UPSTREAM_STORAGE_MB = 32768

AGENT_TIMEOUT_SEC = UPSTREAM_AGENT_TIMEOUT_SEC
VERIFIER_TIMEOUT_SEC = UPSTREAM_VERIFIER_TIMEOUT_SEC
SETUP_TIMEOUT_SEC = 3600.0
"""The setup deadline doubles the upstream declaration (ruling 2026-11-21: the heavy-install
tail exceeds the declared 1800 s at the 4 CPU profile; the fidelity gate proved every row at
3600 s)."""
BUILD_TIMEOUT_SEC = UPSTREAM_BUILD_TIMEOUT_SEC
CPUS = 4
MEMORY_MB = 8192
STORAGE_MB = 10240

DECLARED_LIMITS_DEVIATION = {
    "type": "declared_limits_reduction",
    "reason": "upstream spec exceeds the Daytona 4/8/10 per-sandbox cap",
    "evidence": (
        "the fidelity gate ran the full population at 4 CPUs / 8 GiB / 10 GiB; every admitted "
        "row passed its NoP 0.0 + oracle 1.0 trials at exactly this profile "
        "(s3://marin-us-east-02a/marin/rl-data/atlas/swe-repair/2026.11.20.1/wave-v11/seal.json)"
    ),
    "upstream_declaration": {
        "cpus": UPSTREAM_CPUS,
        "memory_mb": UPSTREAM_MEMORY_MB,
        "storage_mb": UPSTREAM_STORAGE_MB,
        "setup_timeout_sec": UPSTREAM_SETUP_TIMEOUT_SEC,
    },
    "declared_limits": {
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "storage_mb": STORAGE_MB,
        "setup_timeout_sec": SETUP_TIMEOUT_SEC,
    },
    "decision": "user decision 2026-11-25 ~5 AM EST (coordinator relay); release v2 re-declaration ratified",
}


def swe_rebench_profile():
    """The SWE_REBENCH family profile landed by the shared-grader-adapters lane."""
    profile = getattr(shared_source, "SWE_REBENCH_PROFILE", None)
    if profile is None:
        raise ValueError(
            "taskcompendium.harbor.shared_source does not declare SWE_REBENCH_PROFILE yet; "
            "the shared-grader-adapters lane lands it with the family extension"
        )
    return profile


def row_base(data: dict) -> str:
    """The row's declared base: its own base_image_name, or the V1-python group default."""
    declared = (data.get("install_config") or {}).get("base_image_name")
    if declared:
        return declared
    if data.get("language") == "python":
        return groups.V1_PYTHON_BASE
    raise ValueError(f"Row {data.get('instance_id')!r} declares no base and is not a V1-python row")


def _flatten(data: dict) -> dict:
    """Expose install_config entries at the top level the vendored renderer reads."""
    flat = dict(data)
    install_config = data.get("install_config") or {}
    flat["test_cmd"] = install_config.get("test_cmd")
    flat["log_parser"] = install_config.get("log_parser")
    flat["install"] = install_config.get("install") or []
    return flat


def _rust_toolchain(base: str) -> str:
    return base.removeprefix("rust_")


def _jdk_version(base: str) -> str:
    if base.startswith("kotlin-jdk-"):
        return base.removeprefix("kotlin-jdk-")
    if base.startswith("java_"):
        return base.removeprefix("java_")
    if base in ("scala_base", "clojure_base"):
        return "21"
    raise ValueError(f"Unknown jvm base: {base}")


def setup_command(flat: dict, base: str, group: groups.Group) -> str:
    """The declared task-start setup: the upstream clone block, toolchain select, installs.

    Order mirrors the upstream image build: toolchain selection and environment preparation
    (no checkout dependency), then the clone block (which ends inside the workspace), then the
    row's install commands verbatim with their upstream `( ... ) || true` wrappers. The row's
    per-repo workspace path is rewritten to the uniform /testbed workspace everywhere it
    appears (the clone destination and any absolute path inside an install command).

    The setup is IDEMPOTENT: a Daytona sandbox whose name is deterministic can be REUSED with
    a previous trial's state (a crashed trial skips its own cleanup), so the environment
    preparation removes the previous state before recreating it — the declared contract
    (workspace at base_commit with the toolchain and deps) is achieved identically either way.
    """
    prepare = []
    if group.toolchain == "conda":
        # The base pre-creates the declared python envs; clone the row's declared env into
        # `testbed` so the source's `conda activate testbed || conda activate base` flow finds
        # the right python, AND so the (unactivated) grading test_cmd's bare `pytest`/`pip`
        # resolve into the row's env: `testbed/bin` is first on PATH in the base image, and
        # the upstream images build unlabeled (v1-python) rows on python 3.9 (spectree
        # 3.9.21, clique 3.9.20) while python_base_310/37 rows pin 3.10/3.7 respectively.
        if base == "python_base_37":
            env_name = "py37"
        elif base == "python_base_310":
            env_name = "py310"
        else:
            env_name = "py39"
        prepare.append("conda env remove -n testbed -y 2>/dev/null || true")
        prepare.append(f"conda create -n testbed --clone {env_name} -y")
    elif group.toolchain == "go":
        prepare.append(f"ln -sfn /opt/toolchains/go/{shlex.quote(base.removeprefix('go_'))} /usr/local/go")
    elif group.toolchain == "node":
        prepare.append(f"ln -sfn /opt/toolchains/node-{shlex.quote(base.removeprefix('node_'))} /usr/local/node")
    elif group.toolchain == "rustup":
        prepare.append(f"rustup default {shlex.quote(_rust_toolchain(base))}")
    elif group.toolchain == "jdk":
        prepare.append(f"ln -sfn /opt/toolchains/jdk-{shlex.quote(_jdk_version(base))} /opt/java/current")
    prepared = "".join(line + "\n" for line in prepare)
    clone = CLONE_BLOCK.format(
        workdir=WORKSPACE,
        repo=shlex.quote(flat["repo"]),
        base_commit=shlex.quote(flat["base_commit"]),
    )
    # Idempotency guard: a reused sandbox may hold the previous trial's checkout (and the
    # exec's working directory may even be inside it), so move to a stable directory before
    # clearing the workspace; the clone then requires an empty destination it now has.
    clone = clone.replace(
        f"mkdir -p {WORKSPACE}\n",
        f"cd / && rm -rf {WORKSPACE} && mkdir -p {WORKSPACE}\n",
        1,
    )
    activate = ""
    if group.toolchain == "conda":
        activate = "source /opt/conda/etc/profile.d/conda.sh && conda activate testbed\n"
    installs = "".join(f"( {to_workspace(command, flat['workdir'])} ) || true\n" for command in flat["install"])
    return prepared + clone + activate + installs


def family_config(flat: dict) -> dict:
    """The tests/config.json the contract reads, keeping the upstream grader's keys verbatim.

    The workspace is the uniform /testbed and the test command's absolute references to the
    row's per-repo path are rewritten to it (ruling 2026-11-20, the 10-snapshot consolidation).
    """
    return {
        "workdir": WORKSPACE,
        "base_commit": flat["base_commit"],
        "test_cmd": to_workspace(flat["test_cmd"], flat["workdir"]),
        "test_files": list(flat["test_files"]),
        "log_parser": flat["log_parser"],
        "language": flat["language"],
        "test_ids": list(flat["FAIL_TO_PASS"]) + list(flat["PASS_TO_PASS"]),
        "FAIL_TO_PASS": list(flat["FAIL_TO_PASS"]),
        "PASS_TO_PASS": list(flat["PASS_TO_PASS"]),
        "test_patch": flat["test_patch"],
    }


def task_toml(flat: dict, base: str, group: groups.Group, setup: str) -> str:
    """The repaired native task.toml: shared base image, task-start setup, upstream deadlines."""
    config = {
        "metadata": {
            "author_name": "SWE-rebench V2.1 (Marin repair)",
            "author_email": "ibragim-bad@nebius.com",
            "category": "debugging",
            "tags": list(flat["tags"]),
            "repo": flat["repo"],
            "language": flat["language"],
            "log_parser": flat["log_parser"],
            "quality_flags": list(flat["quality_flags"]),
            "instance_id": flat["instance_id"],
            "declared_base": base,
            "base_group": group.group_id,
        },
        "verifier": {
            "environment_mode": "shared",
            "restart_environment": False,
            "timeout_sec": VERIFIER_TIMEOUT_SEC,
        },
        "agent": {"timeout_sec": AGENT_TIMEOUT_SEC},
        "environment": {
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "storage_mb": STORAGE_MB,
            "build_timeout_sec": BUILD_TIMEOUT_SEC,
            "allow_internet": True,
            "workdir": WORKSPACE,
            "setup": {"command": setup, "timeout_sec": SETUP_TIMEOUT_SEC},
        },
    }
    env = flat.get("env") or []
    if env:
        config["environment"]["env"] = {item["name"]: item["value"] for item in env}
    return tomlkit.dumps(config)


def dockerfile(group: groups.Group, image: str) -> str:
    """The shared base Dockerfile: byte-identical within a group (one snapshot identity).

    No WORKDIR line: the image's default cwd must not be part of the snapshot identity, and
    the runtime cwd comes from the task's uniform /testbed workspace.
    """
    return f"FROM {image}\n"


def base_image(bindings_path: Path, group_id: str) -> str:
    """The digest-pinned shared base image reference for one group."""
    manifest = json.loads(bindings_path.read_text())
    if manifest["dataset"] != groups.UPSTREAM_DATASET or manifest["revision"] != groups.UPSTREAM_REVISION:
        raise ValueError("Base image bindings do not cover this exact source pin")
    reference = manifest["images"].get(group_id)
    if not reference or "@sha256:" not in reference:
        raise ValueError(f"Base image binding for {group_id} is missing or not digest-pinned")
    return reference


def convert_row(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Convert one V2.1 row into a hardened native Harbor package TaskSpec."""
    data = row.data
    if not (data.get("patch") or "").strip():
        return source_defect("missing_gold_patch", "The source does not provide its original solution patch")
    if not (data.get("FAIL_TO_PASS") or data.get("PASS_TO_PASS")):
        return source_defect("missing_test_ids", "The source scorer requires nonempty F2P/P2P test IDs")
    flat = _flatten(data)
    if not flat["test_cmd"] or not flat["log_parser"]:
        return source_defect("missing_test_recipe", "The row lacks its test command or log parser")
    if not flat["workdir"] or not flat["base_commit"]:
        return source_defect("missing_workspace", "The row lacks its workdir or base commit")
    try:
        base = row_base(data)
    except ValueError as error:
        return source_defect("missing_declared_base", str(error))
    group = groups.GROUP_BY_BASE.get(base)
    if group is None:
        return source_defect("unknown_declared_base", f"No consolidated group covers base {base!r}")

    bindings = context.inputs.get(IMAGE_BINDINGS_INPUT)
    if bindings is None:
        return source_defect(
            "base_images_not_staged",
            "The digest-pinned base image bindings are not staged for this conversion",
        )
    staged = StoragePath(str(bindings))
    image = base_image(staged / IMAGE_BINDINGS_FILENAME, group.group_id)

    upstream_files = task_files.task_files(flat)
    setup = setup_command(flat, base, group)
    files: dict[str, bytes] = {
        "task.toml": task_toml(flat, base, group, setup).encode(),
        "environment/Dockerfile": dockerfile(group, image).encode(),
    }
    for name, content in upstream_files.items():
        if name == "task.toml":
            continue
        if name == "tests/config.json":
            files[name] = json.dumps(family_config(flat), indent=1, ensure_ascii=False).encode() + b"\n"
            continue
        if name.startswith("solution/"):
            # The vendored solve script hard-codes the row's per-repo workspace; the oracle's
            # gold-patch delivery runs in the uniform /testbed workspace (ruling 2026-11-20).
            content = to_workspace(content, flat["workdir"])
        files[name] = content.encode()
    contract = shared_source.harden_shared_source(
        files,
        profile=swe_rebench_profile(),
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
    )
    decoded = {
        "files": {name: base64.b64encode(content).decode() for name, content in files.items()},
        "file_metadata": {name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 0} for name in files},
        "archive_links": {},
        "archive_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
    }
    task = native_task(replace(row, data=decoded), context)
    if isinstance(task, ImportRejection):
        return task
    labels = {
        **task.labels,
        "swe_rebench_upstream_revision": groups.UPSTREAM_REVISION,
        "swe_rebench_declared_base": base,
        "swe_rebench_base_group": group.group_id,
    }
    labels.update(admission_receipt_labels(data.get("instance_id"), context.inputs.get(ADMISSION_RECEIPTS_INPUT)))
    return task.model_copy(
        update={
            "integrity_contract": contract,
            "labels": labels,
        }
    )
