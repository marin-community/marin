# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive pinned, harness-bootstrapped SWE packages for per-task CPU admission."""

import base64
import hashlib
import json
import re
import shlex
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import tomli_w
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.long_loop.pi_rl_data import SOURCE
from experiments.post_training.long_loop.swe_grading import TaskPins

RUNTIME = Path(__file__).with_name("swe_grading.py")
GUEST_RUNTIME = "/opt/long-loop-swe/grading.py"
RECIPE_VERSION = "swe-trusted-worktree-v4"
VERIFYIT_REVISION = "04cee3398e60b9d879396b9bd10be089d56c6f10"
SETUP_BLOCK = re.compile(r"\A## Environment Setup[^\n]*\n\s*```bash\n(?P<setup>.*?)\n```\s*\n---\s*\n", re.DOTALL)
CLONE = re.compile(r"\Agit clone (?P<repository>https://github.com/swesmith/[^\s]+) \.\Z")
CHECKOUT = re.compile(r"\Agit checkout (?P<branch>[\w.\-/]+)\Z")
BUG = re.compile(r"cat-file -e (?P<sha>[0-9a-f]{40})\^\{commit\}")
REFERENCE = re.compile(r'elif \[ -n "(?P<sha>[0-9a-f]{40})" \]')
BOOTSTRAP_NOTE = (
    "## Environment setup\n\n"
    "The harness has cloned the specified repository into /testbed, checked out the task's bug commit, "
    "and run the declared dependency setup. Work in that checkout and fix the issue below. "
    "The original task Python remains available for tools and additional dependency installs.\n\n---\n\n"
)
IMAGE_LAYER = f"""
# Preserve the upstream task environment; isolate only grading and control code.
COPY long-loop-swe/grading.py /opt/long-loop-swe/grading.py
RUN useradd --create-home --user-group --uid 10002 --shell /bin/bash longloop-grade \
    && chmod -R go-w /opt/long-loop-swe \
    && command -v setpriv
RUN UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --force --python ">=3.11" \
    "marin-verifyit @ git+https://github.com/marin-community/marin@{VERIFYIT_REVISION}#subdirectory=lib/verifyit"
"""


@dataclass(frozen=True)
class PreparedTask:
    task: TaskSpec
    pins: TaskPins
    original_instruction_sha256: str
    instruction_sha256: str


def task_pins(
    task: TaskSpec, *, setup_timeout: int, verifier_timeout: int, dependency_constraints: str
) -> tuple[TaskPins, str]:
    """Recover immutable source pins and declared setup without guessing a reference."""
    files = {
        r.path: resource_bytes(r).decode() for r in task.resources.verifier if not r.path.endswith("package-lock.json")
    }
    instruction = files["instruction.md"]
    block = SETUP_BLOCK.match(instruction)
    if block is None:
        raise ValueError(f"{task.id}: unsupported upstream setup block")
    lines = block["setup"].splitlines()
    if len(lines) < 4 or lines[0] != "cd /testbed":
        raise ValueError(f"{task.id}: unexpected setup prefix")
    clone, checkout = CLONE.fullmatch(lines[1]), CHECKOUT.fullmatch(lines[2])
    verifier = tomllib.loads(files["tests/verifier.toml"])
    bug = BUG.search(verifier["setup"])
    reference = REFERENCE.search(verifier["setup"])
    if clone is None or checkout is None or bug is None or reference is None:
        raise ValueError(f"{task.id}: missing exact repository/bug/reference pins")
    pins = TaskPins(
        clone["repository"],
        bug["sha"],
        reference["sha"],
        checkout["branch"],
        "\n".join(lines[3:]),
        dependency_constraints,
        setup_timeout,
        verifier_timeout,
    )
    return pins, BOOTSTRAP_NOTE + instruction[block.end() :]


def prepare_task(
    task: TaskSpec,
    *,
    setup_timeout: int,
    verifier_timeout: int,
    harbor_verifier_timeout: int,
    dependency_constraints: str,
) -> PreparedTask:
    """Build one versioned package; acceptance still requires bug=0/reference=1."""
    if task.source.dataset != SOURCE:
        raise ValueError("Only the pinned SWE source may receive this verifier version")
    pins, instruction = task_pins(
        task,
        setup_timeout=setup_timeout,
        verifier_timeout=verifier_timeout,
        dependency_constraints=dependency_constraints,
    )
    files = {r.path: resource_bytes(r) for r in task.resources.verifier}
    config = tomllib.loads(files["task.toml"].decode())
    payload = base64.b64encode(json.dumps(asdict(pins)).encode()).decode()
    config.setdefault("environment", {})["setup"] = {
        "command": f"python3 -I -S {GUEST_RUNTIME} bootstrap {shlex.quote(payload)}",
        "timeout_sec": setup_timeout,
    }
    config["verifier"]["timeout_sec"] = harbor_verifier_timeout
    config["verifier"]["user"] = "root"
    verifier = tomllib.loads(files["tests/verifier.toml"].decode())
    verifier["setup"] = ""
    verifier["timeout"] = verifier_timeout
    if verifier["python"] != "python3":
        raise ValueError("Unsupported task interpreter; do not substitute its environment")
    verifier["python"] = "/opt/long-loop-trusted/grade-python"
    verifier["protected_paths_files"] = [verifier.pop("protected_paths_file")]
    replacement = {
        "instruction.md": instruction.encode(),
        "task.toml": tomli_w.dumps(config).encode(),
        "tests/verifier.toml": tomli_w.dumps(verifier).encode(),
        "tests/test.sh": (
            (
                "#!/bin/bash\nset -euo pipefail\n"
                f"exec /opt/long-loop-control-bin/python3 -I -S {GUEST_RUNTIME} "
                "grade --tests /tests --logs /logs/verifier\n"
            ).encode()
        ),
        "environment/Dockerfile": files["environment/Dockerfile"] + IMAGE_LAYER.encode(),
        "environment/long-loop-swe/grading.py": RUNTIME.read_bytes(),
    }
    events = tuple(
        (
            event.model_copy(update={"content": instruction})
            if isinstance(event, TextMessage)
            and event.role == "user"
            and event.content == files["instruction.md"].decode()
            else event
        )
        for event in task.context.events
    )
    if events == task.context.events:
        raise ValueError("Original task instruction did not match the emitted prompt")
    resources = tuple(r for r in task.resources.verifier if r.path not in replacement) + tuple(
        inline_resource(p, b) for p, b in replacement.items()
    )
    prepared = task.model_copy(
        update={
            "context": task.context.model_copy(update={"events": events}),
            "resources": task.resources.model_copy(update={"verifier": resources}),
            "source": task.source.model_copy(update={"importer_revision": RECIPE_VERSION}),
        }
    )
    return PreparedTask(
        prepared,
        pins,
        hashlib.sha256(files["instruction.md"]).hexdigest(),
        hashlib.sha256(instruction.encode()).hexdigest(),
    )
