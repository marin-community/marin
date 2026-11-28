# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the Lego-X Terminal-Lego-15k capability and verifier defects.

The qualified release carries two defects measured on the arm64 LOCAL route
(coordinator directive 2026-11-27):

1. Every task declares ``environment.capabilities = ()`` — the consumer advertises
   no tools, so every rollout scores 0. The tasks are terminal tasks driven by
   shell commands; the RST family's working contract declares ``("shell",)``.
2. The source verifier installs its toolchain at grading time
   (``apt-get update``, ``apt-get install curl``, the uv installer, ``uvx``).
   In a sandboxed guest apt's unprivileged ``_apt`` fetch sandbox cannot drop
   privileges (``setgroups: Operation not permitted`` — the guest root keeps uid 0
   but runs without ``CAP_SETGID``), so apt fails, curl/uv/pytest never appear and
   a correct workspace still scores 0. The RL route's local verifier machines run
   with network allowed, so the source's own chain works once the apt fetch
   sandbox is disabled for the verifier run. The uv installer also fails in that
   guest: it extracts its tarball with ``tar`` as root, and restoring the upstream
   ownership (uid 1001) needs ``CAP_CHOWN`` the guest does not keep, so ``uv`` /
   ``uvx`` still never appear. The repair pre-installs the pinned uv release with
   ``tar --no-same-owner`` onto the default PATH, leaving the source chain intact.

The repair is surgical: the canary header stays byte-identical, the apt fetch
sandbox is disabled via a drop-in apt configuration, the RST-style missing-reward
trap keeps a crashed run graded at 0 instead of unparseable, a ``/app``
fallback keeps the verifier's own working-directory check from stranding guests
whose entry working directory is ``/``, and the pinned uv release is pre-installed
sandbox-safe (``tar --no-same-owner``) because the source installer's tar cannot
restore ownership in the guest. The patched verifier bytes replace both
the live capture archive inside the grader's collect command and the archived
``source_archive/tests/test.sh`` resource. The 15 x86-only contexts are NOT
repaired here; their rows stay byte-identical pending the user ruling.
"""

import base64
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import PurePosixPath

from pydantic import BaseModel
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

REPAIR_VERSION = "terminal-lego-repair-v2"
LINEAGE_PATH = "taskcompendium/terminal-lego-repair.json"
PATCH_MARKER = f"harbor-sandbox-patched ({REPAIR_VERSION})"
SHELL_CAPABILITIES = ("shell",)
CAPTURE_MARKER = ".capture.tar.gz"
CAPTURE_PREFIX = "mkdir -p /tests && printf '%s' "
CAPTURE_SUFFIX = f" | base64 -d > /tests/{CAPTURE_MARKER}"
TEST_SCRIPT_BASENAME = "test.sh"
UV_INSTALLER_RE = re.compile(r"astral\.sh/uv/([0-9.]+)/install\.sh")

PATCH_HEADER = (
    f"# {PATCH_MARKER}: the verifier installs its toolchain at grading time, but in a\n"
    "# sandboxed guest apt's unprivileged _apt fetch sandbox cannot drop privileges\n"
    "# (setgroups: Operation not permitted), which left curl/uv/pytest missing and scored 0\n"
    "# even on a correct workspace. The apt fetch sandbox is disabled for this verifier run,\n"
    "# the source uv installer's tar cannot restore upstream ownership in the guest\n"
    "# (chown: Operation not permitted), so uv is pre-installed sandbox-safe below, and\n"
    "# a crashed run still writes its reward (graded 0) instead of leaving none.\n"
)
SANDBOX_LINE = "echo 'APT::Sandbox::User \"root\";' > /etc/apt/apt.conf.d/99harbor-sandbox 2>/dev/null || true\n"
TRAP_BLOCK = (
    "mkdir -p /logs/verifier\n"
    "_harbor_write_missing_reward() {\n"
    "  if [ ! -f /logs/verifier/reward.txt ] && [ ! -f /logs/verifier/reward.json ]; then\n"
    "    echo 0 > /logs/verifier/reward.txt\n"
    "  fi\n"
    "}\n"
    "trap _harbor_write_missing_reward EXIT\n"
    'if [ "$PWD" = "/" ]; then cd /app 2>/dev/null || true; fi\n'
)


def _uv_preinstall_block(version: str) -> str:
    """A sandbox-safe install of the pinned uv release ahead of the source chain.

    The source installer's ``tar`` runs as root and tries to restore the tarball's
    upstream ownership (uid 1001), which the sandboxed guest cannot do, so the
    install fails and ``uvx`` never appears. Extracting with ``--no-same-owner``
    onto ``/usr/local/bin`` (already on the default PATH) avoids the chown; a
    failure here is non-fatal and leaves the source chain to proceed as before.
    """
    return (
        "_harbor_uv_preinstall() {\n"
        '  case "$(uname -m)" in\n'
        "    x86_64) _harbor_uv_target=x86_64-unknown-linux-gnu ;;\n"
        "    aarch64|arm64) _harbor_uv_target=aarch64-unknown-linux-gnu ;;\n"
        "    *) return 0 ;;\n"
        "  esac\n"
        "  _harbor_uv_tar=$(mktemp)\n"
        '  curl -LsSf "https://github.com/astral-sh/uv/releases/download/'
        + version
        + '/uv-${_harbor_uv_target}.tar.gz" -o "$_harbor_uv_tar" || return 0\n'
        '  tar -xzf "$_harbor_uv_tar" --strip-components 1 --no-same-owner -C /usr/local/bin || return 0\n'
        '  rm -f "$_harbor_uv_tar"\n'
        "}\n"
        "_harbor_uv_preinstall || true\n"
    )


class TerminalLegoTaskLineage(BaseModel):
    """The recorded repair of one task's capability and verifier defects."""

    version: str
    capabilities_before: list[str]
    capabilities_after: list[str]
    capture_sha256_before: str
    capture_sha256_after: str
    test_script_sha256_before: str
    test_script_sha256_after: str
    input_task_sha256: str


def _patch_insert_offset(script: str) -> int:
    """Byte offset just past the shebang, where the patch is inserted.

    The canary lines precede the shebang in the surveyed verifiers; inserting after
    the shebang keeps both the canary header and the interpreter line byte-identical.
    """
    lines = script.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("#!"):
            return sum(len(line) for line in lines[: index + 1])
    raise ValueError("The source verifier carries no shebang line to patch after")


def repaired_test_script(original: str) -> str:
    """Patch the verifier while keeping its canary header and body verbatim.

    The sandbox line, the reward trap, the ``/app`` fallback and — when the source
    installs uv via the astral installer — the sandbox-safe uv pre-install are
    inserted right after the shebang; everything the source wrote follows
    unchanged, so per-task toolchain variations keep their exact bytes. Already
    patched scripts pass through unchanged (idempotent re-runs).
    """
    if PATCH_MARKER in original:
        return original
    offset = _patch_insert_offset(original)
    uv_version = UV_INSTALLER_RE.search(original)
    uv_block = _uv_preinstall_block(uv_version.group(1)) if uv_version else ""
    return original[:offset] + "\n" + PATCH_HEADER + SANDBOX_LINE + TRAP_BLOCK + uv_block + original[offset:]


def _capture_of(argv: tuple[str, ...]) -> bytes | None:
    """The gzip-compressed tar capture embedded in a collect command's argv."""
    embedded = next((part for part in argv if part.endswith(CAPTURE_SUFFIX)), None)
    if embedded is None:
        return None
    if not embedded.startswith(CAPTURE_PREFIX):
        raise ValueError("The capture collect command has an unexpected shape")
    return base64.b64decode(embedded[len(CAPTURE_PREFIX) : -len(CAPTURE_SUFFIX)])


def _repack_capture(capture: bytes) -> bytes:
    """Patch the verifier script inside a capture archive; deterministic bytes."""
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(capture))) as tar:
        members = [(member, tar.extractfile(member).read() if member.isfile() else None) for member in tar]
    replaced = 0
    patched = []
    for member, content in members:
        if member.isfile() and PurePosixPath(member.name).name == TEST_SCRIPT_BASENAME:
            script = content.decode()
            updated = repaired_test_script(script)
            replaced += bool(updated != script)
            content = updated.encode()
        patched.append((member, content))
    if not any(member.isfile() and PurePosixPath(member.name).name == TEST_SCRIPT_BASENAME for member, _ in members):
        raise ValueError("The capture archive carries no verifier script to patch")
    if not replaced:
        return capture  # already patched: the bytes stay identical
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for member, content in patched:
            info = tarfile.TarInfo(member.name)
            info.size = len(content) if content is not None else 0
            info.mode = member.mode
            info.mtime = 0
            if content is not None:
                tar.addfile(info, io.BytesIO(content))
            else:
                tar.addfile(info)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def repair_terminal_lego_task(task: TaskSpec) -> TaskSpec:
    """Grant the shell capability and patch the verifier; record the lineage.

    The task passes through unchanged when the capability is present and the
    verifier is already patched (idempotent re-runs).
    """
    if not isinstance(task.grader, ScriptGrader) or task.grader.environment is None:
        raise ValueError(f"{task.id}: the Lego-X repair expects the script-grader root-delivery contract")
    input_sha = task_spec_sha256(task)
    capabilities_before = list(task.environment_requirements.capabilities)
    collect = []
    capture_shas = None
    for command in task.grader.collect:
        capture = _capture_of(command.argv)
        if capture is None:
            collect.append(command)
            continue
        if capture_shas is not None:
            raise ValueError(f"{task.id}: the grader carries more than one capture command")
        patched = _repack_capture(capture)
        capture_shas = (hashlib.sha256(capture).hexdigest(), hashlib.sha256(patched).hexdigest())
        argv = tuple(
            (
                CAPTURE_PREFIX + base64.b64encode(patched).decode() + CAPTURE_SUFFIX
                if part.endswith(CAPTURE_SUFFIX)
                else part
            )
            for part in command.argv
        )
        collect.append(command.model_copy(update={"argv": argv}))
    if capture_shas is None:
        raise ValueError(f"{task.id}: the grader carries no capture command to patch")
    script_shas = None
    verifier = []
    for resource in task.resources.verifier:
        if PurePosixPath(resource.path).name == TEST_SCRIPT_BASENAME:
            script = resource_bytes(resource).decode()
            script_shas = (
                hashlib.sha256(script.encode()).hexdigest(),
                hashlib.sha256(repaired_test_script(script).encode()).hexdigest(),
            )
            verifier.append(
                inline_resource(resource.path, repaired_test_script(script).encode()).model_copy(
                    update={"mode": resource.mode, "mtime_ns": resource.mtime_ns}
                )
            )
        else:
            verifier.append(resource)
    capabilities_after = sorted({*capabilities_before, *SHELL_CAPABILITIES})
    if capabilities_after == capabilities_before and capture_shas[0] == capture_shas[1]:
        return task
    lineage = TerminalLegoTaskLineage(
        version=REPAIR_VERSION,
        capabilities_before=capabilities_before,
        capabilities_after=capabilities_after,
        capture_sha256_before=capture_shas[0],
        capture_sha256_after=capture_shas[1],
        test_script_sha256_before=script_shas[0] if script_shas else "",
        test_script_sha256_after=script_shas[1] if script_shas else "",
        input_task_sha256=input_sha,
    )
    environment = task.environment_requirements.model_copy(update={"capabilities": tuple(capabilities_after)})
    grader = task.grader.model_copy(update={"collect": tuple(collect)})
    resources = task.resources.model_copy(
        update={
            "verifier": tuple(verifier),
            "oracle": (
                *(resource for resource in task.resources.oracle if resource.path != LINEAGE_PATH),
                inline_resource(LINEAGE_PATH, lineage.model_dump_json().encode()),
            ),
        }
    )
    repaired = task.model_copy(
        update={
            "environment_requirements": environment,
            "grader": grader,
            "resources": resources,
            "labels": {**task.labels, "terminal_lego_repair_version": REPAIR_VERSION},
        }
    )
    contract = repaired.integrity_contract
    if contract is not None and getattr(contract, "verifier_files", None):
        # The repair rewrote the verifier bytes: re-pin the contract's verifier_files to the
        # patched resources' content and mode so the root-delivery integrity check stays honest.
        repinned = tuple(
            file.model_copy(
                update={
                    "sha256": next(
                        (
                            hashlib.sha256(resource_bytes(resource)).hexdigest()
                            for resource in repaired.resources.verifier
                            if resource.path.split("/")[-1] == file.name.split("/")[-1]
                        ),
                        file.sha256,
                    )
                }
            )
            for file in contract.verifier_files
        )
        repaired = repaired.model_copy(
            update={"integrity_contract": contract.model_copy(update={"verifier_files": repinned})},
        )
    return TaskSpec.model_validate_json(repaired.model_dump_json())


def lineage_of(task: TaskSpec) -> TerminalLegoTaskLineage | None:
    """Read a task's recorded Lego-X repair lineage, or None when the task was not repaired."""
    for resource in task.resources.oracle:
        if resource.path == LINEAGE_PATH:
            return TerminalLegoTaskLineage.model_validate_json(resource_bytes(resource))
    return None


def lineage_json(lineage: TerminalLegoTaskLineage) -> str:
    """The release-level lineage document entry for one repaired task."""
    return json.dumps(lineage.model_dump(), indent=1)
