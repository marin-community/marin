# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grading-time host delivery for root-actor sources with hidden verifiers.

Sources like RST keep the original root actor semantics: no UID demotion and no
in-container protected root. Their verifier material therefore never exists in
the container during the actor phase. At grading time, after the actor finishes,
two trusted root ``collect`` commands run on the agent machine: one quiesces
every process outside the exec's own ancestry, and the next writes the private
verifier files (base64-embedded in the declared argv) and executes the
source's original ``tests/test.sh`` bytes verbatim, so the source writes its own
reward. Copied artifacts carry the reward to the grading machine; the container
is never trusted to score beyond the source's own script.
"""

import base64
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from taskcompendium.models import VerifierCommand

QUARANTINE_KILL_SIGNAL = "9"
# The delivered payload is argv-embedded; keep a conservative bound so the
# declared command never approaches exec argument limits.
MAX_DELIVERY_BYTES = 512 * 1024
DELIVERY_ROOT = "/tests"
REWARD_PATH = "/logs/verifier/reward.txt"


@dataclass(frozen=True)
class DeliveredVerifier:
    """The private verifier material one source delivers at grading time."""

    files: dict[str, bytes]
    executable_names: frozenset[str] = frozenset({"test.sh"})

    def __post_init__(self) -> None:
        total = sum(len(content) for content in self.files.values())
        if total == 0:
            raise ValueError("Delivered verifier material is empty")
        if total > MAX_DELIVERY_BYTES:
            raise ValueError(
                f"Delivered verifier material exceeds the argv delivery bound: {total} > {MAX_DELIVERY_BYTES}"
            )
        for name in self.files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Delivered verifier path escapes its root: {name}")


def quiesce_command() -> VerifierCommand:
    """Kill every process outside this exec's own ancestry, as the original root."""
    return VerifierCommand(
        argv=(
            "python3",
            "-I",
            "-c",
            QUIESCE_PROGRAM,
        ),
        cwd="/",
    )


def delivery_commands(
    verifier: DeliveredVerifier, *, entrypoint: str = "test.sh", cwd: str | None = None
) -> tuple[VerifierCommand, ...]:
    """Write the private verifier files and run the source's original entry script.

    The commands carry no deadline of their own: the grading layer applies the
    resolved verifier deadline (a root-delivery contract with
    ``undeclared_deadline`` grades unbounded through ``ScriptGrader.timeout = None``).
    """
    if entrypoint not in verifier.files:
        raise ValueError(f"Delivered verifier lacks its entry script: {entrypoint}")
    if entrypoint not in verifier.executable_names:
        raise ValueError(f"The source entry script must stay executable: {entrypoint}")
    writes = []
    for name, content in sorted(verifier.files.items()):
        mode = 0o755 if name in verifier.executable_names else 0o644
        writes.append(
            "printf '%s' "
            + shlex.quote(base64.b64encode(content).decode())
            + " | base64 -d > "
            + shlex.quote(str(PurePosixPath(DELIVERY_ROOT) / name))
            + f"\nchmod {mode:o} "
            + shlex.quote(str(PurePosixPath(DELIVERY_ROOT) / name))
        )
    script = "\n".join(
        (
            "set -eu",
            f"mkdir -p {DELIVERY_ROOT}",
            *writes,
            f"mkdir -p {PurePosixPath(REWARD_PATH).parent}",
            # The source's own script bytes run verbatim and write their own reward.
            "bash " + shlex.quote(str(PurePosixPath(DELIVERY_ROOT) / entrypoint)),
        )
    )
    return (VerifierCommand(argv=("bash", "-c", script), cwd=cwd),)


QUIESCE_PROGRAM = """import os, signal
mine = set()
pid = os.getpid()
while pid > 1:
    mine.add(pid)
    try:
        with open(f"/proc/{pid}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        pid = int(fields[1])
    except (OSError, IndexError, ValueError):
        break
killed = []
for entry in os.scandir("/proc"):
    if not entry.name.isdecimal():
        continue
    victim = int(entry.name)
    if victim in mine or victim == 1:
        continue
    try:
        with open(f"/proc/{victim}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        state = fields[0]
    except (OSError, IndexError, ValueError):
        continue
    if state.startswith("Z"):
        continue
    try:
        os.kill(victim, signal.SIGKILL)
        killed.append(victim)
    except (OSError, ProcessLookupError):
        pass
print(len(killed))
"""
