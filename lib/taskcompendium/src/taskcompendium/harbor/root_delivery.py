# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grading-time host delivery for root-actor sources with hidden verifiers.

Sources like RST and Multi-SWE keep the original root actor semantics: no UID
demotion and no in-container protected root. Their private grading material
therefore never exists in the container during the actor phase. At grading
time, after the actor finishes, trusted root ``collect`` commands run on the
agent machine: one quiesces every process outside the exec's own ancestry,
and the next writes the private grading material (base64-embedded in the
declared argv, delivered under ``/tests`` for RST and restored from the
original compressed capture archive under ``/home`` for Multi-SWE), then
executes the source's own pipeline verbatim so the source produces its own
reward inputs. Copied artifacts carry the reward inputs to the grading
machine; the container is never trusted to score beyond the source's own
parser, which runs host-side.
"""

import base64
import gzip
import io
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from taskcompendium.models import (
    ArtifactKind,
    EnvironmentRequirements,
    FileReward,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskResource,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.staging import resource_files, write_archive

QUARANTINE_KILL_SIGNAL = "9"
# The delivered payload is argv-embedded; keep a conservative bound so the
# declared command never approaches exec argument limits.
MAX_DELIVERY_BYTES = 512 * 1024
DELIVERY_ROOT = "/tests"
REWARD_PATH = "/logs/verifier/reward.txt"
TEST_OUTPUT = "/logs/verifier/test-output.txt"
TEST_EXIT = "/logs/verifier/exit-code.txt"
COPIED_REWARD = "/source-reward.txt"


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
    """Kill every process outside this exec's own ancestry, as the original root.

    The machine's own lifecycle sleeper (``SANDBOX_KEEPER_SLEEP_ARGS``) is
    spared alongside PID 1: on the local tier it is the sandbox's PID 2 and
    killing it closes the machine mid-grading.
    """
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
    verifier: DeliveredVerifier,
    *,
    entrypoint: str = "test.sh",
    cwd: str | None = None,
    output_path: str | None = None,
    exit_code_path: str | None = None,
) -> tuple[VerifierCommand, ...]:
    """Write the private verifier files and run the source's original entry script.

    The commands carry no deadline of their own: the grading layer applies the
    resolved verifier deadline (a root-delivery contract with
    ``undeclared_deadline`` grades unbounded through ``ScriptGrader.timeout = None``).
    ``output_path`` redirects the entry script's stdout and stderr into that file
    (an absolute guest path) instead of the collect result, so the caller can
    declare the file as a copied artifact; the source's scoring bytes stay unchanged.
    ``exit_code_path`` selects the native Harbor reward-file result mode: the
    script runs unchanged even on failure, its exact exit code is persisted to
    that file, and the collect command finishes 0 so the source's own reward
    (trap-written even on a failing run) reaches the reward parser; a missing
    or invalid reward file still fails through the caller's reward parsing.
    Without ``exit_code_path`` the command keeps the strict behavior: a nonzero
    script exit fails the collect command.
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
    if output_path is not None and not PurePosixPath(output_path).is_absolute():
        raise ValueError(f"The delivery output path must be absolute: {output_path}")
    if exit_code_path is not None and not PurePosixPath(exit_code_path).is_absolute():
        raise ValueError(f"The delivery exit-code path must be absolute: {exit_code_path}")
    entry = "bash " + shlex.quote(str(PurePosixPath(DELIVERY_ROOT) / entrypoint))
    if output_path is not None:
        entry += " > " + shlex.quote(output_path) + " 2>&1"
    if exit_code_path is not None:
        # Native reward-file result mode: the source's exact exit code is
        # persisted, the collect command finishes 0, and the source's own
        # trap-written reward reaches the reward parser even on a failing run.
        # The || assignment keeps the failing entry off `set -e`'s exit path.
        entry += f" || status=$?; printf '%s\\n' \"${{status-0}}\" > {shlex.quote(exit_code_path)}; exit 0"
    parents = {PurePosixPath(REWARD_PATH).parent}
    if output_path is not None:
        parents.add(PurePosixPath(output_path).parent)
    if exit_code_path is not None:
        parents.add(PurePosixPath(exit_code_path).parent)
    script = "\n".join(
        (
            "set -eu",
            f"mkdir -p {DELIVERY_ROOT}",
            *writes,
            *(f"mkdir -p {parent}" for parent in sorted(parents)),
            # The source's own script bytes run verbatim and write their own reward.
            entry,
        )
    )
    return (VerifierCommand(argv=("bash", "-c", script), cwd=cwd),)


SANDBOX_KEEPER_SLEEP_ARGS = ("sleep", "2147483647")
"""The machine keeper's payload argv: shellbox's local sandbox runs
``exec sleep 2147483647`` under bubblewrap to hold the machine open.

On the LOCAL tier the sandbox's PID 1 is bwrap itself and this sleeper is PID 2,
so a quiesce that spares only PID 1 kills the keeper: bwrap exits 137, the
machine closes, and the attempt dies as ``Machine is closed`` mid-grading
(facet_v2's 6,122+ unscored attempts, 2026-11-24). Both quiesce variants
therefore spare exactly this argv. A task that reused the same argv would
merely survive one quiesce: the machine is torn down right after grading, so
the residual risk is bounded and tiny.
"""

QUIESCE_PROGRAM = (
    "import os, signal\n"
    "KEEPER_SLEEP_ARGS = " + repr(SANDBOX_KEEPER_SLEEP_ARGS) + "\n"
    "mine = set()\n"
    "pid = os.getpid()\n"
    "while pid > 1:\n"
    "    mine.add(pid)\n"
    "    try:\n"
    "        with open(f\"/proc/{pid}/stat\") as stream:\n"
    "            fields = stream.read().rsplit(\") \", 1)[1].split()\n"
    "        pid = int(fields[1])\n"
    "    except (OSError, IndexError, ValueError):\n"
    "        break\n"
    "killed = []\n"
    "for entry in os.scandir(\"/proc\"):\n"
    "    if not entry.name.isdecimal():\n"
    "        continue\n"
    "    victim = int(entry.name)\n"
    "    if victim in mine or victim == 1:\n"
    "        continue\n"
    "    try:\n"
    "        with open(f\"/proc/{victim}/stat\") as stream:\n"
    "            fields = stream.read().rsplit(\") \", 1)[1].split()\n"
    "        state = fields[0]\n"
    "    except (OSError, IndexError, ValueError):\n"
    "        continue\n"
    "    if state.startswith(\"Z\"):\n"
    "        continue\n"
    "    try:\n"
    "        with open(f\"/proc/{victim}/cmdline\", \"rb\") as stream:\n"
    "            argv = stream.read().split(b\"\\x00\")\n"
    "    except OSError:\n"
    "        argv = []\n"
    "    if tuple(argv[: len(KEEPER_SLEEP_ARGS)]) == KEEPER_SLEEP_ARGS:\n"
    "        continue\n"
    "    try:\n"
    "        os.kill(victim, signal.SIGKILL)\n"
    "        killed.append(victim)\n"
    "    except ProcessLookupError:\n"
    "        continue\n"
    "    except OSError as error:\n"
    "        # A failed kill means the victim may survive: the quiesce cannot be\n"
    "        # guaranteed, so the advisory must fail instead of passing silently.\n"
    "        print(f\"quiesce kill failed for {victim}: {error}\", file=sys.stderr)\n"
    "        raise\n"
    "print(len(killed))\n"
)


# Multi-SWE delivery: the original compressed capture archive (exact modes and
# mtimes inside) is delivered in bounded base64 chunks and restored verbatim
# under the home root, then the source's own extract-and-run pipeline executes
# there. All guest commands are pure POSIX shell + GNU coreutils: the source
# images span C/Go/Java/JS/TS and guarantee no interpreter.

HOME_RESTORE_CHUNK_BYTES = 48 * 1024
"""Raw bytes per chunk; base64 renders ~64 KiB, well inside the argv cap."""


def archive_chunk_commands(archive: bytes, restore_root: str, chunk_bytes: int) -> tuple[VerifierCommand, ...]:
    """Chunk a gzip archive onto the machine as base64 argv pieces under ``restore_root``.

    Renamed public for the cohort conversion's tests-tree restore; the encoding
    is unchanged.
    """
    if not archive:
        raise ValueError("The home delivery archive is empty")
    encoded = base64.b64encode(archive)
    staging = PurePosixPath(restore_root) / ".capture.tar.gz"
    encoded_chunk = max(4, chunk_bytes * 4 // 3 // 4 * 4)
    commands: list[VerifierCommand] = []
    for index in range(0, len(encoded), encoded_chunk):
        chunk = encoded[index : index + encoded_chunk]
        # The first chunk truncates: staging always starts fresh.
        redirect = ">" if index == 0 else ">>"
        commands.append(
            VerifierCommand(
                argv=(
                    "bash",
                    "-c",
                    "mkdir -p "
                    + shlex.quote(restore_root)
                    + " && printf '%s' "
                    + shlex.quote(chunk.decode("ascii"))
                    + f" | base64 -d {redirect} "
                    + shlex.quote(str(staging)),
                ),
                cwd="/",
            )
        )
    return tuple(commands)


def home_restore_commands(
    archive: bytes, *, restore_root: str = "/home", chunk_bytes: int = HOME_RESTORE_CHUNK_BYTES
) -> tuple[VerifierCommand, ...]:
    """Deliver the original compressed capture archive in chunks and restore it verbatim.

    The archive carries the exact member modes and mtimes, so ``tar`` reproduces
    them without any per-file repair. Each chunk renders under ~64 KiB of argv
    (encoded chunks are multiples of four so ``base64 -d`` decodes them
    standalone); the staging file is removed after the restore.
    """
    commands = archive_chunk_commands(archive, restore_root, chunk_bytes)
    staging = PurePosixPath(restore_root) / ".capture.tar.gz"
    # Reproduce the source's RESTORE_GRADING_ARTIFACTS exactly (taskset.py at
    # pinned b41d875): each archive member must be a plain relative filename and
    # its destination is removed (rm -rf) before extraction, so a destination the
    # actor turned into a directory cannot survive; then extract and clean up.
    restore = "\n".join(
        (
            "set -euo pipefail",
            f"tar -tzf {shlex.quote(str(staging))} | while IFS= read -r artifact; do",
            '  case "$artifact" in',
            '    ""|/*|*/*)',
            '    echo "unexpected Multi-SWE grading artifact path: $artifact" >&2',
            "    exit 1",
            "    ;;",
            "  esac",
            f'  rm -rf -- "{restore_root}/$artifact"',
            "done",
            f"tar -C {shlex.quote(restore_root)} -xzf {shlex.quote(str(staging))}",
            f"rm -f {shlex.quote(str(staging))}",
        )
    )
    return (*commands, VerifierCommand(argv=("bash", "-c", restore), cwd="/"))


def multiswe_run_command(
    *,
    repo: str,
    base: str,
    output_path: str,
    read_exit_code_path: str,
    pipeline_stdout_path: str,
    pipeline_stderr_path: str,
    pipeline_rc_path: str,
    cat_rc_path: str,
    restore_root: str = "/home",
    extract_entry: str = "extract_multiswe_fix_patch.sh",
    run_entry: str = "fix-run.sh",
    test_output_path: str | None = None,
) -> VerifierCommand:
    """Run the source's extract-and-run pipeline with independently captured streams.

    Faithful to the upstream taskset: extract runs at ``/home/{repo}`` with the
    literal ``.`` argument and the row's base commit (prep nonzero fails the
    collection), the pipeline's exit is ignored with its stdout and stderr
    captured separately, and the exact ``cat`` stdout and exit code of
    ``/home/test_output.txt`` are persisted for the host-side selection. The
    container never scores: the host parser consumes the copied artifacts.
    """
    if not repo or not base:
        raise ValueError("Multi-SWE delivery requires the row repo and base commit")
    for path in (
        output_path,
        read_exit_code_path,
        pipeline_stdout_path,
        pipeline_stderr_path,
        pipeline_rc_path,
        cat_rc_path,
    ):
        if not PurePosixPath(path).is_absolute():
            raise ValueError(f"The Multi-SWE delivery paths must be absolute: {path}")
    root = restore_root.rstrip("/")
    workdir = f"{root}/{repo}"
    extract = f"{root}/{extract_entry}"
    pipeline = f"{root}/{run_entry}"
    test_output = test_output_path or f"{root}/test_output.txt"
    script = "\n".join(
        (
            "set -eu",
            f"cd {shlex.quote(workdir)}",
            f"chmod +x {shlex.quote(extract)}",
            f"bash {shlex.quote(extract)} . {shlex.quote(base)}",
            "set +e",
            f"bash -o pipefail -c 'bash {shlex.quote(pipeline)} 2>&1 | tee {shlex.quote(test_output)}'"
            f" > {shlex.quote(pipeline_stdout_path)} 2> {shlex.quote(pipeline_stderr_path)}",
            "pipeline_rc=$?",
            f"cat {shlex.quote(test_output)} > {shlex.quote(output_path)}",
            "cat_rc=$?",
            "set -e",
            f"printf '%s\\n' \"$pipeline_rc\" > {shlex.quote(pipeline_rc_path)}",
            f"printf '%s\\n' \"$cat_rc\" > {shlex.quote(cat_rc_path)}",
            f"printf '%s\\n' \"$cat_rc\" > {shlex.quote(read_exit_code_path)}",
            "exit 0",
        )
    )
    return VerifierCommand(argv=("bash", "-c", script), cwd="/")


SHELL_QUIESCE_SCRIPT = """set -eu
self=$$
ancestry=" $self "
pid=$self
while [ "$pid" -gt 1 ]; do
  line=$(cat "/proc/$pid/stat" 2>/dev/null) || break
  rest=${line##*) }
  state=${rest%% *}
  ppid=${rest#* }
  ppid=${ppid%% *}
  case $ppid in ''|*[!0-9]*) break;; esac
  ancestry="$ancestry $ppid "
  pid=$ppid
done
killed=0
for entry in /proc/[0-9]*; do
  victim=${entry#/proc/}
  case $ancestry in *" $victim "*) continue;; esac
  [ "$victim" -gt 1 ] || continue
  line=$(cat "/proc/$victim/stat" 2>/dev/null) || continue
  rest=${line##*) }
  case ${rest%% *} in Z*) continue;; esac
  # The machine keeper's payload (see SANDBOX_KEEPER_SLEEP_ARGS): the local
  # sandbox runs `exec sleep 2147483647` as the init's child, so killing it
  # closes the machine mid-grading.
  args=$(tr '\\0' '\\n' < "/proc/$victim/cmdline" 2>/dev/null || true)
  if [ "$args" = "$(printf '%s\\n%s' sleep 2147483647)" ]; then
    continue
  fi
  if kill -9 "$victim" 2>/dev/null; then
    killed=$((killed + 1))
  elif [ ! -d "/proc/$victim" ]; then
    continue
  else
    echo "quiesce kill failed for $victim" >&2
    exit 1
  fi
done
echo "$killed"
"""


def shell_quiesce_command() -> VerifierCommand:
    """Quiesce with pure shell for images that ship no guest interpreter.

    Same ancestry semantics as :func:`quiesce_command`: every process outside
    the exec's own ancestry is killed (PID 1, the machine keeper's sleeper, the
    exec's own tree and zombies spared), and a failed kill fails the command
    instead of silently passing.
    """
    return VerifierCommand(argv=("bash", "-c", SHELL_QUIESCE_SCRIPT), cwd="/")


def native_file_reward_grader(
    environment: EnvironmentRequirements,
    private_files: tuple[TaskResource, ...],
    execute: VerifierCommand,
    timeout: float,
) -> ScriptGrader:
    """Restore native private files after the actor and collect its numeric reward.

    File order, contents, modes and timestamps are retained in the delivered
    archive. The caller supplies the source's execution command and deadline.
    """
    with TemporaryDirectory(prefix="native-verifier-") as temporary:
        archive = Path(temporary) / "tests.tar"
        write_archive(archive, resource_files(private_files, ""), [])
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as writer:
            writer.write(archive.read_bytes())
    restore = home_restore_commands(compressed.getvalue(), restore_root="/tests")
    if any("/" in resource.path for resource in private_files):
        roots = set()
        for resource in private_files:
            path = PurePosixPath(resource.path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Native verifier path escapes its private root: {resource.path}")
            roots.add(path.parts[0])
        # Replace each private top-level member before tar extraction so an
        # actor-created directory or parent symlink cannot redirect the upload.
        script = "\n".join(
            (
                "set -euo pipefail",
                *("rm -rf -- " + shlex.quote("/tests/" + name) for name in sorted(roots)),
                "tar -C /tests -xzf /tests/.capture.tar.gz",
                "rm -f /tests/.capture.tar.gz",
            )
        )
        restore = (
            *archive_chunk_commands(compressed.getvalue(), "/tests", HOME_RESTORE_CHUNK_BYTES),
            VerifierCommand(argv=("bash", "-c", script), cwd="/"),
        )
    transport = environment.model_copy(
        update={"working_directory": "/", "setup_commands": (), "environment_variables": {}}
    )
    return ScriptGrader(
        argv=("cp", COPIED_REWARD, REWARD_PATH),
        cwd="/",
        environment=transport,
        answer_path=None,
        collect=(
            shell_quiesce_command(),
            *restore,
            execute,
        ),
        artifacts=(
            VerifierArtifact(source=TEST_OUTPUT, target="/source-test-output.txt", kind=ArtifactKind.FILE),
            VerifierArtifact(source=TEST_EXIT, target="/source-exit-code.txt", kind=ArtifactKind.FILE),
            VerifierArtifact(source=REWARD_PATH, target=COPIED_REWARD, kind=ArtifactKind.FILE),
        ),
        reward=FileReward(files=(RewardFile(path=REWARD_PATH, format=RewardFileFormat.NUMBER),)),
        timeout=timeout,
    )
