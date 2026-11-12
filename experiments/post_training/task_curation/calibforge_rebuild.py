# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild CalibForge V1 as a snapshot-safe cohort.

The native cohort bakes per-task data and dependencies into 2,181 distinct
build contexts. The rebuild lowers every decontaminated task onto a small
fixed set of base images, moving the task's files and dependency installs
into a setup script that runs at sandbox start — so a whole cohort needs a
handful of snapshots instead of one per task. Anything the fixed bases
cannot serve is deferred with an explicit reason, never silently altered.
"""

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import shlex
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from harbor_config.models.task.config import EnvironmentConfig
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    InlineFile,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from tomlkit.items import Table

from experiments.post_training.task_curation.calibforge_fixtures import (
    FIXTURE_ARCHIVE,
    SETUP_ROOT,
    SETUP_SCRIPT,
    SETUP_SCRIPT_PREAMBLE,
    SHELL_METACHARACTERS,
    copy_commands,
    copy_target,
    fixture_generator,
    teardown_commands,
)
from experiments.post_training.task_curation.environment import dockerfile_lines
from experiments.post_training.task_curation.environment_fidelity import (
    DEFAULT_WORKDIR,
    ENVIRONMENT_DOCKERFILE,
    ENVIRONMENT_PREFIX,
    PINNED_BASES,
    dynamic_copy_error,
)
from experiments.post_training.task_curation.lowered_envelope import (
    LOWERED_TASK_SPEC_COLUMN,
    LoweredSpec,
    MachineShape,
)
from experiments.post_training.task_curation.task_tars import open_task_tar

REBUILT = "rebuilt_pending_oracle_and_admission"
DEFERRED = "deferred"
EXCLUDED_DECONTAMINATION = "excluded_decontamination"
EXCLUDED_SETUP_FAILURE = "excluded_setup_failure"


NORMALIZED_CPUS = 2
NORMALIZED_MEMORY_MB = 4096
NORMALIZED_STORAGE_MB = 10240
"""One machine shape for the rebuilt cohort.

Snapshot identity includes the resource shape, so leaving each task's declared
shape in place multiplies the cohort's snapshots by every distinct size. Tasks
that need more than this shape are deferred rather than silently shrunk."""


SETUP_FAILURES = {
    # Exercised through the lowered spec's own setup command in its base image: the
    # package install segfaults reproducibly at one worker, so the task never reaches
    # a workspace a verifier could grade.
    "contrastive_solver/software-engineering_20260507_093408_007": (
        "setup_failure: segfault during pip install, reproducible at 1 worker"
    ),
}
"""Tasks excluded from the cohort because their own setup does not complete."""

_META_COMMANDS = {"from", "cmd", "entrypoint", "label", "expose", "stopsignal", "healthcheck", "shell"}
_INNOCUOUS_RUN = {"mkdir", "chmod", "touch", "ln", "rm"}


class DeferredRebuild(ValueError):
    """A native recipe needs a contract the fixed bases do not serve."""


@dataclass(frozen=True)
class Instruction:
    """One parsed Dockerfile instruction."""

    command: str
    args: list[str]
    flags: list[str]
    json_form: bool
    raw: str = ""
    """The unsplit remainder after the command word, as written in the Dockerfile."""


@dataclass(frozen=True)
class RebuiltTask:
    """A lowered task plus the evidence admission needs."""

    task_binary: bytes
    base_ref: str
    workdir: str
    payload_sha256: str
    setup_sha256: str


def parse_instructions(dockerfile: str) -> list[Instruction]:
    """Parse Dockerfile instructions for the setup compiler.

    Handles line continuations, comments, JSON-form RUN arrays and heredoc
    detection. Heredocs defer; compilation decides which instructions are supported.
    """
    instructions: list[Instruction] = []
    try:
        lines = dockerfile_lines(dockerfile)
    except ValueError as error:
        raise DeferredRebuild("requires_continuation_terminator") from error
    for line in lines:
        if "<<" in line:
            raise DeferredRebuild("requires_heredoc_support")
        words = line.split(None, 1)
        if not words:
            continue
        command = words[0].lower()
        raw_args = words[1] if len(words) > 1 else ""
        json_form = False
        # Dockerfile instruction flags lead the line; anything after content begins
        # (e.g. apt-get's --no-install-recommends) is a shell argument, not a flag.
        flags = []
        while raw_args.startswith("--"):
            flag = raw_args.split(None, 1)[0]
            flags.append(flag)
            raw_args = raw_args[len(flag) :].lstrip()
        if command == "run" and raw_args.lstrip().startswith("["):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as error:
                raise DeferredRebuild("requires_run_json_support") from error
            json_form = True
            args = [str(item) for item in parsed]
        else:
            try:
                parts = shlex.split(raw_args, comments=True, posix=True)
            except ValueError as error:
                raise DeferredRebuild("requires_balanced_quotation") from error
            if not parts:
                continue
            args = parts
        instructions.append(Instruction(command=command, args=args, flags=flags, json_form=json_form, raw=raw_args))
    return instructions


def _dependency_run(arguments: list[str]) -> bool:
    """True for package installs and index refreshes the setup script may run at start."""
    if arguments[0] in {"apt-get", "apt"}:
        return "install" in arguments or "update" in arguments
    if arguments[0] in {"pip", "pip3"}:
        return len(arguments) > 1 and arguments[1] == "install"
    if arguments[0] == "python3" and len(arguments) > 3:
        return arguments[1:3] == ["-m", "pip"] and "install" in arguments[3:]
    return False


def _has_shell_metacharacters(arguments: list[str]) -> bool:
    return any(character in argument for argument in arguments for character in SHELL_METACHARACTERS)


@dataclass(frozen=True)
class CompiledRecipe:
    base_ref: str
    workdir: str
    setup_script: str
    payload: bytes


def _pack_tar(files: dict[str, bytes], modes: dict[str, int]) -> bytes:
    """Pack files deterministically: sorted members, fixed modes, gzip with mtime=0.

    Unlike ``tasktrove.harbor_export.archive_bytes`` this takes integer modes
    straight from the native tar and uses the default gzip level: the rebuild's
    payload digests must stay stable against previously built cohorts, not match
    the verifier-export archives.
    """
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = modes.get(name, 0o644)
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(payload.getvalue(), mtime=0)


def compile_recipe(instructions: list[Instruction], files: dict[str, bytes], modes: dict[str, int]) -> CompiledRecipe:
    """Compile a native recipe into a base, workdir, setup script and fixture archive.

    The compiled script must run the task's full native recipe: dependency
    installs move into the setup script verbatim, file copies are staged from
    the fixture payload, and ENV assignments persist in the setup script. A
    step that cannot run that way without changing the task defers with the
    reason admission records.
    """
    bases = [item for item in instructions if item.command == "from"]
    if len(bases) != 1:
        raise DeferredRebuild("requires_multistage_support")
    base_ref = _resolved_base(bases[0])
    workdir, steps = _recipe_body(instructions, files)
    steps.commands.extend(teardown_commands())
    if steps.env_exports:
        steps.commands[2:2] = steps.env_exports
    return CompiledRecipe(base_ref, workdir, "\n".join(steps.commands) + "\n", _pack_tar(steps.resources, modes))


def _resolved_base(from_instruction: Instruction) -> str:
    """The pinned base reference a native recipe's single FROM line maps onto."""
    if len(from_instruction.args) != 1:
        raise DeferredRebuild("requires_multistage_support")
    original_from = from_instruction.args[0]
    base_key, _, original_digest = original_from.partition("@")
    if base_key not in PINNED_BASES:
        raise DeferredRebuild("requires_unmapped_base:" + original_from)
    # The mapped base already carries a digest; a digest-pinned original keeps its own.
    return f"{base_key}@{original_digest}" if original_digest else PINNED_BASES[base_key]


@dataclass
class _RecipeSteps:
    """Setup commands, exports and staged files accumulated while lowering a recipe."""

    commands: list[str] = field(default_factory=list)
    env_exports: list[str] = field(default_factory=list)
    resources: dict[str, bytes] = field(default_factory=dict)
    copied_files: set[str] = field(default_factory=set)


def _recipe_body(instructions: list[Instruction], files: dict[str, bytes]) -> tuple[str, _RecipeSteps]:
    """Lower the recipe's steps, returning the final workdir and accumulated steps."""
    steps = _RecipeSteps(commands=list(SETUP_SCRIPT_PREAMBLE))
    workdir = "/"
    for item in instructions:
        if item.command in _META_COMMANDS:
            continue
        if item.command == "env":
            _extend_env_exports(item, steps)
        elif item.command == "arg":
            raise DeferredRebuild("requires_arg_support")
        elif item.command == "workdir":
            workdir = _enter_workdir(item, workdir, steps)
        elif item.command == "run":
            steps.commands.extend(_run_commands(item, workdir, steps.copied_files))
        elif item.command == "copy":
            _stage_copy(item, workdir, files, steps)
        else:
            raise DeferredRebuild("requires_instruction_support:" + item.command)
    return workdir, steps


def _extend_env_exports(item: Instruction, steps: _RecipeSteps) -> None:
    if item.json_form or item.flags:
        raise DeferredRebuild("requires_env_flags_support")
    if not item.args or any("=" not in argument for argument in item.args) or _has_shell_metacharacters(item.args):
        raise DeferredRebuild("requires_env_support")
    steps.env_exports.extend("export " + shlex.quote(argument) for argument in item.args)


def _enter_workdir(item: Instruction, workdir: str, steps: _RecipeSteps) -> str:
    value = item.args[0]
    if "$" in value or ".." in PurePosixPath(value).parts:
        raise DeferredRebuild("requires_dynamic_workdir_support")
    workdir = str(PurePosixPath(workdir) / value)
    steps.commands.append("mkdir -p " + shlex.quote(workdir))
    return workdir


def _run_commands(item: Instruction, workdir: str, copied_files: set[str]) -> list[str]:
    """The setup commands one RUN's segments become, or a deferral when one cannot be expressed."""
    if item.flags:
        raise DeferredRebuild("requires_run_flags_support")
    if item.json_form:
        segments = [item.args]
    else:
        segments: list[list[str]] = [[]]
        for token in item.raw.split():
            if token in {"&&", ";"}:
                segments.append([])
            elif token in {"|", "<", ">", "&"}:
                raise DeferredRebuild("requires_pipeline_support")
            else:
                segments[-1].append(token)
    lowered: list[str] = []
    for raw_segment in segments:
        if not raw_segment:
            raise DeferredRebuild("requires_empty_run")
        inline_env = ""
        while raw_segment and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", raw_segment[0]):
            inline_env += raw_segment[0] + " "
            raw_segment = raw_segment[1:]
        arguments = raw_segment
        if _dependency_run(arguments) or arguments[0] in _INNOCUOUS_RUN or fixture_generator(arguments, copied_files):
            if "`" in " ".join(raw_segment):
                raise DeferredRebuild("requires_command_substitution_support")
            prefix = inline_env
            if arguments[0] in {"apt-get", "apt"} and "DEBIAN_FRONTEND" not in inline_env:
                prefix = "DEBIAN_FRONTEND=noninteractive " + prefix
            # Emit the segment verbatim: globs and variables keep their shell
            # semantics at sandbox start (the recipe is the task's own content).
            lowered.append(f"(cd {shlex.quote(workdir)} && {prefix}{' '.join(raw_segment)})")
            continue
        raise DeferredRebuild("requires_unsupported_run:" + arguments[0])
    return lowered


def _stage_copy(item: Instruction, workdir: str, files: dict[str, bytes], steps: _RecipeSteps) -> None:
    """Stage one COPY's sources into the fixture payload and emit its copy commands.

    The rebuild must faithfully reproduce every COPY the recipe makes: sources
    come from the tar's bytes, the destination resolves against the recipe's
    final workdir, and every staged file lands in the fixture archive the
    setup script unpacks. A COPY that cannot be expressed defers.
    """
    if item.flags:
        raise DeferredRebuild("requires_copy_flags_support")
    *sources, destination = item.args
    for source_name in sources:
        dynamic = dynamic_copy_error(destination, source_name)
        if dynamic is not None:
            raise DeferredRebuild(dynamic)
    target, directory_target = copy_target(workdir, destination, len(sources))
    for source_name in sources:
        prefix = source_name.rstrip("/") + "/"
        matches = [name for name in files if name == source_name or name.startswith(prefix)]
        if not matches:
            raise DeferredRebuild("missing_source:" + source_name)
        is_directory = any(name != source_name and name.startswith(prefix) for name in matches)
        if is_directory:
            directory_source = source_name.rstrip("/")
            for name in matches:
                # Every file under a directory source must reach the setup payload;
                # the script copies the staged directory, not the build context.
                steps.resources[name] = files[name]
                steps.copied_files.add(str(PurePosixPath(target) / name[len(prefix) :]))
            steps.commands.extend(
                copy_commands('"$stage"/' + shlex.quote(directory_source), target, into_directory=True)
            )
            continue
        name = matches[0]
        steps.resources[name] = files[name]
        file_target = str(PurePosixPath(target) / PurePosixPath(name).name) if directory_target else target
        steps.copied_files.add(file_target)
        steps.commands.extend(copy_commands('"$stage"/' + shlex.quote(name), file_target, into_directory=False))


def rebuild_task(task_binary: bytes) -> RebuiltTask:
    """Lower one native task tar onto the fixed base set."""
    with open_task_tar(task_binary) as archive:
        files: dict[str, bytes] = {}
        modes: dict[str, int] = {}
        environment_modes: dict[str, int] = {}
        for member in archive.getmembers():
            if member.isdir() or member.issym():
                if member.issym():
                    raise DeferredRebuild("requires_symlink_support")
                continue
            files[member.name] = archive.extractfile(member).read()
            modes[member.name] = member.mode
            if member.name.startswith(ENVIRONMENT_PREFIX):
                environment_modes[member.name[len(ENVIRONMENT_PREFIX) :]] = member.mode
    dockerfile = files.pop(ENVIRONMENT_DOCKERFILE, None)
    if dockerfile is None:
        raise DeferredRebuild("missing_environment_dockerfile")
    instructions = parse_instructions(dockerfile.decode(errors="replace"))
    context_files = {
        name[len(ENVIRONMENT_PREFIX) :]: content
        for name, content in files.items()
        if name.startswith(ENVIRONMENT_PREFIX)
    }
    recipe = compile_recipe(instructions, context_files, environment_modes)
    base_ref, workdir = recipe.base_ref, recipe.workdir
    setup_script, payload = recipe.setup_script, recipe.payload
    output_files = {name: content for name, content in files.items() if not name.startswith(ENVIRONMENT_PREFIX)}
    output_files["task.toml"] = _rewrite_task_toml(files.get("task.toml"), base_ref, workdir)
    output_files[ENVIRONMENT_DOCKERFILE] = f"FROM {base_ref}\nWORKDIR {workdir}\n".encode()
    output_files[f"setup_files/{FIXTURE_ARCHIVE}"] = payload
    output_files[f"setup_files/{SETUP_SCRIPT}"] = setup_script.encode()
    return RebuiltTask(
        task_binary=_pack_tar(output_files, modes),
        base_ref=base_ref,
        workdir=workdir,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        setup_sha256=hashlib.sha256(setup_script.encode()).hexdigest(),
    )


def _rewrite_task_toml(original: bytes | None, base_ref: str, workdir: str) -> bytes:
    """Rewrite task.toml onto the fixed base with a setup-at-start hook.

    The machine shape is normalized so the cohort's snapshots collapse to one
    per base image; a task that declares more than the normalized shape defers.
    """
    if original is None:
        raise DeferredRebuild("missing_task_toml")
    config = tomllib.loads(original.decode(errors="replace"))
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise DeferredRebuild("missing_environment_section")
    if "setup" in environment:
        raise DeferredRebuild("existing_setup_hook")
    declared = EnvironmentConfig(**environment)
    if declared.gpus or declared.tpu:
        raise DeferredRebuild("requires_accelerator_machine")
    if (declared.cpus or 0) > NORMALIZED_CPUS:
        raise DeferredRebuild("requires_larger_machine_shape")
    if (declared.memory_mb or 0) > NORMALIZED_MEMORY_MB:
        raise DeferredRebuild("requires_larger_machine_shape")
    if (declared.storage_mb or 0) > NORMALIZED_STORAGE_MB:
        raise DeferredRebuild("requires_larger_machine_shape")
    document = tomlkit.parse(original.decode(errors="replace"))
    table = document["environment"]
    assert isinstance(table, Table)
    table["docker_image"] = base_ref
    table["workdir"] = workdir
    table["cpus"] = NORMALIZED_CPUS
    table["memory"] = f"{NORMALIZED_MEMORY_MB // 1024}G"
    table["storage"] = f"{NORMALIZED_STORAGE_MB // 1024}G"
    table["setup"] = {"command": f"sh {SETUP_ROOT}/{SETUP_SCRIPT}"}
    return tomlkit.dumps(document).encode()


LOWERED_RUNTIME = MachineShape(cpus=NORMALIZED_CPUS, memory_mb=NORMALIZED_MEMORY_MB, storage_mb=NORMALIZED_STORAGE_MB)
"""The machine shape every lowered task declares."""


def setup_script_name(command: str) -> str:
    """The script filename a rebuilt task's setup command ends with."""
    return command.split()[-1].rsplit("/", 1)[-1]


def lowered_task_spec(task_id: str, task_binary: bytes) -> LoweredSpec:
    """Lower a rebuilt task into the rollout envelope preserving its workspace, setup and verifier."""
    with open_task_tar(task_binary) as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    document = tomllib.loads(files["task.toml"].decode())
    environment = document["environment"]
    base_ref = environment["docker_image"]
    workdir = environment.get("workdir") or DEFAULT_WORKDIR
    if workdir == "/":
        raise DeferredRebuild("requires_explicit_output_workspace")
    setup_script = setup_script_name((environment.get("setup") or {}).get("command", ""))

    def resource(path: str, content: bytes, mode: str | None = None) -> TaskResource:
        payload = base64.b64encode(content).decode()
        return TaskResource(path=path, source=InlineFile(content_base64=payload), mode=mode)

    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=base_ref,
        working_directory=workdir,
        setup_commands=(f"sh {SETUP_ROOT}/{setup_script}",),
    )
    verifier = tuple(resource(path, files[path]) for path in sorted(files) if path.startswith("tests/"))
    setup = tuple(
        resource(path, files[path], "0755" if files[path].startswith(b"#!") else None)
        for path in sorted(files)
        if path.startswith("setup_files/")
    )
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=files["instruction.md"].decode()),)),
        environment_requirements=requirements,
        output_paths=(workdir,),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("bash", "/tests/test.sh"),
            cwd=workdir,
            environment=requirements,
            reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
        ),
        source=Source(dataset="smartie-calibforge-rebuilt", revision="1", row=task_id, importer_revision="1"),
        resources=ResourceGroups(verifier=verifier, all=setup),
    )
    return LoweredSpec(spec, LOWERED_RUNTIME, LOWERED_RUNTIME)


@dataclass(frozen=True)
class TaskEvidence:
    """One cohort task's rebuild outcome, as the evidence manifest records it."""

    task_id: str
    status: str
    source_archive_sha256: str | None = None
    reason: str | None = None
    base_ref: str | None = None
    workdir: str | None = None
    payload_sha256: str | None = None
    setup_sha256: str | None = None
    archive_sha256: str | None = None

    def to_dict(self) -> dict:
        """The manifest row: only the fields this task's status carries."""
        if self.status in (EXCLUDED_DECONTAMINATION,):
            return {"task_id": self.task_id, "status": self.status}
        if self.status in (EXCLUDED_SETUP_FAILURE, DEFERRED):
            return {"task_id": self.task_id, "status": self.status, "reason": self.reason}
        return {
            "task_id": self.task_id,
            "status": self.status,
            "source_archive_sha256": self.source_archive_sha256,
            "base_ref": self.base_ref,
            "workdir": self.workdir,
            "payload_sha256": self.payload_sha256,
            "setup_sha256": self.setup_sha256,
            "archive_sha256": self.archive_sha256,
        }


def iter_task_payloads(path: Path):
    """Each (task_id, payload) pair in a cohort parquet's ``task_binary`` column."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            yield task_id, bytes(blob)


@dataclass(frozen=True)
class _RebuildOutcome:
    """One rebuild run's staged tasks, lowered specs, evidence rows and base refs."""

    records: list[dict]
    lowered: list[dict]
    evidence: list[TaskEvidence]
    base_refs: list[str]


def _rebuild_cohort(cohort: Path, excluded: set[str]) -> _RebuildOutcome:
    """Rebuild each unexcluded task, staging its payload and lowered spec."""
    records: list[dict] = []
    lowered: list[dict] = []
    evidence: list[TaskEvidence] = []
    base_refs: list[str] = []
    for path, raw in iter_task_payloads(cohort):
        if path in excluded:
            evidence.append(TaskEvidence(task_id=path, status=EXCLUDED_DECONTAMINATION))
            continue
        if path in SETUP_FAILURES:
            evidence.append(TaskEvidence(task_id=path, status=EXCLUDED_SETUP_FAILURE, reason=SETUP_FAILURES[path]))
            continue
        source_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            rebuilt = rebuild_task(raw)
        except DeferredRebuild as error:
            evidence.append(TaskEvidence(task_id=path, status=DEFERRED, reason=str(error)))
            continue
        base_refs.append(rebuilt.base_ref)
        evidence.append(
            TaskEvidence(
                task_id=path,
                status=REBUILT,
                source_archive_sha256=source_sha256,
                base_ref=rebuilt.base_ref,
                workdir=rebuilt.workdir,
                payload_sha256=rebuilt.payload_sha256,
                setup_sha256=rebuilt.setup_sha256,
                archive_sha256=hashlib.sha256(rebuilt.task_binary).hexdigest(),
            )
        )
        records.append({"path": path, "source": path, "task_binary": rebuilt.task_binary})
        lowered.append(
            {
                "path": path,
                LOWERED_TASK_SPEC_COLUMN: json.dumps(lowered_task_spec(path, rebuilt.task_binary).envelope()),
            }
        )
    return _RebuildOutcome(records, lowered, evidence, base_refs)


def _evidence_manifest(
    cohort: Path, excluded: Path, records: list[dict], evidence: list[TaskEvidence], base_refs: list[str]
) -> dict:
    """The admission-facing summary of one rebuild run."""
    return {
        "cohort_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "excluded_sha256": hashlib.sha256(excluded.read_bytes()).hexdigest(),
        "cohort_count": len(evidence),
        "excluded_count": sum(1 for row in evidence if row.status == EXCLUDED_DECONTAMINATION),
        "rebuilt_count": len(records),
        "deferred_count": sum(1 for row in evidence if row.status == DEFERRED),
        # The Atlas counter measures the snapshot identities from these refs; the
        # rebuild itself stays free of the Harbor measurement environment.
        "snapshot_identity_count": None,
        "base_refs": sorted(set(base_refs)),
        "digest_pinned": True,
        # The cohort ships no oracle solutions; these record the outcome of
        # no_oracle_qualification over the lowered specs. Rebuild alone knows neither.
        "admission": "not_run",
        "oracle_validation": "skipped",
        "tasks": [row.to_dict() for row in evidence],
    }


def main() -> None:
    """Rebuild a native cohort parquet: staged tasks + evidence manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="native cohort parquet (path, task_binary)")
    parser.add_argument("--excluded", type=Path, required=True, help="decontamination exclusion list (json array)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    excluded = set(json.loads(args.excluded.read_text()))
    outcome = _rebuild_cohort(args.cohort, excluded)
    schema = pa.schema([("path", pa.string()), ("source", pa.string()), ("task_binary", pa.binary())])
    pq.write_table(pa.Table.from_pylist(outcome.records, schema=schema), args.output / "pending-tasks.parquet")
    lowered_schema = pa.schema([("path", pa.string()), (LOWERED_TASK_SPEC_COLUMN, pa.string())])
    pq.write_table(pa.Table.from_pylist(outcome.lowered, schema=lowered_schema), args.output / "lowered-tasks.parquet")
    manifest = _evidence_manifest(args.cohort, args.excluded, outcome.records, outcome.evidence, outcome.base_refs)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
