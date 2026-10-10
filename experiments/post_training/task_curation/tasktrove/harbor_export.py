# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower normalized TaskSpecs to Task Trove's Harbor parquet wire format."""

import gzip
import hashlib
import io
import json
import re
import shlex
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from finestore.schema import arrow_schema
from rigging.filesystem.storage_path import StoragePath
from verifyit.modes.extract import collapse_whitespace
from verifyit.spec import (
    DEFAULT_WORKSPACE,
    ExactSpec,
    GotestSpec,
    JudgeSpec,
    JunitSpec,
    MathSpec,
    NumericSpec,
    PytestSpec,
    ScriptSpec,
    Spec,
    mode_of,
    parse_spec,
    render_spec,
)

from taskcompendium.convert.script_grader import GRADE_ARGV
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import DOCKERFILE, TASKTROVE_REPO, TEST_SH
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import VERIFYIT_CONTEXT
from experiments.post_training.task_curation.tasktrove.source_images import source_actor_build
from taskcompendium.models import (
    DOCKER_IMAGE_PATTERN,
    AnswerType,
    ArtifactKind,
    FileReward,
    MissingArtifactPolicy,
    PlainText,
    ScriptGrader,
    StdoutReward,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_answer_file,
    verifyit_spec,
)
from taskcompendium.runtime.grading import DIAGNOSTIC_OUTPUT_BYTES
from taskcompendium.runtime.local import RUNTIME_PACKAGES, context_paths
from taskcompendium.runtime.resources import resource_bytes


class VerifierEnvironmentMode(StrEnum):
    SHARED = "shared"
    SEPARATE = "separate"


TASKS_FILENAME = "tasks.parquet"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class HarborSourceMetadata:
    name: str
    source_id: str
    family: str


@dataclass
class VerifierPayloadIdentity:
    """Identify emitted verifier files, recipes and dispatch configuration without building them."""

    _payloads: set[str] = field(default_factory=set, init=False)

    def add(self, task_binary: bytes) -> None:
        files = []
        with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as archive:
            for member in archive:
                if not member.isfile() or not (member.name.startswith("tests/") or member.name == "task.toml"):
                    continue
                content = archive.extractfile(member)
                assert content is not None
                files.append((member.name, member.mode, hashlib.sha256(content.read()).hexdigest()))
        self._payloads.add(hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest())

    @property
    def ref(self) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(sorted(self._payloads)).encode()).hexdigest()


class UnsupportedHarborTask(ValueError):
    """A task contract this exporter cannot preserve."""


@dataclass(frozen=True)
class HarborMetadata:
    path: str
    source: str
    family: str
    template_id: str
    converter: str
    mode: str
    dockerfile_id: str
    language: str
    tags: list[str]
    has_solution: bool


@dataclass(frozen=True)
class HarborRecord(HarborMetadata):
    task_binary: bytes
    solution_binary: bytes | None


@dataclass(frozen=True)
class HarborPayload:
    metadata: HarborMetadata
    files: dict[str, bytes]
    modes: dict[str, str]
    solution: dict[str, bytes]
    solution_modes: dict[str, str]


TASKS_SCHEMA = arrow_schema(HarborRecord)
IN_PROCESS_FILE_MODES = frozenset({"exact", "math", "json-schema", "mcq", "ifeval", "xml-elements", "csv-columns"})
VERIFIER_SPEC_PATH = "tests/taskcompendium-verifier.toml"
VERIFIER_RESOURCES_PATH = "tests/taskcompendium-resources.json"
HARBOR_REWARD_PATH = "/logs/verifier/reward.txt"
GRADER_STDOUT_PATH = "/logs/verifier/taskcompendium-stdout.txt"
REPOSITORY_WORKSPACE = "/testbed"
LEGACY_AGENT_TIMEOUT = 900.0
MIN_GOLD_LEAK_CHARS = 12
GOLD_LEAK_PREVIEW_CHARS = 60
HARBOR_SCRIPT_ARGV = ("bash", f"/{TEST_SH}")


def archive_file_mode(name: str, modes: dict[str, str]) -> int:
    return int(modes.get(name, "755" if name.endswith(".sh") else "644"), 8)


def archive_bytes(files: dict[str, bytes], modes: dict[str, str]) -> bytes:
    """Write deterministic regular-file archives without inheriting host ownership."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = archive_file_mode(name, modes)
            archive.addfile(entry, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), compresslevel=1, mtime=0)


@cache
def verifier_runtime() -> dict[str, bytes]:
    """Ship the same verifier implementation used by task-curation local runtimes."""
    return {
        f"tests/runtime/{package.name}/{path.relative_to(package).as_posix()}": path.read_bytes()
        for package in RUNTIME_PACKAGES
        for path in context_paths(package)
    }


@dataclass(frozen=True)
class _VerifierProgram:
    files: dict[str, bytes]
    answer_path: str | None
    mode: str
    timeout: float
    env: dict[str, str]
    cwd: str
    spec: Spec | None


def _environment_mode(task: TaskSpec) -> VerifierEnvironmentMode:
    if task.source.dataset == TASKTROVE_REPO and any(resource.path == DOCKERFILE for resource in task.resources.oracle):
        return VerifierEnvironmentMode.SHARED
    if task.answer_type == AnswerType.WORKSPACE_STATE:
        return VerifierEnvironmentMode.SHARED
    grader = task.grader
    if (
        task.answer_type == AnswerType.FILE
        and isinstance(grader, VerifyitGrader)
        and grader.mode == "stdio"
        and task.environment_requirements.docker_build is not None
        and grader.environment is not None
        and grader.environment.docker_build == task.environment_requirements.docker_build
    ):
        return VerifierEnvironmentMode.SHARED
    return VerifierEnvironmentMode.SEPARATE


def _validate_harbor_task(task: TaskSpec, environment_mode: VerifierEnvironmentMode) -> None:
    environment = task.environment_requirements
    grader = task.grader
    repository_state = task.answer_type == AnswerType.WORKSPACE_STATE
    if repository_state:
        if (
            "git_repository" not in environment.capabilities
            or not isinstance(grader, ScriptGrader)
            or grader.argv != HARBOR_SCRIPT_ARGV
            or not isinstance(grader.reward, FileReward)
            or len(grader.artifacts) != 1
        ):
            raise UnsupportedHarborTask("Repository state requires git_repository and one explicit workspace artifact")
        artifact = grader.artifacts[0]
        if (
            artifact.source != REPOSITORY_WORKSPACE
            or artifact.target != artifact.source
            or artifact.kind != ArtifactKind.DIRECTORY
            or artifact.exclude
            or artifact.missing != MissingArtifactPolicy.ERROR
            or grader.answer_path is not None
            or environment.docker_build is None
            or grader.environment is None
            or grader.environment.docker_build is None
        ):
            raise UnsupportedHarborTask(
                "Only required full /testbed repository capture with declared builds is supported"
            )
    elif environment_mode == VerifierEnvironmentMode.SEPARATE and (
        environment.docker_build is not None
        or (
            isinstance(grader, ScriptGrader | VerifyitGrader)
            and grader.environment is not None
            and grader.environment.docker_build is not None
        )
    ):
        raise UnsupportedHarborTask("Docker build contexts require a supported shared execution contract")
    if len(task.context.events) != 1 or not isinstance(task.context.events[0], TextMessage):
        raise UnsupportedHarborTask("Only a single public text instruction is supported")
    if task.answer_type not in (AnswerType.TEXT, AnswerType.FILE, AnswerType.WORKSPACE_STATE):
        raise UnsupportedHarborTask(f"Unsupported answer type: {task.answer_type}")
    if task.answer_type == AnswerType.TEXT and not isinstance(task.answer_format, PlainText):
        raise UnsupportedHarborTask("Text extraction beyond plain text requires dedicated Harbor lowering")
    if task.output_directories or task.final_tools:
        raise UnsupportedHarborTask("Directory capture and final tool calls require dedicated Harbor lowering")
    if environment.setup_commands or environment.packages_lock:
        raise UnsupportedHarborTask("Agent setup commands and package locks require an environment build")
    if set(environment.tool_providers) - {"shell"}:
        raise UnsupportedHarborTask("Only shell tool providers have Harbor lowering")


def _verifier_program(
    task: TaskSpec, grader_image: str | None, environment_mode: VerifierEnvironmentMode
) -> _VerifierProgram:
    grader = task.grader
    repository_state = task.answer_type == AnswerType.WORKSPACE_STATE
    files: dict[str, bytes] = {}
    if environment_mode == VerifierEnvironmentMode.SEPARATE:
        if grader_image is None:
            raise ValueError("This task requires an explicit digest-pinned verifier base image")
        # Separate Harbor verifiers own their tests; native execution skips uploading them.
        files["tests/Dockerfile"] = f"FROM {grader_image}\nCOPY . /tests\n".encode()
    answer_path = None
    spec: Spec | None = None
    if isinstance(grader, VerifyitGrader):
        spec = verifyit_spec(grader)
        answer_path = verifyit_answer_file(spec) if task.answer_type == AnswerType.TEXT else None
        # Harbor reserves tests/verifier.toml for its image-installed dispatch.
        files[VERIFIER_SPEC_PATH] = render_spec(spec).encode()
        if environment_mode == VerifierEnvironmentMode.SHARED:
            files[TEST_SH] = (f"#!/bin/bash\nset -euo pipefail\nexec verifyit /{VERIFIER_SPEC_PATH}\n").encode()
        else:
            files.update(verifier_runtime())
            files[TEST_SH] = (
                "#!/bin/bash\nset -euo pipefail\n"
                "export PYTHONPATH=/tests/runtime\n"
                "exec python3 -c 'from verifyit.grade import main; raise SystemExit(main())' "
                f"/{VERIFIER_SPEC_PATH}\n"
            ).encode()
        mode = grader.mode
        timeout = spec.timeout if isinstance(spec, (GotestSpec, JunitSpec, PytestSpec, ScriptSpec)) else 600.0
        grader_env = {}
        grader_cwd = "/"
    elif isinstance(grader, ScriptGrader):
        if grader.collect or (grader.artifacts and not repository_state):
            raise UnsupportedHarborTask("Script collection hooks require dedicated Harbor lowering")
        if grader.argv == GRADE_ARGV and isinstance(grader.reward, StdoutReward):
            files.update(verifier_runtime())
            files[TEST_SH] = (
                "#!/bin/bash\nset -euo pipefail\n"
                f"mkdir -p /logs/verifier\nrm -f {HARBOR_REWARD_PATH}\n"
                f"{shlex.join(grader.argv)} > {GRADER_STDOUT_PATH}\n"
                "export PYTHONPATH=/tests/runtime\npython3 - <<'PY'\n"
                "from pathlib import Path\n"
                "from verifyit.modes.extract import last_line\n"
                "from verifyit.modes.grade_script import parse_reward_number\n"
                f"with Path({GRADER_STDOUT_PATH!r}).open('rb') as stdout:\n"
                f"    output = stdout.read({DIAGNOSTIC_OUTPUT_BYTES}).decode(errors='replace')\n"
                "reward = parse_reward_number(last_line(output))\n"
                f"Path({HARBOR_REWARD_PATH!r}).write_text(str(reward))\nPY\n"
            ).encode()
        elif grader.argv == HARBOR_SCRIPT_ARGV and isinstance(grader.reward, FileReward):
            if len(grader.reward.files) != 1:
                raise UnsupportedHarborTask("Script graders must emit one Harbor reward file")
            reward = grader.reward.files[0]
            if reward.path != HARBOR_REWARD_PATH or reward.format != "number":
                raise UnsupportedHarborTask("Script graders must emit Harbor's numeric reward.txt")
        else:
            raise UnsupportedHarborTask("Unsupported script command or reward contract")
        answer_path = grader.answer_path
        mode, timeout, grader_env, grader_cwd = "script", grader.timeout, grader.env, grader.cwd
    else:
        raise UnsupportedHarborTask(f"Unsupported grader: {grader.kind}")
    if grader.environment is None and environment_mode == VerifierEnvironmentMode.SEPARATE:
        if not isinstance(grader, VerifyitGrader) or grader.mode not in IN_PROCESS_FILE_MODES:
            raise UnsupportedHarborTask("In-process grader mode has no supported file-delivery lowering")
        if task.answer_type != AnswerType.TEXT or answer_path is None:
            raise UnsupportedHarborTask("In-process graders require a plain-text answer file")
        files[VERIFIER_RESOURCES_PATH] = json.dumps([resource.path for resource in task.resources.verifier]).encode()
        files[TEST_SH] = (
            "#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH=/tests/runtime\n"
            f"exec python3 -m verifyit.candidate_file --spec /{VERIFIER_SPEC_PATH} "
            f"--answer {shlex.quote(answer_path)} --workspace {shlex.quote(DEFAULT_WORKSPACE)} "
            f"--logs-dir /logs/verifier --resources-manifest /{VERIFIER_RESOURCES_PATH}\n"
        ).encode()
    elif grader.environment is not None and grader.environment.setup_commands:
        raise UnsupportedHarborTask("Verifier setup commands require an environment build")
    return _VerifierProgram(
        files=files,
        answer_path=answer_path,
        mode=mode,
        timeout=timeout,
        env=grader_env,
        cwd=grader_cwd,
        spec=spec,
    )


def _validate_tasktrove_dockerfile(files: dict[str, bytes]) -> None:
    """Retain the old TaskTrove static environment exclusions without building an image."""
    dockerfile = files["environment/Dockerfile"].decode()
    if "# --- verifyit ---" not in dockerfile:
        raise UnsupportedHarborTask("tool install block missing")
    for line in dockerfile.splitlines():
        if re.search(r"rewardkit|litellm", line, re.IGNORECASE):
            raise UnsupportedHarborTask(f"old grader dependency: {line.strip()[:120]}")
        if line.lower().startswith("copy ") and " tests/" in f" {line}":
            source = line.split()[1]
            if source not in files and not any(path.startswith(source.rstrip("/") + "/") for path in files):
                raise UnsupportedHarborTask(f"COPY of a file not in the task: {source}")


def _reference_leak(instruction: str, spec: Spec | None) -> str | None:
    """Retain the legacy exclusion for a long reference disclosed in the public prompt."""
    if isinstance(spec, MathSpec | NumericSpec):
        expected = (str(spec.expected),)
    elif isinstance(spec, ExactSpec):
        expected = spec.expected
    elif isinstance(spec, JudgeSpec):
        expected = spec.references
    else:
        return None
    prompt = collapse_whitespace(instruction).lower()
    for reference in expected:
        needle = collapse_whitespace(reference).lower()
        if len(needle) >= MIN_GOLD_LEAK_CHARS and needle in prompt:
            return f"expected value appears in instruction: {needle[:GOLD_LEAK_PREVIEW_CHARS]!r}"
    return None


def harbor_payload(
    row: dict[str, Any],
    *,
    grader_image: str | None,
    family: str,
    fallback_actor_image: str,
    verifyit_package_root: Path | None = None,
) -> HarborPayload:
    """Lower tasks to Harbor, retaining TaskTrove's shared actor and grader environment."""
    if grader_image is not None and re.fullmatch(DOCKER_IMAGE_PATTERN, grader_image) is None:
        raise ValueError("The verifier image must be explicitly pinned by digest")
    task = TaskSpec.model_validate_json(row["task_json"])
    environment_mode = _environment_mode(task)
    _validate_harbor_task(task, environment_mode)
    environment = task.environment_requirements
    grader = task.grader
    repository_state = task.answer_type == AnswerType.WORKSPACE_STATE
    verifier = _verifier_program(task, grader_image, environment_mode)
    assert isinstance(grader, (ScriptGrader, VerifyitGrader))
    files = verifier.files
    modes: dict[str, str] = {}
    answer_path = verifier.answer_path
    prompt = cast(TextMessage, task.context.events[0]).content
    if task.answer_type == AnswerType.TEXT:
        if answer_path is None:
            raise UnsupportedHarborTask("Text grader has no answer-file destination")
        prompt += (
            f"\n\nWrite your final answer to `{answer_path}`. "
            "The contents of this file are graded as your final response."
        )
    if leak := _reference_leak(prompt, verifier.spec):
        raise UnsupportedHarborTask(f"gold_leak: {leak}")
    files["instruction.md"] = prompt.encode()
    public = () if environment.docker_build is not None else (*task.resources.all, *task.resources.worker)
    actor_build = environment.docker_build
    if actor_build is None:
        actor_build = source_actor_build(
            task, source=row["source_row"].split("/", 1)[0], mode=verifier.mode, package=verifyit_package_root
        )
    if actor_build is not None:
        for resource in actor_build.files:
            if (
                environment_mode == VerifierEnvironmentMode.SHARED
                and resource.path != "Dockerfile"
                and not resource.path.startswith(VERIFYIT_CONTEXT + "/")
            ):
                continue
            name = "environment/" + resource.path
            files[name] = resource_bytes(resource)
            if resource.mode:
                modes[name] = resource.mode
        dockerfile = files["environment/Dockerfile"].decode()
        if not dockerfile.endswith("\n"):
            dockerfile += "\n"
    else:
        dockerfile = f"FROM {environment.docker_image or fallback_actor_image}\n"
    if public:
        dockerfile += "COPY files/ /\n"
    for resource in public:
        if resource.path.startswith(("tests/", "solution/", "logs/verifier/")):
            raise UnsupportedHarborTask(f"Public resource overlaps a private Harbor root: {resource.path}")
        name = "environment/files/" + resource.path
        if name in files:
            raise UnsupportedHarborTask(f"Public resource collides with build input: {name}")
        files[name] = resource_bytes(resource)
        if resource.mode:
            modes[name] = resource.mode
    files["environment/Dockerfile"] = dockerfile.encode()
    for resource in task.resources.verifier:
        name = "tests/" + resource.path
        if name in files:
            raise UnsupportedHarborTask(f"Verifier resource collides with generated file: {name}")
        files[name] = resource_bytes(resource)
        if resource.mode:
            modes[name] = resource.mode
    if TEST_SH not in files:
        raise UnsupportedHarborTask("Script grader has no test.sh resource")
    language = next((tag.removeprefix("language:") for tag in task.tags if tag.startswith("language:")), "")
    mode = verifier.mode
    if repository_state:
        if VERIFIER_SPEC_PATH not in files or "swe-repo" not in task.tags:
            raise UnsupportedHarborTask("Shared repository lowering requires a legacy SWE verifier spec")
        spec = parse_spec(files[VERIFIER_SPEC_PATH].decode())
        if not (
            isinstance(spec, PytestSpec)
            or (isinstance(spec, ScriptSpec) and {"swe-repo", "patched", "script-fallback"}.issubset(task.tags))
        ):
            raise UnsupportedHarborTask(
                "Shared repository lowering requires the legacy SWE pytest or patched script contract"
            )
        mode = mode_of(spec)
        if "tests/verifier.toml" in files:
            raise UnsupportedHarborTask("Repository verifier spec uses Harbor's reserved entrypoint")
        # Legacy shared execution retains the actor's installed dependencies and workspace.
        files[TEST_SH] = f"#!/bin/bash\nset -euo pipefail\ncd {shlex.quote(verifier.cwd)}\n".encode() + files[TEST_SH]
    if actor_build is not None:
        _validate_tasktrove_dockerfile(files)
    outputs = list(task.output_paths)
    if answer_path:
        outputs.append(answer_path)
    # Harbor uploads submissions before running test.sh. Do not restore initial
    # copies of files the agent edits, including when the agent deleted a file.
    grader_public = (
        []
        if environment_mode == VerifierEnvironmentMode.SHARED
        else [resource for resource in public if "/" + resource.path not in outputs]
    )
    if grader_public:
        for resource in grader_public:
            files["tests/public/" + resource.path] = resource_bytes(resource)
            if resource.mode:
                modes["tests/public/" + resource.path] = resource.mode
        files[TEST_SH] = b"#!/bin/bash\nset -euo pipefail\ncp -a /tests/public/. /\n" + files[TEST_SH]
    metadata = {
        "taskcompendium_id": task.id,
        "source_dataset": task.source.dataset,
        "source_revision": task.source.revision,
        "source_row": task.source.row,
        "tasktrove_source": row["source_row"].split("/", 1)[0],
        "tasktrove_path": row["original_path"],
        "family": family,
        "conversion_only": True,
        "runtime_verified": False,
    }
    config: dict[str, Any] = {
        "schema_version": "1.2",
        "metadata": metadata,
        "agent": {"timeout_sec": LEGACY_AGENT_TIMEOUT},
        "environment": {"env": environment.environment_variables},
        "verifier": {
            "environment_mode": "separate",
            "timeout_sec": verifier.timeout,
            "env": verifier.env,
            "environment": {
                "workdir": verifier.cwd,
                "env": grader.environment.environment_variables if grader.environment is not None else {},
            },
        },
        "artifacts": [{"source": path, "destination": path.removeprefix("/")} for path in dict.fromkeys(outputs)],
    }
    if environment_mode == VerifierEnvironmentMode.SHARED:
        config["verifier"]["environment_mode"] = "shared"
        config["verifier"].pop("environment")
        config["artifacts"] = []
        config["verifier"]["env"] = {
            **(grader.environment.environment_variables if grader.environment is not None else {}),
            **verifier.env,
        }
        metadata["execution_policy"] = "tasktrove_shared"
    if environment.working_directory is not None:
        config["environment"]["workdir"] = environment.working_directory
    files["task.toml"] = tomlkit.dumps(config).encode()
    solution = {
        resource.path: resource_bytes(resource)
        for resource in task.resources.oracle
        if resource.path.startswith(("solution/", "tests/setup_files/"))
    }
    solution_modes = {resource.path: resource.mode for resource in task.resources.oracle if resource.mode}
    template = hashlib.sha256(files[TEST_SH]).hexdigest()[:12]
    record_metadata = HarborMetadata(
        path=row["original_path"],
        source=metadata["tasktrove_source"],
        family=family,
        template_id=template,
        converter="taskcompendium",
        mode=mode,
        dockerfile_id=hashlib.sha256(dockerfile.encode()).hexdigest()[:12],
        language=language,
        tags=[tag for tag in task.tags if not tag.startswith("language:")],
        has_solution=bool(solution),
    )
    return HarborPayload(record_metadata, files, modes, solution, solution_modes)


def harbor_record(
    row: dict[str, Any],
    *,
    grader_image: str | None,
    family: str,
    fallback_actor_image: str,
    verifyit_package_root: Path | None = None,
) -> HarborRecord:
    """Package the shared file assembly into the TaskTrove parquet record."""
    payload = harbor_payload(
        row,
        grader_image=grader_image,
        family=family,
        fallback_actor_image=fallback_actor_image,
        verifyit_package_root=verifyit_package_root,
    )
    return HarborRecord(
        **asdict(payload.metadata),
        task_binary=archive_bytes(payload.files, payload.modes),
        solution_binary=archive_bytes(payload.solution, payload.solution_modes) if payload.solution else None,
    )


def export_harbor(
    input_root: StoragePath,
    output_root: StoragePath,
    *,
    grader_image: str | None,
    source: HarborSourceMetadata,
    fallback_actor_image: str,
    verifyit_package_root: Path,
) -> dict[str, Any]:
    """Write the legacy parquet view and account for normalization and lowering failures."""
    manifest = json.loads((input_root / MANIFEST_FILENAME).read_text())
    if manifest["source"] != source.name:
        raise ValueError(f"Normalized source {manifest['source']!r} does not match bound export source {source.name!r}")
    for filename in (TASKS_FILENAME, MANIFEST_FILENAME):
        if (output_root / filename).exists():
            raise FileExistsError(f"Harbor export already exists: {output_root / filename}")
    # The artifact runner creates this directory for its status and provenance files.
    output_root.mkdirs()
    rejected = []
    input_count, exported_count = 0, 0
    counts: Counter[str] = Counter()
    verifier_identity = VerifierPayloadIdentity()
    paths = sorted((input_root / "normalize/*.parquet").glob(), key=str)
    if not paths:
        raise ValueError(f"No normalized parquet shards under {input_root}")
    with (output_root / TASKS_FILENAME).open("wb") as output_file, pq.ParquetWriter(output_file, TASKS_SCHEMA) as writer:
        for path in paths:
            with path.open("rb") as input_file:
                for batch in pq.ParquetFile(input_file).iter_batches(batch_size=64):
                    converted = []
                    for row in batch.to_pylist():
                        input_count += 1
                        counts.setdefault(row["source_row"].split("/", 1)[0], 0)
                        reason = row["normalization_reason"] if row["task_json"] is None else None
                        if row["task_json"] is not None:
                            try:
                                record = harbor_record(
                                    row,
                                    grader_image=grader_image,
                                    family=source.family,
                                    fallback_actor_image=fallback_actor_image,
                                    verifyit_package_root=verifyit_package_root,
                                )
                            except UnsupportedHarborTask as error:
                                reason = str(error)
                            else:
                                converted.append(asdict(record))
                                verifier_identity.add(record.task_binary)
                                counts[record.source] += 1
                                exported_count += 1
                        if reason is not None:
                            rejected.append({"task_id": row["task_id"], "path": row["original_path"], "reason": reason})
                    writer.write_table(pa.Table.from_pylist(converted, schema=TASKS_SCHEMA))
    manifest = {
        "input_rows": input_count,
        "exported_rows": exported_count,
        "rejected_rows": len(rejected),
        "by_source": dict(counts),
        "rejections": rejected,
        "grader_base_image": grader_image,
        "verifyit_package_root": str(verifyit_package_root),
        "verify_tool_ref": verifier_identity.ref,
        "environment_build_required": True,
        "source": source.name,
        "source_id": source.source_id,
        "runtime_verified": False,
        "limitation": (
            "Harbor schema, builds, and execution are unverified; "
            "base-image dependencies and source recipes have not been checked."
        ),
    }
    (output_root / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
