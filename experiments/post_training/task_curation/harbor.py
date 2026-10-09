# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower normalized TaskSpecs to Task Trove's Harbor parquet wire format."""

import gzip
import hashlib
import io
import json
import shlex
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import click
import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from finestore.schema import arrow_schema
from harbor_config.models.task.config import TaskConfig
from taskcompendium.models import (
    AnswerType,
    FileReward,
    PlainText,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_answer_file,
    verifyit_spec,
)
from taskcompendium.runtime.local import RUNTIME_PACKAGES, context_paths
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import render_spec

from experiments.post_training.task_curation.environment import PINNED_IMAGE
from experiments.post_training.task_curation.images.build import BASE_IMAGE
from experiments.post_training.task_curation.sources import all_sources


class UnsupportedHarborTask(ValueError):
    """A task contract this exporter cannot preserve."""


@dataclass(frozen=True)
class HarborRecord:
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
    task_binary: bytes
    solution_binary: bytes | None


TASKS_SCHEMA = arrow_schema(HarborRecord)
IN_PROCESS_FILE_MODES = frozenset({"exact", "math", "json-schema", "mcq", "ifeval", "xml-elements", "csv-columns"})


def archive_bytes(files: dict[str, bytes], modes: dict[str, str]) -> bytes:
    """Write deterministic regular-file archives without inheriting host ownership."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = int(modes.get(name, "755" if name.endswith(".sh") else "644"), 8)
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


def harbor_record(row: dict[str, Any], *, grader_image: str, family: str) -> HarborRecord:
    """Export supported file-delivery contracts with separate agent and verifier environments."""
    if PINNED_IMAGE.fullmatch(grader_image) is None:
        raise ValueError("The verifier image must be explicitly pinned by digest")
    task = TaskSpec.model_validate_json(row["task_json"])
    if len(task.context.events) != 1 or not isinstance(task.context.events[0], TextMessage):
        raise UnsupportedHarborTask("Only a single public text instruction is supported")
    if task.answer_type not in (AnswerType.TEXT, AnswerType.FILE):
        raise UnsupportedHarborTask(f"Unsupported answer type: {task.answer_type}")
    if task.answer_type == AnswerType.TEXT and not isinstance(task.answer_format, PlainText):
        raise UnsupportedHarborTask("Text extraction beyond plain text requires dedicated Harbor lowering")
    if task.output_directories or task.final_tools:
        raise UnsupportedHarborTask("Directory capture and final tool calls require dedicated Harbor lowering")
    environment = task.environment_requirements
    if environment.setup_commands or environment.packages_lock:
        raise UnsupportedHarborTask("Agent setup commands and package locks require an environment build")
    if set(environment.tool_providers) - {"shell"}:
        raise UnsupportedHarborTask("Only shell tool providers have Harbor lowering")
    grader = task.grader
    files, modes = {}, {}
    answer_path = None
    if isinstance(grader, VerifyitGrader):
        spec = verifyit_spec(grader)
        answer_path = verifyit_answer_file(spec) if task.answer_type == AnswerType.TEXT else None
        files.update(verifier_runtime())
        # Harbor reserves tests/verifier.toml for its image-installed verifyit command.
        # Keep the bundled runtime wrapper in control of grading and public-file staging.
        files["tests/taskcompendium-verifier.toml"] = render_spec(spec).encode()
        files["tests/test.sh"] = (
            b"#!/bin/bash\nset -euo pipefail\n"
            b"export PYTHONPATH=/tests/runtime\n"
            b"exec python3 -c 'from verifyit.grade import main; raise SystemExit(main())' "
            b"/tests/taskcompendium-verifier.toml\n"
        )
        mode = grader.mode
        timeout = float(grader.parameters.get("timeout", 600))
        grader_env = {}
        grader_cwd = "/"
    elif isinstance(grader, ScriptGrader):
        if grader.argv != ("bash", "/tests/test.sh") or grader.collect or grader.artifacts:
            raise UnsupportedHarborTask("Only archived test.sh script graders without collection hooks are supported")
        if not isinstance(grader.reward, FileReward) or len(grader.reward.files) != 1:
            raise UnsupportedHarborTask("Script graders must emit one Harbor reward file")
        reward = grader.reward.files[0]
        if reward.path != "/logs/verifier/reward.txt" or reward.format != "number":
            raise UnsupportedHarborTask("Script graders must emit Harbor's numeric reward.txt")
        answer_path = grader.answer_path
        mode, timeout, grader_env, grader_cwd = "script", grader.timeout, grader.env, grader.cwd
    else:
        raise UnsupportedHarborTask(f"Unsupported grader: {grader.kind}")
    if grader.environment is None:
        if not isinstance(grader, VerifyitGrader) or grader.mode not in IN_PROCESS_FILE_MODES:
            raise UnsupportedHarborTask("In-process grader mode has no supported file-delivery lowering")
        if task.answer_type != AnswerType.TEXT or answer_path is None:
            raise UnsupportedHarborTask("In-process graders require a plain-text answer file")
        files["tests/grade_candidate.py"] = Path(__file__).with_name("harbor_candidate.py").read_bytes()
        files["tests/taskcompendium-resources.json"] = json.dumps(
            [resource.path for resource in task.resources.verifier]
        ).encode()
        files["tests/test.sh"] = (
            "#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH=/tests/runtime\n"
            "exec python3 /tests/grade_candidate.py /tests/taskcompendium-verifier.toml "
            f"{shlex.quote(answer_path)}\n"
        ).encode()
    elif grader.environment.setup_commands:
        raise UnsupportedHarborTask("Verifier setup commands require an environment build")
    prompt = task.context.events[0].content
    if task.answer_type == AnswerType.TEXT:
        if answer_path is None:
            raise UnsupportedHarborTask("Text grader has no answer-file destination")
        if grader.environment is None:
            prompt += (
                f"\n\nWrite your final answer to `{answer_path}`. "
                "The contents of this file are graded as your final response."
            )
        else:
            rewrites = [change for change in row["normalization_changes"] if change["field"] == "instruction"]
            if (
                len(rewrites) != 1
                or rewrites[0]["replacement"].strip() != prompt.strip()
                or answer_path not in rewrites[0]["original"]
            ):
                raise UnsupportedHarborTask(
                    "Text delivery requires a recorded original instruction naming the answer file"
                )
            prompt = rewrites[0]["original"]
    files["instruction.md"] = prompt.encode()
    public = (*task.resources.all, *task.resources.worker)
    dockerfile = f"FROM {environment.docker_image or BASE_IMAGE}\n"
    if public:
        dockerfile += "COPY files/ /\n"
    for resource in public:
        if resource.path.startswith(("tests/", "solution/", "logs/verifier/")):
            raise UnsupportedHarborTask(f"Public resource overlaps a private Harbor root: {resource.path}")
        name = "environment/files/" + resource.path
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
    if "tests/test.sh" not in files:
        raise UnsupportedHarborTask("Script grader has no test.sh resource")
    outputs = list(task.output_paths)
    if answer_path:
        outputs.append(answer_path)
    # Harbor uploads submissions before running test.sh. Do not restore initial
    # copies of files the agent edits, including when the agent deleted a file.
    grader_public = [resource for resource in public if "/" + resource.path not in outputs]
    if grader_public:
        for resource in grader_public:
            files["tests/public/" + resource.path] = resource_bytes(resource)
            if resource.mode:
                modes["tests/public/" + resource.path] = resource.mode
        files["tests/test.sh"] = b"#!/bin/bash\nset -euo pipefail\ncp -a /tests/public/. /\n" + files["tests/test.sh"]
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
        "environment": {"env": environment.environment_variables},
        "verifier": {
            "environment_mode": "separate",
            "timeout_sec": timeout,
            "env": grader_env,
            "environment": {
                "docker_image": grader_image,
                "workdir": grader_cwd,
                "env": grader.environment.environment_variables if grader.environment is not None else {},
            },
        },
        "artifacts": [{"source": path, "destination": path.removeprefix("/")} for path in dict.fromkeys(outputs)],
    }
    if environment.working_directory is not None:
        config["environment"]["workdir"] = environment.working_directory
    files["task.toml"] = tomlkit.dumps(config).encode()
    TaskConfig.model_validate_toml(files["task.toml"].decode())
    solution = {
        resource.path: resource_bytes(resource)
        for resource in task.resources.oracle
        if resource.path.startswith(("solution/", "tests/setup_files/"))
    }
    solution_modes = {resource.path: resource.mode for resource in task.resources.oracle if resource.mode}
    template = hashlib.sha256(files["tests/test.sh"]).hexdigest()[:12]
    return HarborRecord(
        path=row["original_path"],
        source=metadata["tasktrove_source"],
        family=family,
        template_id=template,
        converter="taskcompendium",
        mode=mode,
        dockerfile_id=hashlib.sha256(dockerfile.encode()).hexdigest()[:12],
        language="",
        tags=list(task.tags),
        has_solution=bool(solution),
        task_binary=archive_bytes(files, modes),
        solution_binary=archive_bytes(solution, solution_modes) if solution else None,
    )


def export_harbor(input_root: Path, output_root: Path, *, grader_image: str) -> dict[str, Any]:
    """Write the legacy parquet view and account for normalization and lowering failures."""
    manifest = json.loads((input_root / "manifest.json").read_text())
    sources = {source.name: source for source in all_sources().values() if source.pipeline is not None}
    source = sources[manifest["source"]]
    output_root.mkdir(parents=True, exist_ok=False)
    rejected = []
    input_count, exported_count = 0, 0
    counts: Counter[str] = Counter()
    paths = sorted(input_root.glob("normalize/*.parquet"))
    if not paths:
        raise ValueError(f"No normalized parquet shards under {input_root}")
    with pq.ParquetWriter(output_root / "tasks.parquet", TASKS_SCHEMA) as writer:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                converted = []
                for row in batch.to_pylist():
                    input_count += 1
                    counts.setdefault(row["source_row"].split("/", 1)[0], 0)
                    reason = row["normalization_reason"] if row["task_json"] is None else None
                    if row["task_json"] is not None:
                        try:
                            record = harbor_record(row, grader_image=grader_image, family=source.info.family)
                        except UnsupportedHarborTask as error:
                            reason = str(error)
                        else:
                            converted.append(asdict(record))
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
        "grader_image": grader_image,
        "source": source.name,
        "atlas_id": source.info.id,
        "harbor_config_validated": True,
        "runtime_verified": False,
        "limitation": "The supplied verifier image's dependency parity with the source package lock is unverified.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@click.command(help=__doc__)
@click.option("--input-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--grader-image", required=True, help="Explicit digest-pinned verifier image; image builds are separate.")
def main(input_root: Path, output_root: Path, grader_image: str) -> None:
    result = export_harbor(input_root, output_root, grader_image=grader_image)
    click.echo(json.dumps({key: value for key, value in result.items() if key != "rejections"}))


if __name__ == "__main__":
    main()
