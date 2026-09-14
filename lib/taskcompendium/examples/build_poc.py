# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a bounded semantic dataset and Harbor exports from checked-in source samples."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import msgspec
from tasktrove_verify.spec import Mode

from taskcompendium.execution import (
    Chat,
    ChatWithTools,
    DockerEnvironment,
    HarborTaskBinding,
    HarnessToolBinding,
    NoEnvironment,
    ShellSimEnvironment,
    ShellToolBinding,
    environment_for_requirements,
)
from taskcompendium.importers import (
    gsm8k,
    r2egym,
    tasktrove_answers,
    tasktrove_coding,
    tasktrove_judge,
    tasktrove_math,
    tasktrove_shell,
    tasktrove_structured,
)
from taskcompendium.importers.sequential import greeting_task, sentence_revision_task
from taskcompendium.importers.tasktrove import (
    CLEAN_09_RELEASE,
    INSPECTED_CONVERTER_REVISION,
    RELEASE_PRODUCER_REVISION,
    RELEASE_ROOT,
    read_archive,
)
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    HARBOR_REVISION,
    VERIFIER_REVISION,
    AssistantFinal,
    BoxedLatex,
    ContainerRuntime,
    FileSubmission,
    FinalState,
    ImageOverlay,
    JsonPath,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    PlainText,
    Rejected,
    Rendering,
    XmlPath,
)
from taskcompendium.rendering import TaskSpec
from taskcompendium.serialization import read_parquet, specification_hash, to_json, write_parquet

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"

CLEAN_09_ANSWER_SAMPLES = (
    (
        "tasktrove-clean-09/answers/mcq-1961bdb52b5a.tar.gz",
        "Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz",
        "qa-short-answer",
        lambda task, judge: tasktrove_answers.import_task(task),
    ),
    (
        "tasktrove-clean-09/answers/mcq-cce3426cf566.tar.gz",
        "Nemotron-RL-knowledge-mcqa-cce3426cf566.tar.gz",
        "qa-short-answer",
        lambda task, judge: tasktrove_answers.import_task(task),
    ),
    (
        "tasktrove-clean-09/judge/openqa-80f6c461ebcf.tar.gz",
        "openqa-80f6c461ebcf.tar.gz",
        "qa-short-answer",
        lambda task, judge: tasktrove_judge.import_task(task, judge),
    ),
    (
        "tasktrove-clean-09/judge/openqa-c7e9374b56ea.tar.gz",
        "openqa-c7e9374b56ea.tar.gz",
        "qa-short-answer",
        lambda task, judge: tasktrove_judge.import_task(task, judge),
    ),
)


def build(output: Path, runtime_image: str | None, r2e_runtimes: dict[str, ContainerRuntime] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    specs = []
    rejected = []
    inputs = []
    judge = JudgeConfig(
        JudgeModelPolicy("scripted-validation-judge", "small", "validation-fixture", "http://127.0.0.1:1/v1"),
        JudgeView(),
    )
    groups = (
        ("tasktrove/answers", "mcq", "qa-short-answer", tasktrove_answers.import_task),
        ("tasktrove/answers", "exact", "math-answer", tasktrove_answers.import_task),
        ("tasktrove/math", "math", "math-answer", tasktrove_math.import_task),
        ("tasktrove/judge", "judge", "qa-short-answer", lambda task: tasktrove_judge.import_task(task, judge)),
        ("structured", "json", "other", tasktrove_structured.import_task),
        ("structured", "xml", "other", tasktrove_structured.import_task),
    )
    for directory, prefix, family, importer in groups:
        for path in sorted((FIXTURES / directory).glob(f"{prefix}-row-*.tar.gz")):
            row = path.name.removeprefix(f"{prefix}-row-").removesuffix(".tar.gz")
            raw = path.read_bytes()
            inputs.append({"path": str(path.relative_to(FIXTURES)), "sha256": hashlib.sha256(raw).hexdigest()})
            result = importer(read_archive(raw, row, family))
            (rejected if isinstance(result, Rejected) else specs).append(result)
    for relative_path, row, family, importer in CLEAN_09_ANSWER_SAMPLES:
        path = FIXTURES / relative_path
        raw = path.read_bytes()
        inputs.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "tasktrove_release": CLEAN_09_RELEASE.root,
                "source_row": row,
            }
        )
        result = importer(read_archive(raw, row, family, CLEAN_09_RELEASE), judge)
        (rejected if isinstance(result, Rejected) else specs).append(result)
    if runtime_image is not None:
        for path in sorted((FIXTURES / "coding").glob("*.tar.gz")):
            row = path.name.split("-row-")[1].removesuffix(".tar.gz")
            raw = path.read_bytes()
            inputs.append({"path": str(path.relative_to(FIXTURES)), "sha256": hashlib.sha256(raw).hexdigest()})
            family = "unit-test-gen" if path.name.startswith("pytest") else "competitive-programming"
            result = tasktrove_coding.import_task(
                read_archive(raw, row, family), python_image=runtime_image, native_image=runtime_image
            )
            (rejected if isinstance(result, Rejected) else specs).append(result)
    if runtime_image is not None:
        for path in sorted((FIXTURES / "shell").glob("script-row-*.tar.gz")):
            raw = path.read_bytes()
            row = path.name.split("-row-")[1].removesuffix(".tar.gz")
            inputs.append({"path": str(path.relative_to(FIXTURES)), "sha256": hashlib.sha256(raw).hexdigest()})
            result = tasktrove_shell.import_task(
                read_archive(raw, row, "shell-cmd"), verifier_runtime=ContainerRuntime(runtime_image)
            )
            (rejected if isinstance(result, Rejected) else specs).append(result)
    path = FIXTURES / "gsm8k.json"
    raw = path.read_bytes()
    inputs.append({"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    data = json.loads(raw)
    family: TaskSpec = gsm8k.Gsm8kTaskSpec(
        {f'{data["split"]}/{row["row"]}': (row["data"]["question"], row["data"]["answer"]) for row in data["rows"]}
    )
    for row in data["rows"]:
        result = family.instantiate(f'{data["split"]}/{row["row"]}')
        (rejected if isinstance(result, Rejected) else specs).append(result)
    if r2e_runtimes:
        rows = []
        for filename in ("rows.json.gz", "broadened_rows.json.gz"):
            path = FIXTURES / "r2egym" / filename
            raw = path.read_bytes()
            rows.extend(json.loads(gzip.decompress(raw)))
            inputs.append({"path": str(path.relative_to(FIXTURES)), "sha256": hashlib.sha256(raw).hexdigest()})
        unknown = r2e_runtimes.keys() - {row["commit_hash"] for row in rows}
        if unknown:
            raise ValueError(f"R2E runtime keys do not identify source rows: {sorted(unknown)}")
        for row in rows:
            runtime = r2e_runtimes.get(row["commit_hash"])
            if runtime is None:
                continue
            result = r2egym.import_row(row, verifier_runtime=runtime)
            (rejected if isinstance(result, Rejected) else specs).append(result)
    if runtime_image is not None:
        specs.append(greeting_task(runtime_image))
    specs.append(sentence_revision_task())
    (output / "specifications").mkdir()
    exports = []
    for spec in specs:
        selected_environment = environment_for_requirements(spec.requirements)
        task_id = spec.id.replace("/", "-")
        (output / "specifications" / f"{task_id}.json").write_bytes(to_json(spec))
        if len(spec.steps) > 1:
            if isinstance(selected_environment, NoEnvironment):
                variants = [
                    (
                        Rendering(name, AssistantFinal(extractor)),
                        HarborTaskBinding(
                            selected_environment,
                            Chat(),
                            context="conversation",
                        ),
                    )
                    for name, extractor in (("chat-plain", PlainText()), ("chat-json", JsonPath()))
                ]
            else:
                variants = [
                    (
                        Rendering("workspace", FinalState(("greeting.py",))),
                        HarborTaskBinding(
                            selected_environment,
                            ChatWithTools((HarnessToolBinding("terminal", "docker"),)),
                        ),
                    )
                ]
            for rendering, execution in variants:
                destination = output / "harbor" / f"{task_id}-{rendering.id}"
                lower_to_harbor(spec, (rendering,) * len(spec.steps), execution, destination)
                exports.append(
                    {"task": spec.id, "rendering": rendering.id, "path": str(destination.relative_to(output))}
                )
            continue
        if spec.steps[0].answer_requirements.kind == "final_state":
            excluded_paths = ()
            if isinstance(spec.steps[0].verifier.runtime, ContainerRuntime) and isinstance(
                spec.steps[0].verifier.runtime.workspace, ImageOverlay
            ):
                excluded_paths = spec.steps[0].verifier.runtime.workspace.preserved_directories
            if spec.metadata.source.dataset == r2egym.DATASET:
                excluded_paths = tuple(sorted(set(excluded_paths) | set(r2egym.SNAPSHOT_EXCLUSIONS)))
            protocols = [
                Rendering(
                    "workspace",
                    FinalState(
                        (
                            ("/output/command_capture.txt",)
                            if isinstance(selected_environment, ShellSimEnvironment)
                            else (".",)
                        ),
                        excluded_paths=excluded_paths,
                    ),
                )
            ]
        elif spec.steps[0].answer_requirements.kind != "text":
            protocols = [Rendering("plain", AssistantFinal())]
        else:
            protocols = [
                Rendering(name, AssistantFinal(extractor))
                for name, extractor in (
                    ("plain", PlainText()),
                    ("json", JsonPath()),
                    ("xml", XmlPath()),
                )
            ]
            if spec.steps[0].verifier.mode == Mode.MATH:
                protocols += [
                    Rendering("boxed", AssistantFinal(BoxedLatex())),
                    Rendering("file", FileSubmission("/app/answer.txt")),
                ]
        lowerings = []
        for protocol in protocols:
            environment = selected_environment
            if isinstance(protocol.submission, FileSubmission) and isinstance(environment, NoEnvironment):
                environment = ShellSimEnvironment()
            lowerings.append(
                (
                    protocol,
                    HarborTaskBinding(
                        environment,
                        (
                            Chat()
                            if isinstance(environment, NoEnvironment)
                            else ChatWithTools(
                                (
                                    (
                                        ShellToolBinding("shell", "shellsim")
                                        if isinstance(environment, ShellSimEnvironment)
                                        else HarnessToolBinding("terminal", "docker")
                                    ),
                                )
                            )
                        ),
                    ),
                )
            )
        if spec.steps[0].verifier.mode == Mode.MCQ:
            for name, extractor, filename in (("plain", PlainText(), "answer.txt"), ("json", JsonPath(), "answer.json")):
                lowerings.append(
                    (
                        Rendering(f"shellsim-file-{name}", FileSubmission(f"/app/{filename}", extractor)),
                        HarborTaskBinding(
                            ShellSimEnvironment(),
                            ChatWithTools((ShellToolBinding("shell", "shellsim"),)),
                        ),
                    )
                )
                lowerings.append(
                    (
                        Rendering(f"shellsim-tool-chat-file-{name}", FileSubmission(f"/app/{filename}", extractor)),
                        HarborTaskBinding(
                            ShellSimEnvironment(),
                            ChatWithTools((ShellToolBinding("shell", "shellsim"),)),
                        ),
                    )
                )
                if runtime_image is not None:
                    lowerings.append(
                        (
                            Rendering(f"docker-terminal-file-{name}", FileSubmission(f"/app/{filename}", extractor)),
                            HarborTaskBinding(
                                DockerEnvironment(runtime_image),
                                ChatWithTools((HarnessToolBinding("terminal", "docker"),)),
                            ),
                        )
                    )
        if isinstance(selected_environment, DockerEnvironment) and selected_environment.image == runtime_image:
            lowerings.append(
                (
                    msgspec.structs.replace(protocols[0], id="terminal-workspace"),
                    HarborTaskBinding(
                        selected_environment,
                        ChatWithTools((HarnessToolBinding("terminal", "docker"),)),
                    ),
                )
            )
        for protocol, execution in lowerings:
            destination = output / "harbor" / f"{task_id}-{protocol.id}"
            lower_to_harbor(spec, (protocol,), execution, destination)
            exports.append({"task": spec.id, "rendering": protocol.id, "path": str(destination.relative_to(output))})
    write_parquet(specs, str(output / "specifications.parquet"))
    restored = list(read_parquet(str(output / "specifications.parquet")))
    if [specification_hash(s) for s in restored] != [specification_hash(s) for s in specs]:
        raise RuntimeError("Generated Parquet does not round-trip to the source specifications")
    (output / "rejections.jsonl").write_bytes(b"".join(msgspec.json.encode(r) + b"\n" for r in rejected))
    manifest = {
        "tasktrove_releases": {
            "2026.09.10.8": {
                "root": RELEASE_ROOT,
                "verifier_revision": "b2b68d8b0a770cdc0ab3903780172c4b3eea81b1",
            },
            CLEAN_09_RELEASE.version: {
                "root": CLEAN_09_RELEASE.root,
                "verifier_revision": CLEAN_09_RELEASE.verifier_revision,
            },
        },
        "inspected_converter_revision": INSPECTED_CONVERTER_REVISION,
        "release_cleanup_code_revision": RELEASE_PRODUCER_REVISION,
        "default_verifier_revision": VERIFIER_REVISION,
        "harbor_revision": HARBOR_REVISION,
        "specifications": len(specs),
        "rejections": len(rejected),
        "inputs": inputs,
        "exports": exports,
        "runtime_image": runtime_image,
        "r2e_runtimes": msgspec.to_builtins(r2e_runtimes or {}),
        "validation": "Serialization and export only; execution evidence is recorded separately.",
        "judge_validation": "Plumbing-only fixture policy; replace endpoint and policy explicitly for live judging.",
        "execution": (
            "Replay templates need response or commands; Terminus-2 templates need a model and provider configuration."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "specifications": len(specs), "exports": len(exports), "rejections": len(rejected)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-image", help="Immutable compatible coding/verifier image; omitting it excludes coding samples"
    )
    parser.add_argument(
        "--r2e-runtimes", type=Path, help="JSON mapping R2E source commits to compatible ContainerRuntime objects"
    )
    args = parser.parse_args()
    runtimes = (
        msgspec.json.decode(args.r2e_runtimes.read_bytes(), type=dict[str, ContainerRuntime])
        if args.r2e_runtimes
        else None
    )
    build(args.output, args.runtime_image, runtimes)
