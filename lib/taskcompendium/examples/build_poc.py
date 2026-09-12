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
from taskcompendium.importers.tasktrove import (
    INSPECTED_CONVERTER_REVISION,
    RELEASE_PRODUCER_REVISION,
    RELEASE_ROOT,
    read_archive,
)
from taskcompendium.lowering import export_task
from taskcompendium.models import (
    HARBOR_REVISION,
    VERIFIER_REVISION,
    AssistantFinal,
    BoxedLatex,
    Chat,
    ChatWithTools,
    ContainerRuntime,
    DockerEnvironment,
    ExecutionConfig,
    FileSubmission,
    FinalState,
    ImageOverlay,
    JsonPath,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    NoEnvironment,
    PlainText,
    Protocol,
    Rejected,
    ShellSimEnvironment,
    Source,
    XmlPath,
)
from taskcompendium.serialization import read_parquet, specification_hash, to_json, write_parquet

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"


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
    for row in data["rows"]:
        result = gsm8k.import_row(
            row["data"]["question"],
            row["data"]["answer"],
            Source(gsm8k.DATASET, gsm8k.REVISION, f'{data["split"]}/{row["row"]}', gsm8k.IMPORTER_REVISION),
        )
        (rejected if isinstance(result, Rejected) else specs).append(result)
    if r2e_runtimes:
        path = FIXTURES / "r2egym/rows.json.gz"
        raw = path.read_bytes()
        rows = json.loads(gzip.decompress(raw))
        unknown = r2e_runtimes.keys() - {row["commit_hash"] for row in rows}
        if unknown:
            raise ValueError(f"R2E runtime keys do not identify source rows: {sorted(unknown)}")
        inputs.append({"path": str(path.relative_to(FIXTURES)), "sha256": hashlib.sha256(raw).hexdigest()})
        for row in rows:
            runtime = r2e_runtimes.get(row["commit_hash"])
            if runtime is None:
                continue
            result = r2egym.import_row(row, verifier_runtime=runtime)
            (rejected if isinstance(result, Rejected) else specs).append(result)
    (output / "specifications").mkdir()
    exports = []
    for spec in specs:
        task_id = spec.id.replace("/", "-")
        (output / "specifications" / f"{task_id}.json").write_bytes(to_json(spec))
        if spec.answer_requirements.kind == "final_state":
            excluded_paths = ()
            if isinstance(spec.verifier_runtime, ContainerRuntime) and isinstance(
                spec.verifier_runtime.workspace, ImageOverlay
            ):
                excluded_paths = spec.verifier_runtime.workspace.preserved_directories
            if spec.metadata.source.dataset == r2egym.DATASET:
                excluded_paths = tuple(sorted(set(excluded_paths) | set(r2egym.SNAPSHOT_EXCLUSIONS)))
            protocols = [
                Protocol(
                    "workspace",
                    ChatWithTools(),
                    FinalState(
                        (
                            ("/output/command_capture.txt",)
                            if isinstance(spec.environment, ShellSimEnvironment)
                            else (".",)
                        ),
                        excluded_paths=excluded_paths,
                    ),
                )
            ]
        elif spec.answer_requirements.kind != "value":
            protocols = [Protocol("plain", Chat(), AssistantFinal())]
        else:
            protocols = [
                Protocol(name, Chat(), AssistantFinal(extractor))
                for name, extractor in (
                    ("plain", PlainText()),
                    ("json", JsonPath()),
                    ("xml", XmlPath()),
                )
            ]
            if spec.verifier.mode == Mode.MATH:
                protocols += [
                    Protocol("boxed", Chat(), AssistantFinal(BoxedLatex())),
                    Protocol("file", ChatWithTools(), FileSubmission("/app/answer.txt")),
                ]
        lowerings = []
        for protocol in protocols:
            environment = spec.environment
            if isinstance(protocol.interaction, ChatWithTools) and isinstance(environment, NoEnvironment):
                environment = ShellSimEnvironment()
            lowerings.append((protocol, ExecutionConfig("replay", environment)))
        if spec.verifier.mode == Mode.MCQ:
            for name, extractor, filename in (("plain", PlainText(), "answer.txt"), ("json", JsonPath(), "answer.json")):
                lowerings.append(
                    (
                        Protocol(
                            f"shellsim-file-{name}", ChatWithTools(), FileSubmission(f"/app/{filename}", extractor)
                        ),
                        ExecutionConfig("replay", ShellSimEnvironment()),
                    )
                )
                if runtime_image is not None:
                    lowerings.append(
                        (
                            Protocol(
                                f"docker-terminus-file-{name}",
                                ChatWithTools(),
                                FileSubmission(f"/app/{filename}", extractor),
                            ),
                            ExecutionConfig("terminus-2", DockerEnvironment(runtime_image)),
                        )
                    )
        if isinstance(spec.environment, DockerEnvironment) and spec.environment.image == runtime_image:
            lowerings.append(
                (
                    msgspec.structs.replace(protocols[0], id="terminus-workspace"),
                    ExecutionConfig("terminus-2", spec.environment),
                )
            )
        for protocol, execution in lowerings:
            destination = output / "harbor" / f"{task_id}-{protocol.id}"
            export_task(spec, protocol, execution, destination)
            exports.append({"task": spec.id, "protocol": protocol.id, "path": str(destination.relative_to(output))})
    write_parquet(specs, str(output / "specifications.parquet"))
    restored = list(read_parquet(str(output / "specifications.parquet")))
    if [specification_hash(s) for s in restored] != [specification_hash(s) for s in specs]:
        raise RuntimeError("Generated Parquet does not round-trip to the source specifications")
    (output / "rejections.jsonl").write_bytes(b"".join(msgspec.json.encode(r) + b"\n" for r in rejected))
    manifest = {
        "release": RELEASE_ROOT,
        "inspected_converter_revision": INSPECTED_CONVERTER_REVISION,
        "release_cleanup_code_revision": RELEASE_PRODUCER_REVISION,
        "verifier_revision": VERIFIER_REVISION,
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
