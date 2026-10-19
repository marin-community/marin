# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export individually control-qualified staged tasks and run their native sessions."""

import argparse
import asyncio
import base64
import json
import shlex
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from rolloutengine.contracts import ModelTurn
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec, TraceMode
from shellbox.machine import NetworkPolicy
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    InlineFile,
    PlainText,
    ResourceGroups,
    SessionGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation.swesmith_staged import is_test_path, sha256, source_tasks, write_json
from experiments.post_training.task_curation.swesmith_staged_runtime import (
    BACKEND_NAME,
    SESSION_NAME,
    LocalArtifactDockerFactory,
    session_factory,
)


def runtime_task(task: TaskSpec) -> LoweredTaskSpec:
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(
            task_machine=MachineRuntimeSpec(
                backend=BACKEND_NAME,
                network=NetworkPolicy.ALLOW,
                cpus=2,
                memory_mb=3072,
                storage_mb=None,
                gpus=0,
                user=None,
                startup_timeout=120,
                cleanup_timeout=30,
            ),
            verifier_machine=None,
        ),
        session=TaskSessionSpec(
            task_session=SESSION_NAME,
            max_turns=64,
            model_turn_timeout=120,
            command_timeout=600,
            tool_turn_timeout=610,
            total_turn_timeout=900,
            attempt_timeout=1800,
            verifier_timeout=900,
            cleanup_timeout=30,
            max_total_tokens=32768,
        ),
    )


def export(data: Path, output: Path) -> list[dict]:
    sources, _ = source_tasks(data)
    sources = {task.parent_id: task for task in sources}
    results = {}
    quarantined = []
    for path in sorted((output / "pilots").glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        results[result["parent_id"]] = (path, result)
        unsafe_paths = [name for name in result.get("changed_source_paths", []) if is_test_path(name)]
        if unsafe_paths:
            quarantined.append(
                {
                    "qualification": str(path.relative_to(output)),
                    "paths": unsafe_paths,
                    "reason": "historical hidden-test classification defect; not admitted",
                }
            )
    results = {parent: pair for parent, pair in results.items() if pair[1]["status"] == "controls_validated"}
    ledger_path = output / "cohort-stage1-decisions.json"
    ledger = (
        {row["parent_id"]: row for row in json.loads(ledger_path.read_text())["decisions"]}
        if ledger_path.exists()
        else {}
    )
    for path in sorted((output / "cohort-stage1").glob("*/result.json")):
        result = json.loads(path.read_text())
        decision = ledger.get(result["parent_id"], {})
        if (
            result["parent_id"] not in results
            and decision.get("status") == "stage1_controls_validated"
            and result["status"] == "stage1_controls_validated"
            and result.get("qualification_key") == decision.get("qualification_key")
            and result.get("qualification_key") is not None
        ):
            results[result["parent_id"]] = (path, result)
    records = []
    images = {}
    base_record = json.loads((output / "base-image/resolved.json").read_text())
    for parent_id, (path, result) in sorted(results.items()):
        source = sources[parent_id]
        setup_contract = json.loads((path.parent / "private/stage1-contract.json").read_text())
        verifier_paths = tomllib.loads(source.files["tests/verifier.toml"].decode())["paths"]
        unsafe = [
            p
            for p in result.get("changed_source_paths", [])
            if is_test_path(p) or p in verifier_paths or p not in setup_contract["tracked_sha256"]
        ]
        if unsafe:
            quarantined.append(
                {
                    "parent_id": parent_id,
                    "qualification": str(path.relative_to(output)),
                    "reason": "earlier harness classified verifier tests as repair source",
                    "paths": unsafe,
                }
            )
        for stage, image_key in (
            ("stage1", "base_image"),
            ("stage3-complete", "complete_image"),
            ("stage3-partial", "partial_image"),
        ):
            if image_key not in result:
                continue
            if stage != "stage1" and unsafe:
                continue
            image = base_record["image"] if stage == "stage1" else result[image_key]
            locator = "local-artifact/swesmith@" + image
            images[locator] = image
            contract = {**setup_contract, "stage": stage}
            if stage == "stage1":
                context = (
                    "Prepare the repository and Python environment for a later repair task. No bug fix is requested. "
                    "Use /testbed, the exact fork and task branch below; retain all tracked files unchanged. "
                    f"Required HEAD: {result['mutation_commit']}. "
                    f"The {setup_contract['import_name']} module must import "
                    "from this installed checkout when Python runs outside the repository. Finish with a final response "
                    "when ready. Readiness is checked from resulting state, not command spelling or a success claim.\n\n"
                    "```bash\n" + source.setup + "\n" + result["dependency_command"] + "\n```"
                )
                paths = ()
            else:
                contract.update(
                    verifier_image=result["verifier_image"],
                    changed_source_paths=result["changed_source_paths"],
                    verifier=tomllib.loads(source.files["tests/verifier.toml"].decode()),
                )
                context = (
                    "You are taking over a repair in /testbed. "
                    "Previous work is unverified and may be partial or complete. "
                    "Inspect the implementation, run appropriate checks, repair it if necessary, "
                    "and submit a final response. "
                    "A correct handoff needs no source edits. Hidden tests are checked independently after submission; "
                    "their absence from this checkout does not mean the source is incorrect. "
                    "Preserve supplied tests and "
                    "project files. Changes may be made in these repair source files: "
                    + ", ".join(result["changed_source_paths"])
                    + ". The repository history and private verification material are unavailable.\n\nIssue:\n"
                    + source.problem
                )
                paths = tuple("/testbed/" + path for path in sorted(setup_contract["tracked_sha256"]))
            task_id = parent_id + "--" + stage + "--v1"
            task = TaskSpec(
                id=task_id,
                context=ConversationInput(events=(TextMessage(role="user", content=context),)),
                environment_requirements=EnvironmentRequirements(
                    capabilities=("shell", "filesystem", "process", "network"),
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    docker_image=locator,
                    working_directory="/testbed",
                ),
                answer_type=AnswerType.WORKSPACE_STATE,
                answer_format=PlainText(),
                grader=SessionGrader(),
                output_paths=paths,
                source=Source(
                    dataset="open-athena/task-trove",
                    revision="cc94912d3262dc05d538e59761f547a1bb65b894",
                    row=parent_id,
                    importer_revision=sha256(Path(__file__).read_bytes()),
                ),
                resources=ResourceGroups(
                    verifier=(
                        TaskResource(
                            path="staged-contract.json",
                            source=InlineFile(
                                content_base64=base64.b64encode(json.dumps(contract, sort_keys=True).encode()).decode()
                            ),
                        ),
                    )
                ),
                tags=("swesmith", stage, source.split, "local-native-control-qualified"),
            )
            lowered = runtime_task(task)
            records.append(
                {
                    "parent_id": parent_id,
                    "split": source.split,
                    "stage": stage,
                    "qualification": str(path.relative_to(output)),
                    "lowered_task_spec": lowered.model_dump(mode="json"),
                }
            )
    for split in ("train", "validation"):
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records if row["split"] == split)
        (output / f"{split}.jsonl").write_text(text)
    write_json(output / "local-images.json", images)
    write_json(output / "quarantined-stage3-attempts.json", quarantined)
    write_json(
        output / "variants-manifest.json",
        {
            "status": "control_qualified_pending_runtime_controls",
            "variants": len(records),
            "parents": len(results),
            "source_repositories": sorted({sources[parent].origin for parent in results}),
            "output_sha256": {
                f"{split}.jsonl": sha256((output / f"{split}.jsonl").read_bytes()) for split in ("train", "validation")
            },
            "image_locator_contract": (
                "local-artifact image locators carry Docker config IDs; "
                "require the supplied custom factory and local images, not registry pull"
            ),
            "verifier_boundary": (
                "Stage 1 checks actor state with post-episode trusted inspector input, "
                "but is not hardened against malicious Python/runtime tampering. "
                "Stage 3 transfers permitted source only into a separate trusted verifier image "
                "and checks supplied-file integrity."
            ),
            "known_limits": [
                "No model capability evaluation",
                "No public image publication",
                "Parent-task split preserved, existing cross-split repositories retained",
                "No command-log/clone-process behavior reward",
            ],
        },
    )
    return records


class CommandModel:
    def __init__(self, commands: list[str]):
        self.commands = commands
        self.index = 0

    async def complete(self, request):
        if self.index < len(self.commands):
            command = self.commands[self.index]
            self.index += 1
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"control-{self.index}",
                        "type": "function",
                        "function": {
                            "name": request.options["tools"][0]["function"]["name"],
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "The handoff is ready for verification."}
        return ModelTurn(message, (), (), (), "stop")


async def runtime_controls(records: list[dict], output: Path) -> list[dict]:
    images = json.loads((output / "local-images.json").read_text())
    controls = []
    run_path = output / "private" / f"runtime-run-{time.time_ns()}"
    write_json(run_path / "input-records.json", records)
    write_json(run_path / "local-images.json", images)
    code_paths = sorted(Path(__file__).parent.glob("swesmith_staged*.py"))
    code_hashes = {path.name: sha256(path.read_bytes()) for path in code_paths}
    qualification_hashes = {
        row["qualification"]: sha256((output / row["qualification"]).read_bytes()) for row in records
    }
    write_json(
        run_path / "attestation-input.json",
        {
            "code_sha256": code_hashes,
            "qualification_sha256": qualification_hashes,
            "coverage_policy": (
                "Every exact actor-image, stage and repository tuple; every parent has individual authoring controls"
            ),
        },
    )
    evidence_path = run_path / "controls.json"
    exercised = set()
    for row in records:
        lowered = LoweredTaskSpec.model_validate(row["lowered_task_spec"])
        contract_bytes = base64.b64decode(lowered.task.resources.verifier[0].source.content_base64)
        contract = json.loads(contract_bytes)
        coverage_key = (contract["origin"], row["stage"], lowered.task.environment_requirements.docker_image)
        if coverage_key in exercised:
            continue
        exercised.add(coverage_key)
        lowered = lowered.model_copy(
            update={
                "session": lowered.session.model_copy(update={"trace": TraceMode.MESSAGES, "max_total_tokens": None})
            }
        )
        result_path = output / row["qualification"]
        pilot = json.loads(result_path.read_text())
        if row["stage"] == "stage1":
            prompt = lowered.task.context.events[0].content
            setup = prompt.split("```bash\n", 1)[1].split("\n```", 1)[0]
            cases = [("ready", [setup], 1.0), ("false_submission", [], 0.0)]
        elif row["stage"] == "stage3-complete":
            # Verification-only is valid: no source edit and no bespoke added test.
            cases = [("complete_no_edits", ["python -m compileall -q " + " ".join(pilot["changed_source_paths"])], 1.0)]
            supplied_tests = [path for path in contract["tracked_sha256"] if is_test_path(path)]
            if supplied_tests:
                test_path = shlex.quote("/testbed/" + sorted(supplied_tests)[0])
                cases.extend(
                    [
                        ("delete_supplied_test", ["rm -- " + test_path], 0.0),
                        ("weaken_supplied_test", ["printf '# replaced test\\n' > " + test_path], 0.0),
                    ]
                )
        else:
            patch = (result_path.parent / "private/gold.patch").read_bytes()
            partial = (result_path.parent / "private/partial.patch").read_bytes()
            # These control-only patches are fed to the scripted oracle, never the task prompt.
            repair = (
                "import base64,subprocess\n"
                + "subprocess.run(['git','apply','--recount','--reverse','-'],input=base64.b64decode("
                + repr(base64.b64encode(partial).decode())
                + "),check=True)\n"
                + "subprocess.run(['git','apply','--recount','-'],input=base64.b64decode("
                + repr(base64.b64encode(patch).decode())
                + "),check=True)\n"
            )
            cases = [("partial_unmodified", [], 0.0), ("partial_repaired", ["python - <<'PY'\n" + repair + "PY"], 1.0)]
        for name, commands, expected in cases:
            model = CommandModel(commands)
            engine = RolloutEngine(
                model.complete,
                {BACKEND_NAME: LocalArtifactDockerFactory(images)},
                sessions={SESSION_NAME: session_factory(output / "runtime-logs")},
            )
            rollout = await engine.run(lowered)
            controls.append(
                {
                    "task_id": lowered.task.id,
                    "coverage_key": coverage_key,
                    "case": name,
                    "expected": expected,
                    "grade": asdict(rollout.grade),
                    "reward": rollout.grade.reward,
                    "passed": (
                        rollout.grade.status.value == "graded"
                        and rollout.failure is None
                        and rollout.grade.reward == expected
                    ),
                    "messages": rollout.messages,
                }
            )
            write_json(evidence_path, controls)
    write_json(output / "runtime-controls.json", controls)
    current_records = [
        json.loads(line)
        for split in ("train", "validation")
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]
    stable = (
        sorted(current_records, key=lambda row: row["lowered_task_spec"]["task"]["id"])
        == sorted(records, key=lambda row: row["lowered_task_spec"]["task"]["id"])
        and code_hashes == {path.name: sha256(path.read_bytes()) for path in code_paths}
        and qualification_hashes == {name: sha256((output / name).read_bytes()) for name in qualification_hashes}
    )
    attestation = {
        "stable_inputs": stable,
        "controls": len(controls),
        "passed": sum(control["passed"] for control in controls),
        "runtime_evidence": str(run_path.relative_to(output)),
        "code_sha256": code_hashes,
    }
    write_json(run_path / "attestation-result.json", attestation)
    manifest_path = output / "variants-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_attestation"] = attestation
    if stable and controls and all(control["passed"] for control in controls):
        manifest["status"] = "validated_local_scoped_subset"
    write_json(manifest_path, manifest)
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-controls", action="store_true")
    args = parser.parse_args()
    records = export(args.data, args.output)
    if args.runtime_controls:
        controls = asyncio.run(runtime_controls(records, args.output))
        print(
            json.dumps(
                {"variants": len(records), "controls": len(controls), "passed": sum(c["passed"] for c in controls)}
            )
        )
    else:
        print(json.dumps({"variants": len(records)}))


if __name__ == "__main__":
    main()
