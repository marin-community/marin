# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export and exercise shared-image Stage3 tasks without changing sealed v1."""

import argparse
import asyncio
import base64
import json
import time
import tomllib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import LoweredTaskSpec, TraceMode
from taskcompendium.models import ConversationInput, InlineFile, ResourceGroups, TaskResource, TextMessage

from experiments.post_training.task_curation.swesmith_staged import is_test_path, sha256, source_tasks, write_json
from experiments.post_training.task_curation.swesmith_staged_export import CommandModel
from experiments.post_training.task_curation.swesmith_staged_runtime import BACKEND_NAME, LocalArtifactDockerFactory
from experiments.post_training.task_curation.swesmith_staged_v2_runtime import SESSION_NAME, session_factory


def resource(name: str, payload: bytes) -> TaskResource:
    return TaskResource(path=name, source=InlineFile(content_base64=base64.b64encode(payload).decode()))


def export(previous: Path, output: Path) -> list[dict]:
    sources, inventory = source_tasks(previous.parent)
    sources = {task.parent_id: task for task in sources}
    template = next(
        json.loads(line)
        for line in (previous / "train.jsonl").read_text().splitlines()
        if json.loads(line)["stage"] == "stage3-complete"
    )
    lowered = LoweredTaskSpec.model_validate(template["lowered_task_spec"])
    records = []
    images = {}
    for path in sorted((output / "parents").glob("*/result.json")):
        parent = json.loads(path.read_text())
        if not parent["complete_passed"]:
            continue
        repository = json.loads((output / "repositories" / parent["repository"] / "result.json").read_text())
        source = sources[parent["parent_id"]]
        verifier = tomllib.loads(source.files["tests/verifier.toml"].decode())
        if (
            source.split != parent["split"]
            or source.task_hash != parent["source_task_sha256"]
            or verifier != parent["verifier"]
        ):
            raise ValueError("Source split, identity or verifier changed")
        report = json.loads((output / parent["complete_report"]).read_text())
        required = verifier["must_pass"] + verifier.get("must_not_break", [])
        if (
            report["fixed_commit"] != parent["fixed_commit"]
            or report["fixed_commit"] != repository["fixed_commit"]
            or report["dependency_image"] != repository["dependency_image"]
            or report["paths"] != verifier["paths"]
            or report["args"] != verifier.get("args", [])
            or report["check"]["returncode"] != 0
            or not required
            or any(report["outcomes"].get(node) != "passed" for node in required)
        ):
            raise ValueError("Complete handoff lacks matching original-test evidence")
        if any(
            is_test_path(p) or p in verifier["paths"] or p not in repository["tracked_sha256"]
            for p in parent["changed_source_paths"]
        ):
            raise ValueError("Repair scope contains tests or uncaptured files")
        note_path = path.parent / "glm-handoff.json"
        if not note_path.exists():
            continue
        note = json.loads(note_path.read_text())
        if note["status"] != "accepted":
            continue
        locator = "local-artifact/swesmith@" + repository["actor_image"]
        images[locator] = repository["actor_image"]
        partial_ready = (
            parent["partial_status"] == "qualified" and parent.get("partial_control_version") == "fresh-bytecode-v2"
        )
        stages = ["stage3-complete"] + (["stage3-partial"] if partial_ready else [])
        for stage in stages:
            contract = {
                "stage": stage,
                "origin": parent["origin"],
                "verifier_image": repository["verifier_image"],
                "tracked_sha256": repository["tracked_sha256"],
                "verifier": parent["verifier"],
                "changed_source_paths": parent["changed_source_paths"],
            }
            workers = ()
            if stage == "stage3-partial":
                payload = (path.parent / "handoff-files.json").read_bytes()
                if sha256(payload) != parent["handoff_sha256"]:
                    raise ValueError("Qualified handoff changed")
                failed = parent["partial_before"].get("failed_required", {})
                if (
                    not any(node in verifier["must_pass"] and outcome == "failed" for node, outcome in failed.items())
                    or not parent["partial_after"]["passed"]
                ):
                    raise ValueError("Partial lacks original-test pre/post witness")
                contract["handoff_sha256"] = sha256(payload)
                workers = (resource("handoff-files.json", payload),)
            prompt = (
                "Take over the unverified repair in /testbed. Inspect the implementation, run appropriate checks, "
                "repair it if necessary, and submit a final response. The handoff may already be correct; no source "
                "edit or new test is required for a correct handoff. Original hidden tests run independently after "
                "submission. Their absence here is intentional, not evidence of an incorrect implementation. "
                "Preserve project files; permitted repair files: "
                + ", ".join(parent["changed_source_paths"])
                + ". Do not alter other files or add replacement test infrastructure. Repository history is unavailable."
                + "\n\nIssue:\n"
                + parent["problem"]
                + "\n\nUnverified handoff note (not a test result):\n"
                + note["note"]
            )
            task = lowered.task.model_copy(
                update={
                    "id": parent["parent_id"] + "--" + stage + "--v2",
                    "context": ConversationInput(events=(TextMessage(role="user", content=prompt),)),
                    "environment_requirements": lowered.task.environment_requirements.model_copy(
                        update={"docker_image": locator}
                    ),
                    "output_paths": tuple("/testbed/" + p for p in sorted(repository["tracked_sha256"])),
                    "resources": ResourceGroups(
                        worker=workers, verifier=(resource("staged-contract.json", json.dumps(contract).encode()),)
                    ),
                    "source": lowered.task.source.model_copy(
                        update={"row": parent["parent_id"], "importer_revision": sha256(Path(__file__).read_bytes())}
                    ),
                    "tags": ("swesmith", stage, parent["split"], "shared-image-v2"),
                }
            )
            item = lowered.model_copy(
                update={"task": task, "session": lowered.session.model_copy(update={"task_session": SESSION_NAME})}
            )
            records.append(
                {
                    "parent_id": parent["parent_id"],
                    "split": parent["split"],
                    "stage": stage,
                    "qualification": str(path.relative_to(output)),
                    "lowered_task_spec": item.model_dump(mode="json"),
                }
            )
    for split in ("train", "validation"):
        (output / (split + ".jsonl")).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records if row["split"] == split)
        )
    write_json(output / "local-images.json", images)
    write_json(
        output / "variants-manifest.json",
        {
            "status": "qualified_pending_runtime",
            "source_inventory": inventory,
            "variants": len(records),
            "parents": len({row["parent_id"] for row in records}),
            "counts": dict(Counter(row["split"] + "/" + row["stage"] for row in records)),
            "output_sha256": {
                split + ".jsonl": sha256((output / (split + ".jsonl")).read_bytes()) for split in ("train", "validation")
            },
        },
    )
    return records


async def runtime_controls(records: list[dict], output: Path) -> dict:
    run = output / "private" / ("runtime-" + str(time.time_ns()))
    write_json(run / "input-records.json", records)
    images = json.loads((output / "local-images.json").read_text())
    code_paths = sorted(Path(__file__).parent.glob("swesmith_staged*.py"))
    hashes = {p.name: sha256(p.read_bytes()) for p in code_paths}
    evidence_files = set()
    for row in records:
        path = output / row["qualification"]
        parent = json.loads(path.read_text())
        evidence_files.update(
            (
                path,
                output / parent["complete_report"],
                output / "repositories" / parent["repository"] / "result.json",
                path.parent / "glm-handoff.json",
                path.parent / "private/gold-files.json",
            )
        )
        if row["stage"] == "stage3-partial":
            evidence_files.add(path.parent / "handoff-files.json")
    qualifications = {str(path.relative_to(output)): sha256(path.read_bytes()) for path in sorted(evidence_files)}
    write_json(run / "inputs.json", {"code_sha256": hashes, "qualification_sha256": qualifications, "images": images})
    selected = {}
    for row in records:
        parent = json.loads((output / row["qualification"]).read_text())
        selected.setdefault((parent["repository"], row["stage"]), row)
    controls = []
    for key, row in selected.items():
        lowered = LoweredTaskSpec.model_validate(row["lowered_task_spec"])
        lowered = lowered.model_copy(
            update={
                "session": lowered.session.model_copy(update={"trace": TraceMode.MESSAGES, "max_total_tokens": None})
            }
        )
        path = output / row["qualification"]
        parent = json.loads(path.read_text())
        if row["stage"] == "stage3-complete":
            cases = [("complete_no_edits", ["python -m compileall -q " + " ".join(parent["changed_source_paths"])], 1.0)]
            contract = json.loads(base64.b64decode(lowered.task.resources.verifier[0].source.content_base64))
            protected = next(p for p in contract["tracked_sha256"] if p not in parent["changed_source_paths"])
            command = (
                "python - <<'PY'\nfrom pathlib import Path\nPath("
                + repr("/testbed/" + protected)
                + ").write_text('altered')\nPY"
            )
            cases.append(("weaken_protected_file", [command], 0.0))
        else:
            gold = json.loads((path.parent / "private/gold-files.json").read_text())
            command = (
                "python - <<'PY'\nimport pathlib,base64\nfiles="
                + repr(gold)
                + "\nfor p,c in files.items():pathlib.Path('/testbed',p).write_bytes(base64.b64decode(c))\nPY"
            )
            fake_test = parent["verifier"]["paths"][0].split("::", 1)[0]
            inject = (
                "python - <<'PY'\nfrom pathlib import Path\np=Path('/testbed')/"
                + repr(fake_test)
                + "\np.parent.mkdir(parents=True,exist_ok=True)\n"
                + "p.write_text('def test_replacement(): assert True\\n')\nPY"
            )
            cases = [
                ("partial_unmodified", [], 0.0),
                ("partial_injected_passing_test", [inject], 0.0),
                ("partial_repaired", [command], 1.0),
            ]
        for case, commands, expected in cases:
            engine = RolloutEngine(
                CommandModel(commands).complete,
                {BACKEND_NAME: LocalArtifactDockerFactory(images)},
                sessions={SESSION_NAME: session_factory(output / "runtime-logs")},
            )
            result = await engine.run(lowered)
            controls.append(
                {
                    "task_id": lowered.task.id,
                    "coverage_key": key,
                    "case": case,
                    "expected": expected,
                    "grade": asdict(result.grade),
                    "failure": str(result.failure) if result.failure else None,
                    "passed": (
                        result.grade.status.value == "graded"
                        and result.failure is None
                        and result.grade.reward == expected
                    ),
                    "messages": result.messages,
                }
            )
            write_json(run / "controls.json", controls)
    current = [
        json.loads(line)
        for split in ("train", "validation")
        for line in (output / (split + ".jsonl")).read_text().splitlines()
    ]
    stable = (
        images == json.loads((output / "local-images.json").read_text())
        and hashes == {p.name: sha256(p.read_bytes()) for p in code_paths}
        and qualifications == {p: sha256((output / p).read_bytes()) for p in qualifications}
        and sorted(current, key=lambda r: r["lowered_task_spec"]["task"]["id"])
        == sorted(records, key=lambda r: r["lowered_task_spec"]["task"]["id"])
    )
    attestation = {
        "stable_inputs": stable,
        "controls": len(controls),
        "passed": sum(row["passed"] for row in controls),
        "code_sha256": hashes,
        "evidence": str(run.relative_to(output)),
        "runtime_coverage": (
            "One complete and one partial per supported repository; all parent handoffs individually qualified"
        ),
    }
    write_json(run / "attestation.json", attestation)
    manifest = json.loads((output / "variants-manifest.json").read_text())
    manifest["runtime_attestation"] = attestation
    partial_count = sum(row["stage"] == "stage3-partial" for row in records)
    if (
        stable
        and controls
        and all(row["passed"] for row in controls)
        and manifest["parents"] >= 500
        and partial_count >= 100
    ):
        manifest["status"] = "validated_local_stage3_subset"
    write_json(output / "variants-manifest.json", manifest)
    return attestation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-controls", action="store_true")
    args = parser.parse_args()
    records = export(args.previous, args.output)
    if args.runtime_controls:
        print(json.dumps(asyncio.run(runtime_controls(records, args.output))))
    else:
        print(json.dumps({"variants": len(records)}))


if __name__ == "__main__":
    main()
