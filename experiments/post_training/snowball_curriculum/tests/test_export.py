# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted exports keep source identity and use the trainer's real harness settings."""

import hashlib
import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.assistant_prefix import AssistantPrefixHarness
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import ReplayModel, arithmetic_task, machine_runtime, rollout_limits
from taskcompendium.grader import verifyit_package
from taskcompendium.models import ConversationInput, EnvironmentRequirements, ScriptGrader, TextMessage
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.runtime.resources import inline_resource
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from verifyit.spec import JudgeSpec

from experiments.post_training.snowball_curriculum.bundle import file_sha256, verify_manifest, write_manifest, write_pool
from experiments.post_training.snowball_curriculum.native import native_closure_task
from experiments.post_training.snowball_curriculum.tests.closure_rows import closure_row

runtime = pytest.importorskip("skyrl_gym.rollout_specs")
from experiments.post_training.snowball_curriculum.export import (  # noqa: E402
    Deployment,
    DerivedDeployment,
    derive_deployment,
    derived_record,
    export_pool,
    rollout_record,
)
from experiments.post_training.snowball_inventory_panel import FAMILY_COUNTS  # noqa: E402
from experiments.post_training.snowball_inventory_validation import GRADER_IMAGE, freeze_panel  # noqa: E402


def source_pin(source):
    return {
        "source_uri": str(source),
        "source_manifest_sha256": file_sha256(StoragePath(str(source.parent / "artifact-manifest.json"))),
    }


def deployment():
    return Deployment(
        limits=rollout_limits(max_turns=1, tool_turn_timeout=35),
        command_timeout=30,
        shell_tool={"name": "bash", "command_parameter": "script"},
        task_machine=None,
        verifier_machine=None,
    )


def mixed_rows():
    task = arithmetic_task(context=ConversationInput(events=(TextMessage(role="assistant", content="Previous answer"),)))
    ordinary = {
        "source_id": "ordinary",
        "task_id": task.id,
        "task_json": task.model_dump_json(),
        "contract_id": semantic_digest(task, True),
        "public_content_id": semantic_digest(task, False),
        "contract_kind": "task_spec",
        "difficulty": "unknown",
        "family": "reasoning",
        "inclusion_probability": 1.0,
        "source_contract_id": "original-session-contract",
        "source_public_content_id": "original-session-public",
    }
    imported = native_closure_task(**closure_row("closure"))
    native = {
        "source_id": imported.source_id,
        "task_id": imported.task_id,
        "contract_id": imported.contract_id,
        "public_content_id": imported.public_content_id,
        "contract_kind": imported.contract_kind,
        "lowered_task_spec": imported.lowered_task_spec,
        "difficulty": "unknown",
        "family": "structural",
        "inclusion_probability": 1.0,
        "native_format": imported.format,
        "teacher_route": imported.teacher_route,
    }
    return ordinary, native


@pytest.mark.asyncio
async def test_mixed_export_preserves_payload_lineage_and_runs_explicit_harnesses(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = mixed_rows()
    write_pool(rows, StoragePath(str(source)))
    write_manifest(StoragePath(str(source)))
    json_rows = [json.loads(line) for line in (source / "tasks.jsonl").read_text().splitlines()]
    assert pq.read_table(source / "train.parquet").to_pylist() == json_rows
    assert json_rows[1]["native_format"] == "json"
    digest = hashlib.sha256((source / "train.parquet").read_bytes()).hexdigest()
    output = tmp_path / "exported"
    report = export_pool(source / "train.parquet", digest, output, deployment(), **source_pin(source / "train.parquet"))
    exported = pq.read_table(output / "train.parquet").to_pylist()
    audits = [json.loads(line) for line in (output / "deployment-contract-mapping.jsonl").read_text().splitlines()]
    assert report["source_uri"] == str(source / "train.parquet")
    assert report["source_artifact_manifest_sha256"] == source_pin(source / "train.parquet")["source_manifest_sha256"]
    assert report["exported_rows"] == 2 and report["filtered_rows"] == 0
    assert exported[0]["source_contract_id"] == "original-session-contract"
    ordinary, native = [TaskRolloutSpec.model_validate_json(row["task_rollout_spec"]) for row in exported]
    assert ordinary.task.model_dump_json() == rows[0]["task_json"]
    assert runtime.shell_harness(ordinary.harness).command_timeout == 30
    assert runtime.shell_harness(ordinary.harness).shell_tool.name == "bash"
    assert native.harness == "assistant_prefix"
    assert native.task.context.events[-1].content == '{"a":'
    original_native = json.loads(rows[1]["lowered_task_spec"])["task"]
    assert native.task.resources.model_dump(mode="json") == original_native["resources"] | {
        "all": [],
        "worker": [],
        "oracle": [],
    }
    assert audits[1]["task_semantic_changes"] == ["context", "grader"]
    assert audits[1]["historical_equivalence_claimed"] is False
    assert audits[1]["source_payload_sha256"] == hashlib.sha256(rows[1]["lowered_task_spec"].encode()).hexdigest()
    model = ReplayModel([{"role": "assistant", "content": "1} trailing"}])
    result = await RolloutEngine({}, {"assistant_prefix": AssistantPrefixHarness()}).run_task(native, model.complete)
    assert result.grade.reward == 1
    assert model.requests[0].prompt_token_ids is None
    assert model.requests[0].options["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False}
    ordinary_model = ReplayModel([{"role": "assistant", "content": "12"}])
    result = await RolloutEngine({}, {ordinary.harness: runtime.shell_harness(ordinary.harness)}).run_task(
        ordinary, ordinary_model.complete
    )
    assert result.grade.reward == 1
    assert "continue_final_message" not in ordinary_model.requests[0].options.get("extra_body", {})
    manifest = json.loads((output / "artifact-manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((output / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    with pytest.raises(FileExistsError):
        export_pool(source / "train.parquet", digest, output, deployment(), **source_pin(source / "train.parquet"))


def test_deadline_derivation_keeps_task_bytes_and_checks_encoded_command_timeout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_pool(mixed_rows(), StoragePath(str(source)))
    write_manifest(StoragePath(str(source)))
    source_path = source / "train.parquet"
    original = tmp_path / "original"
    export_pool(
        source_path,
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
        original,
        deployment(),
        **source_pin(source / "train.parquet"),
    )
    deployed = original / "train.parquet"
    change = DerivedDeployment(
        packages_locks={},
        environment_provenance_sha256="",
        deadlines={"attempt_timeout": 100},
        verifier_startup_timeout=None,
    )
    target = tmp_path / "derived"
    derive_deployment(
        deployed, hashlib.sha256(deployed.read_bytes()).hexdigest(), target, change, **source_pin(deployed)
    )
    for before, after in zip(
        pq.read_table(deployed).to_pylist(), pq.read_table(target / "train.parquet").to_pylist(), strict=True
    ):
        expected = json.loads(before["task_rollout_spec"])
        expected["limits"]["attempt_timeout"] = 100
        assert json.loads(after["task_rollout_spec"]) == expected
        assert {key: value for key, value in before.items() if key != "task_rollout_spec"} == {
            key: value for key, value in after.items() if key != "task_rollout_spec"
        }
    invalid = change.model_copy(update={"deadlines": {"tool_turn_timeout": 30}})
    with pytest.raises(ValueError, match="command timeout"):
        derive_deployment(
            deployed,
            hashlib.sha256(deployed.read_bytes()).hexdigest(),
            tmp_path / "invalid",
            invalid,
            **source_pin(deployed),
        )
    assert not (tmp_path / "invalid" / "artifact-manifest.json").exists()
    with pytest.raises(ValueError, match="immutable pin"):
        export_pool(source_path, "0" * 64, tmp_path / "corrupt", deployment(), **source_pin(source / "train.parquet"))
    assert not (tmp_path / "corrupt").exists()


def test_dependency_derivation_keeps_private_resources_and_original_identity():
    source_row = mixed_rows()[0]
    task = arithmetic_task(
        grader=ScriptGrader(
            argv=("python", "/tests/grade.py"),
            environment=EnvironmentRequirements(
                packages_lock="original/requirements.lock", command_semantics="linux_process"
            ),
        )
    )
    source_row.update(
        task_json=task.model_dump_json(),
        contract_id=semantic_digest(task, True),
        public_content_id=semantic_digest(task, False),
    )
    configured = deployment().model_copy(update={"verifier_machine": machine_runtime(startup_timeout=60)})
    original, _ = rollout_record(source_row, configured)
    derived, audit = derived_record(
        original,
        DerivedDeployment(
            packages_locks={"original/requirements.lock": "derived/requirements.lock"},
            environment_provenance_sha256="a" * 64,
            deadlines={"attempt_timeout": 200},
            verifier_startup_timeout=120,
        ),
    )
    before = TaskRolloutSpec.model_validate_json(original["task_rollout_spec"])
    after = TaskRolloutSpec.model_validate_json(derived["task_rollout_spec"])
    expected = before.model_dump(mode="json")
    expected["task"]["grader"]["environment"]["packages_lock"] = "derived/requirements.lock"
    expected["limits"]["attempt_timeout"] = 200
    expected["verifier_machine"]["startup_timeout"] = 120
    assert after.model_dump(mode="json") == expected
    assert derived["source_contract_id"] == original["source_contract_id"]
    assert derived["contract_id"] != original["contract_id"]
    assert derived["public_content_id"] == original["public_content_id"]
    assert audit["task_semantic_changes"] == ["grader.environment.packages_lock"]
    assert audit["input_runtime_sha256"] != audit["runtime_sha256"]

    with pytest.raises(ValueError, match="logical task ID"):
        derived_record(
            {**original, "task_id": "wrong-metadata-id"},
            DerivedDeployment(
                packages_locks={},
                environment_provenance_sha256="",
                deadlines={},
                verifier_startup_timeout=None,
            ),
        )


def test_dependency_derivation_replaces_sandbox_lock_and_preserves_host_judge_metadata(tmp_path):
    environment = EnvironmentRequirements(packages_lock="original/requirements.lock", command_semantics="linux_process")
    graders = {
        "script": ScriptGrader(argv=("python", "/tests/grade.py"), environment=environment),
        "judge": verifyit_package(JudgeSpec(references=("answer",)), environment=environment).grader,
    }
    originals = []
    for identifier, grader in graders.items():
        task = arithmetic_task(id=identifier, grader=grader)
        row = {
            **mixed_rows()[0],
            "task_id": task.id,
            "task_json": task.model_dump_json(),
            "contract_id": semantic_digest(task, True),
            "public_content_id": semantic_digest(task, False),
        }
        configured = deployment().model_copy(update={"verifier_machine": machine_runtime(startup_timeout=60)})
        original, _ = rollout_record(row, configured)
        originals.append(original)
    source = tmp_path / "original"
    source.mkdir()
    path = source / "train.parquet"
    pq.write_table(pa.Table.from_pylist(originals), path)
    write_manifest(StoragePath(str(source)))
    target = tmp_path / "derived"
    provenance = "a" * 64
    derive_deployment(
        path,
        file_sha256(StoragePath(str(path))),
        target,
        DerivedDeployment(
            packages_locks={"original/requirements.lock": "derived/requirements.lock"},
            environment_provenance_sha256=provenance,
            deadlines={},
            verifier_startup_timeout=None,
        ),
        **source_pin(path),
    )
    rows = pq.read_table(target / "train.parquet").to_pylist()
    audits = [json.loads(line) for line in (target / "deployment-contract-mapping.jsonl").read_text().splitlines()]
    script = TaskRolloutSpec.model_validate_json(rows[0]["task_rollout_spec"])
    assert script.task.grader.environment.packages_lock == "derived/requirements.lock"
    assert rows[0]["contract_id"] != originals[0]["contract_id"]
    assert rows[0]["public_content_id"] == originals[0]["public_content_id"]
    assert rows[1] == originals[1]
    assert audits[0]["task_semantic_changes"] == ["grader.environment.packages_lock"]
    assert audits[1]["task_semantic_changes"] == []
    persisted_provenance = json.loads((target / "provenance.json").read_text())
    assert persisted_provenance["configuration"]["environment_provenance_sha256"] == provenance
    verify_manifest(StoragePath(str(target)), file_sha256(StoragePath(str(target / "artifact-manifest.json"))))


def test_frozen_panel_manifest_supports_validation_derivation_and_reload(tmp_path):
    rows = []
    families = {
        "calendar": "ultra_sft_step3200_calendar_v2",
        "competitive_coding": "ultra_sft_step3200_comp_coding",
        "reasoning_gym": "ultra_sft_step3200_reasoning_gym",
        "stem": "ultra_sft_step3200_stem_mcqa",
    }
    for family, count in FAMILY_COUNTS.items():
        for index in range(count):
            identifier = f"{family}-{index}"
            if family == "structural":
                document = json.loads(closure_row(identifier)["lowered_task_spec"])
                document["session"].update(rollout_limits(max_turns=1).model_dump(mode="json"))
            else:
                task = arithmetic_task(
                    id=identifier,
                    grader=ScriptGrader(
                        argv=("python3", "/tests/grade.py"),
                        environment=EnvironmentRequirements(
                            docker_image=GRADER_IMAGE, command_semantics="linux_process"
                        ),
                    ),
                    resources={
                        "verifier": (
                            inline_resource(
                                "config.json", json.dumps({"contract": {"dataset": families[family]}}).encode()
                            ),
                        )
                    },
                )
                document = {
                    "task": task.model_dump(mode="json"),
                    "runtime": {
                        "task_machine": None,
                        "verifier_machine": machine_runtime(backend="daytona").model_dump(mode="json"),
                    },
                    "session": {
                        **rollout_limits(max_turns=1, tool_turn_timeout=35).model_dump(mode="json"),
                        "task_session": "shellbox",
                        "shell_tool": {"name": "shell", "command_parameter": "command"},
                        "command_timeout": 30,
                    },
                }
            rows.append({"lowered_task_spec": json.dumps(document)})
    original = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), original)
    training = tmp_path / "training.parquet"
    identities = [f"training-{index}" for index in range(100000)]
    pq.write_table(
        pa.table(
            {
                column: identities
                for column in (
                    "task_id",
                    "contract_id",
                    "public_content_id",
                    "source_contract_id",
                    "source_public_content_id",
                )
            }
        ),
        training,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_directory)
    frozen = tmp_path / "frozen"
    freeze_panel(
        original,
        file_sha256(StoragePath(str(original))),
        training,
        file_sha256(StoragePath(str(training))),
        frozen,
        tokenizer_directory,
        1024,
    )
    source = frozen / "validation.parquet"
    pin = source_pin(source)
    verify_manifest(StoragePath(str(frozen)), pin["source_manifest_sha256"])
    change = DerivedDeployment(
        packages_locks={},
        environment_provenance_sha256="",
        deadlines={"attempt_timeout": 120},
        verifier_startup_timeout=None,
    )
    configuration = tmp_path / "derivation.json"
    configuration.write_text(change.model_dump_json())
    derived = tmp_path / "derived"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.post_training.snowball_curriculum.export",
            "derive",
            "--source",
            str(source),
            "--source-sha256",
            file_sha256(StoragePath(str(source))),
            "--source-uri",
            str(source),
            "--source-manifest-sha256",
            pin["source_manifest_sha256"],
            "--configuration",
            str(configuration),
            "--output",
            str(derived),
            "--output-filename",
            "validation.parquet",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    verify_manifest(StoragePath(str(derived)), file_sha256(StoragePath(str(derived / "artifact-manifest.json"))))
    for before, after in zip(
        pq.read_table(source).to_pylist(), pq.read_table(derived / "validation.parquet").to_pylist(), strict=True
    ):
        assert {key: value for key, value in after.items() if key != "task_rollout_spec"} == {
            key: value for key, value in before.items() if key != "task_rollout_spec"
        }
        old = TaskRolloutSpec.model_validate_json(before["task_rollout_spec"])
        new = TaskRolloutSpec.model_validate_json(after["task_rollout_spec"])
        assert new == old.model_copy(update={"limits": old.limits.model_copy(update={"attempt_timeout": 120})})
