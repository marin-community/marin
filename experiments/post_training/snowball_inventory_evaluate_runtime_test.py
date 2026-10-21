# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from rolloutengine.models import RolloutLimits, TaskRolloutSpec
from taskcompendium.models import (
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.runtime.resources import inline_resource


@pytest.mark.manual
def test_reference_panel_preparation_changes_only_outer_deadlines_and_refuses_overwrite(tmp_path):
    from experiments.post_training.snowball_inventory_evaluate_runtime import (  # noqa: PLC0415 - optional frozen runtime
        prepare_reference_panel,
        require_reference_limits,
    )

    rows = []
    for index, context in enumerate(
        [
            ConversationInput(events=(TextMessage(role="user", content="Complete the structure"),)),
            ConversationInput(
                events=(
                    TextMessage(role="user", content="Complete the structure"),
                    TextMessage(role="assistant", content='{"x":'),
                ),
            ),
        ]
    ):
        task = TaskSpec(
            id=f"paired-{index}",
            context=context,
            environment_requirements=EnvironmentRequirements(),
            answer_type="text",
            answer_format={"kind": "plain_text"},
            source=Source(dataset="original", revision="frozen", row=str(index), importer_revision="1"),
            grader=VerifyitGrader(
                mode="structure_closure",
                parameters={"prefix": '{"x":', "expected": '{"x":1}', "format": "json"},
            ),
            resources={"verifier": (inline_resource("private-reference.json", b'{"hidden":"unaltered"}'),)},
        )
        spec = TaskRolloutSpec(
            task=task,
            harness=(
                "prefill"
                if index
                else 'shellbox:{"command_timeout":null,"shell_tool":{"name":"shell","command_parameter":"command"}}'
            ),
            task_machine=None,
            verifier_machine=None,
            limits=RolloutLimits(
                max_turns=1,
                max_total_tokens=None,
                model_turn_timeout=180,
                tool_turn_timeout=30,
                total_turn_timeout=240,
                attempt_timeout=300,
                verifier_timeout=900,
                cleanup_timeout=30,
            ),
        )
        rows.append(
            {
                "task_id": task.id,
                "task_rollout_spec": spec.model_dump_json(),
                "contract_id": semantic_digest(task, True),
                "public_content_id": semantic_digest(task, False),
                "source_contract_id": f"original-contract-{index}",
                "source_public_content_id": f"original-public-{index}",
                "evaluation_family": "structural",
                "lowered_task_spec": "byte-exact original source payload",
            }
        )
    source, derived = tmp_path / "original.parquet", tmp_path / "derived.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    original_bytes = source.read_bytes()
    report = prepare_reference_panel(
        source,
        derived,
        source_sha256=hashlib.sha256(original_bytes).hexdigest(),
        family_counts={"structural": 2},
    )
    written = pq.read_table(derived).to_pylist()
    for before, after in zip(rows, written, strict=True):
        original = TaskRolloutSpec.model_validate_json(before["task_rollout_spec"])
        updated = TaskRolloutSpec.model_validate_json(after["task_rollout_spec"])
        assert updated.task == original.task
        assert updated.limits.tool_turn_timeout == original.limits.tool_turn_timeout
        assert updated.limits.verifier_timeout == original.limits.verifier_timeout
        assert updated.limits.cleanup_timeout == original.limits.cleanup_timeout
        assert updated.limits.model_turn_timeout == 1800
        assert updated.limits.total_turn_timeout == 2400
        assert updated.limits.attempt_timeout == 3600
        require_reference_limits(updated)
        with pytest.raises(ValueError, match="frozen panel"):
            require_reference_limits(original)
        assert {key: value for key, value in after.items() if key != "task_rollout_spec"} == {
            key: value for key, value in before.items() if key != "task_rollout_spec"
        }
    assert source.read_bytes() == original_bytes
    assert report["row_count"] == 2
    assert [row["task_id"] for row in report["runtime_mappings"]] == ["paired-0", "paired-1"]
    with pytest.raises(FileExistsError):
        prepare_reference_panel(
            source,
            derived,
            source_sha256=report["source_sha256"],
            family_counts={"structural": 2},
        )


@pytest.mark.manual
@pytest.mark.parametrize("override_template", [False, True])
def test_cpu_preflight_tokenizer_loads_without_staged_model(tmp_path, override_template):
    # Run this optional-dependency regression in the frozen MarinSkyRL CPU environment.
    from omegaconf import OmegaConf  # noqa: PLC0415 - optional frozen runtime dependency
    from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: PLC0415 - optional runtime dependencies
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    from experiments.post_training.snowball_inventory_probe import preflight_tokenizer  # noqa: PLC0415

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    source = tmp_path / "pinned-tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(source)
    staged = tmp_path / "not-yet-staged-model"
    template = tmp_path / "serving-template.jinja"
    if override_template:
        template.write_text("{{ messages[-1]['content'] }}")
    config = OmegaConf.create(
        {
            "inputs": {"model": {"tokenizer_uri": str(source), "tokenizer_revision": "a" * 40}},
            "skyrl": {
                "generator": {
                    "chat_template": {"source": "file", "name_or_path": str(template)} if override_template else None
                },
                "trainer": {
                    "disable_fast_tokenizer": False,
                    "policy": {"model": {"tokenizer_path": str(staged), "tokenizer_revision": "a" * 40}},
                },
            },
        }
    )
    original = OmegaConf.to_container(config, resolve=True)

    tokenizer = preflight_tokenizer(config)

    assert tokenizer.encode("hello world", add_special_tokens=False) == [1, 2]
    if override_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello world"}],
            tokenize=True,
            return_dict=False,
        )
        assert rendered == [1, 2]
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == 3
    assert not staged.exists()
    assert OmegaConf.to_container(config, resolve=True) == original


@pytest.mark.manual
def test_probe_launch_binds_exact_bytes_and_rejects_other_runtime(tmp_path):
    from cloud.iris.runtime_bundle import resolve_launcher_source  # noqa: PLC0415

    from experiments.post_training.snowball_inventory_probe import ProbeReport, validated_probe_config  # noqa: PLC0415

    digest = resolve_launcher_source().digest
    document: dict[str, Any] = {
        "schema_version": 1,
        "run": {"id": "probe-unit", "attempt_id": "one", "export_hf": False},
        "runtime": {"source_digest": digest, "profile": "megatron"},
        "iris": {
            "cluster": "unit",
            "cluster_config": "unit.yaml",
            "job_name": "probe-unit",
            "allocation": {
                "num_nodes": 1,
                "gpus_per_node": 1,
                "gpu_variant": "H100",
                "cpu": 4,
                "memory": "32GB",
                "disk": "32GB",
            },
        },
        "ray": {"spill_dir": "/tmp/unit-spill", "rendezvous_dir": "s3://unit/rendezvous", "log_dir": "s3://unit/logs"},
        "artifacts": {
            "checkpoint_root": "s3://unit/checkpoints",
            "export_root": "s3://unit/exports",
            "attempts_root": "s3://unit/attempts",
            "resolved_config_uri": "s3://unit/config.yaml",
            "terminal_manifest_uri": "s3://unit/terminal.json",
        },
        "inputs": {
            "model": {
                "uri": "unit/model",
                "identity": "unit-model",
                "local_path": "/tmp/unit-model",
                "tokenizer_uri": "unit/model",
                "tokenizer_revision": "a" * 40,
            }
        },
        "skyrl": {
            "entrypoint": "standard",
            "context_budget": {"request_window_tokens": 1024, "max_new_tokens_per_turn": 256, "max_turns": 1},
            "model_num_attention_heads": 8,
            "trainer": {
                "strategy": "megatron",
                "algorithm": {"use_kl_loss": False},
                "placement": {"colocate_all": True, "policy_num_nodes": 1, "policy_num_gpus_per_node": 1},
                "train_batch_size": 1,
                "policy_mini_batch_size": 1,
                "micro_train_batch_size_per_gpu": 1,
            },
            "generator": {
                "backend": "vllm",
                "run_engines_locally": True,
                "num_inference_engines": 1,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": 1,
            },
        },
    }
    source = tmp_path / "launch.yaml"
    raw = "# exact submitted bytes\n" + yaml.safe_dump(document)
    source.write_text(raw)
    report = ProbeReport()

    config = validated_probe_config(source, report)

    assert config.runtime.source_digest == report.runtime_source_digest == digest
    assert report.launch_config_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    document["runtime"]["source_digest"] = "0" * 64
    source.write_text(yaml.safe_dump(document))
    rejected = ProbeReport()
    with pytest.raises(ValueError, match="runtime identity"):
        validated_probe_config(source, rejected)
    assert rejected.runtime_source_digest == digest
    assert rejected.launch_config_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.manual
def test_initializer_admission_rejects_wrong_model_or_revision(tmp_path):
    from marinskyrl.model_manifest import (  # noqa: PLC0415 - optional frozen runtime
        MODEL_MANIFEST_FILENAME,
        ModelManifestFile,
        build_model_manifest,
    )

    from experiments.post_training.snowball_inventory_probe import verify_initializer_model  # noqa: PLC0415

    # A real on-disk, internally valid manifest represents a mirrored model. No
    # model weights or serving calls are needed to test this provenance gate.
    manifest = build_model_manifest(
        tuple(
            ModelManifestFile(path=name, size=1, sha256="a" * 64)
            for name in ("config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors")
        ),
        model_id="example/cold-sft",
        revision="b" * 40,
        tokenizer_mode="policy",
    )
    marker = tmp_path / MODEL_MANIFEST_FILENAME
    marker.write_text(manifest.model_dump_json())
    original = marker.read_bytes()

    accepted = verify_initializer_model(str(tmp_path), "example/cold-sft", "b" * 40)
    assert accepted.identity == manifest.identity
    for repository, revision in (("example/dr-doom", "b" * 40), ("example/cold-sft", "c" * 40)):
        with pytest.raises(ValueError, match="selected initializer"):
            verify_initializer_model(str(tmp_path), repository, revision)
    assert marker.read_bytes() == original


@pytest.mark.manual
def test_saved_answer_scoring_preserves_original_failures_and_never_regrades_scored_rows(tmp_path, monkeypatch):
    from marinskyrl.saved_answers import admit_answer, save_answer  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415 - optional frozen runtime dependency
    from rolloutengine.harness.prefill import prefill_prompt  # noqa: PLC0415
    from rolloutengine.models import ModelRequest, ModelTurn  # noqa: PLC0415
    from taskcompendium.grader import verifyit_package  # noqa: PLC0415
    from verifyit.closure_scoring import ClosureFormat  # noqa: PLC0415
    from verifyit.spec import StructureClosureSpec  # noqa: PLC0415

    from experiments.post_training import snowball_inventory_evaluate_runtime as worker  # noqa: PLC0415

    specs = []
    for index in range(2):
        task = TaskSpec(
            id=f"paired-{index}",
            context=ConversationInput(events=(TextMessage(role="assistant", content='{"value":'),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type="text",
            answer_format={"kind": "plain_text"},
            grader=verifyit_package(StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON)).grader,
            source=Source(dataset="original", revision="frozen", row=str(index), importer_revision="1"),
        )
        specs.append(
            TaskRolloutSpec(
                task=task,
                harness="prefill",
                task_machine=None,
                verifier_machine=None,
                limits=RolloutLimits(
                    max_turns=1,
                    max_total_tokens=None,
                    model_turn_timeout=30,
                    tool_turn_timeout=30,
                    total_turn_timeout=30,
                    attempt_timeout=60,
                    verifier_timeout=10,
                    cleanup_timeout=5,
                ),
            )
        )
    panel = tmp_path / "panel"
    panel.mkdir()
    source = panel / "validation.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"task_id": spec.task.id, "task_rollout_spec": spec.model_dump_json()} for spec in specs]),
        source,
    )
    export = tmp_path / "exports"
    dumps = export / "dumped_evals" / "eval_only"
    dumps.mkdir(parents=True)
    rows = [
        {"uid": "0", "stop_reason": "stop", "verification": {"status": "verified", "score": 0.0}},
        {
            "uid": "1",
            "stop_reason": "error",
            "exception_type": "VerifierRuntimeError",
            "verification": {"status": "error", "reason": "original grader failed"},
        },
    ]
    original_dump = dumps / "dataset.jsonl"
    original_dump.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original_bytes = original_dump.read_bytes()
    saved = str(export / "saved_answers")
    messages = ({"role": "assistant", "content": '{"value":'},)
    turn = ModelTurn({"role": "assistant", "content": "12}"}, (1,), (2,), None, "stop", text="12}")
    save_answer(saved, specs[1], "1", 0, ModelRequest(messages, dict(prefill_prompt(specs[1].task).options)), turn)
    admit_answer(saved, "1", 0, (*messages, turn.message), "grade", "RuntimeError", "unavailable")
    config = OmegaConf.create(
        {
            "artifacts": {
                "export_root": str(export),
                "terminal_manifest_uri": str(tmp_path / "score" / "terminal.json"),
            },
            "inputs": {"validation_data": [{"uri": str(panel), "relative_path": "validation.parquet"}]},
            "skyrl": {"environment": {"task_sessions": {}}, "trajectory_runner": {"eval_saved_answers_root": saved}},
        }
    )
    monkeypatch.setattr(worker, "load_document", lambda _: config)
    request = {
        "launch_config_yaml": "test",
        "dataset_uids": {"0": "paired-0", "1": "paired-1"},
        "panel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_path": str(tmp_path / "score"),
        "per_answer_timeout": 2,
        "total_timeout": 10,
    }
    recovered = asyncio.run(worker.recover_results(request))
    assert [row["score"] for row in recovered["outcomes"]] == [0.0, 1.0]
    assert [row["task_id"] for row in recovered["recovery"]["receipts"]] == ["paired-1"]
    assert recovered["recovery"]["regenerated_completions"] == 0
    original = json.loads((tmp_path / "score" / "original-outcomes.json").read_text())
    assert original["outcomes"][1]["score"] is None
    assert original["outcomes"][1]["failure_type"] == "VerifierRuntimeError"
    assert original_dump.read_bytes() == original_bytes
    assert asyncio.run(worker.recover_results(request)) == recovered
