# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json

import pytest
import yaml
from fray.iris_backend import convert_constraints
from marin.execution.lazy import materialized_config
from marin.rl.curriculum import validation_payload
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ExecutionRequirements

from experiments.post_training.long_loop import campaign
from experiments.post_training.long_loop.campaign import REFERENCE, campaign_round
from experiments.post_training.long_loop.campaign_data import CampaignMix, load_mix, partition_mix, retained_task
from experiments.post_training.long_loop.campaign_runtime import bind_runtime
from experiments.post_training.long_loop.execution_policy import RlStopping, rl_training_callbacks
from experiments.post_training.long_loop.round_runner import adopted_round_rl, load_round
from experiments.post_training.task_curation.curriculum_manifest import curriculum_from_manifest


def test_equal_mix_holdout_preserves_task_bytes_and_pins():
    pins = {
        label: {
            "name": label,
            "version": "3",
            "repo": f"tests/{label}",
            "revision": "a" * 40,
            "url": "https://example.org/pin",
            "recipe": "atlas:fixture/retained@3",
        }
        for label in ("first", "second")
    }
    mix = CampaignMix("v2-fixture", pins, "b" * 64, "c" * 64)
    rows = {
        label: [
            {
                "task_spec": json.dumps({"id": f"{label}-{i}", "grader": {"contract": "fixed"}}),
                "runtime": '{"task_machine":null}',
                "task_sha256": str(i),
            }
            for i in range(180)
        ]
        for label in pins
    }
    prepared = partition_mix(mix, rows, "2026.11.26.501")
    curriculum, train, held = prepared.curriculum, prepared.training, prepared.validation
    assert len(held) == 200
    assert {label: len(records) for label, records in train.items()} == {"first": 80, "second": 80}
    held_ids = {json.loads(row["task_spec"])["id"] for row in held}
    train_ids = {json.loads(row["task_spec"])["id"] for records in train.values() for row in records}
    assert not held_ids & train_ids
    assert len(held_ids | train_ids) == 360
    original = {row["task_spec"]: row for records in rows.values() for row in records}
    for row in [*held, *(row for records in train.values() for row in records)]:
        assert {key: value for key, value in row.items() if key != "teacher_route"} == original[row["task_spec"]]
    repeated = partition_mix(mix, {key: list(reversed(value)) for key, value in rows.items()}, "2026.11.26.501")
    assert repeated.curriculum.identity == curriculum.identity
    definition = dataclasses.asdict(curriculum)
    definition.pop("fingerprint")
    assert (
        curriculum_from_manifest(
            {
                "curriculum": curriculum.identity,
                "curriculum_definition": definition,
                "validation": validation_payload(curriculum.validation),
            }
        )
        == curriculum
    )


def test_cross_source_holdout_leakage_is_rejected():
    pin = {
        "name": "fixture",
        "version": "1",
        "repo": "tests/pin",
        "revision": "a" * 40,
        "url": "https://example.org/pin",
        "recipe": "atlas:fixture/retained@3",
    }
    mix = CampaignMix("v2-leak", {"first": pin, "second": pin}, "b" * 64, "c" * 64)
    rows = [{"task_spec": json.dumps({"id": f"shared-{i}"})} for i in range(180)]
    with pytest.raises(ValueError, match="Cross-source holdout leakage"):
        partition_mix(mix, {"first": rows, "second": rows}, "2026.11.26.501")


def test_retained_checksum_uses_published_json_bytes_without_reserializing():
    task = arithmetic_task()
    encoded = json.dumps(json.loads(task.model_dump_json()), indent=2, ensure_ascii=False)
    row = {"task_spec": encoded, "task_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    assert retained_task(row, "fixture") == task
    # Even semantically equivalent JSON must retain the publication's literal bytes.
    with pytest.raises(ValueError, match="Task checksum changed"):
        retained_task({**row, "task_spec": encoded + "\n"}, "fixture")


def test_campaign_preserves_131_controls_and_disables_learning_clock():
    mix = load_mix("v2-mix-032")
    campaign = campaign_round(mix, "2026.11.26.501", max_steps=50)
    reference = load_round(REFERENCE)
    before, after = dataclasses.asdict(reference), dataclasses.asdict(campaign)
    for field in ("name", "slug", "version", "evaluation_checkpoints"):
        after[field] = before[field]
    for field in ("max_steps", "stopping", "native_checkpoint_storage", "pools"):
        after["rl"][field] = before["rl"][field]
    after["rl"]["execution"]["coordinator_name"] = before["rl"]["execution"]["coordinator_name"]
    after["rl"]["execution"].pop("coordinator_secret_sources")
    after["rl"]["recipe"]["trainer"]["callbacks"] = before["rl"]["recipe"]["trainer"]["callbacks"]
    after["rl"]["recipe"]["trainer"]["rollout_buffer"]["source_routes"] = before["rl"]["recipe"]["trainer"][
        "rollout_buffer"
    ]["source_routes"]
    after["rl"]["recipe"]["environment"]["task_sessions"]["task_spec"]["harness_by_dataset"] = before["rl"]["recipe"][
        "environment"
    ]["task_sessions"]["task_spec"]["harness_by_dataset"]
    assert after == before
    callbacks = campaign.rl.recipe["trainer"]["callbacks"]
    assert all(callback["type"] != "wall_time_budget" for callback in callbacks)
    assert campaign.rl.max_steps == 50
    # Going through both callback-normalization layers must not reinsert the clock.
    assert rl_training_callbacks(callbacks, groups_per_step=64, stopping=campaign.rl.stopping) == callbacks
    assert rl_training_callbacks(callbacks, groups_per_step=64)[-1] == {
        "type": "wall_time_budget",
        "seconds": 43200,
        "drain_margin": 1800,
        "groups_per_step": 64,
    }
    assert campaign.rl.stopping is RlStopping.MAX_STEPS


def test_unpublished_source_does_not_fall_back_to_raw_converter(tmp_path):
    candidates = b"mix_id,k,sources\nfixture,1,tests:raw\n"
    (tmp_path / "mix_candidates_v2.csv").write_bytes(candidates)
    (tmp_path / "source_pins.json").write_text(
        json.dumps(
            {
                "catalog_load": {"load_id": "54405be9b4db4c2d9828549469abe759"},
                "candidate_sha256": {"v2": hashlib.sha256(candidates).hexdigest()},
                "sources": {"tests:raw": {"status": "Pending", "intended_use": "train", "files": ["raw.parquet"]}},
            }
        )
    )
    with pytest.raises(ValueError, match=r"Served R40 catalog cannot admit.*tests:raw"):
        load_mix("v2-fixture", tmp_path)


def test_native_round_render_stops_at_fifty_without_reinserting_clock(tmp_path, monkeypatch):
    mix = load_mix("v2-mix-032")
    spec = campaign_round(mix, "2026.11.26.501", max_steps=50)
    rows = {
        label: [{"task_spec": json.dumps({"id": f"{label}-{index}"})} for index in range(60)] for label in mix.sources
    }
    curriculum = partition_mix(mix, rows, spec.version).curriculum
    definition = dataclasses.asdict(curriculum)
    definition.pop("fingerprint")
    pools = tmp_path / "pools"
    pools.mkdir()
    manifest = {
        "curriculum": curriculum.identity,
        "curriculum_definition": definition,
        "validation": validation_payload(curriculum.validation),
        "composition": {label: 60 for label in mix.sources},
        "routes": {label: pin["route"] for label, pin in mix.sources.items()},
        "files": {"validation.parquet": "c" * 64},
    }
    (pools / "manifest.json").write_text(json.dumps(manifest))
    model = tmp_path / "model"
    model.mkdir()
    config = json.dumps(
        {"grugmoe_artifact_schema_version": 1, "num_hidden_layers": 36, "num_attention_heads": 32, "num_experts": 128}
    ).encode()
    export = b'{"step":2100}'
    (model / "config.json").write_bytes(config)
    (model / "export-manifest.json").write_bytes(export)
    spec = dataclasses.replace(
        spec,
        adopted_model=dataclasses.replace(
            spec.adopted_model,
            uri=str(model),
            config_sha256=hashlib.sha256(config).hexdigest(),
            export_manifest_sha256=hashlib.sha256(export).hexdigest(),
        ),
        rl=dataclasses.replace(spec.rl, pools={"uri": str(pools)}),
    )
    authored = tmp_path / "campaign.yaml"
    authored.write_text(yaml.safe_dump(json.loads(json.dumps(dataclasses.asdict(spec)))))
    spec = load_round(authored)
    step = adopted_round_rl(spec, verify_pool_at_runtime=True)
    native = yaml.safe_load(materialized_config(step, str(tmp_path / "outputs")).launch_config_yaml)
    assert native["skyrl"]["trainer"]["max_steps"] == 50
    assert native["skyrl"]["trainer"]["epochs"] == 50
    assert not any(callback["type"] == "wall_time_budget" for callback in native["skyrl"]["trainer"]["callbacks"])
    assert native["skyrl"]["generator"]["speculative_decoding"]["num_speculative_tokens"] == 4
    # Rendering must not open the intentionally absent parquet bodies on the Mac.
    assert not (pools / "validation.parquet").exists()
    dispatched = []
    real_remote = campaign.remote

    def capture_remote(*args, **kwargs):
        wrapped = real_remote(*args, **kwargs)
        dispatched.append(wrapped.resources)
        return wrapped

    monkeypatch.setattr(campaign, "REFERENCE", authored)
    monkeypatch.setattr(campaign, "remote", capture_remote)
    campaign.campaign_graph(mix, spec)
    # The production coordinator is already regional. A child cluster constraint
    # becomes a second federation request, which an 08a controller rejects.
    assert len(dispatched) == 2
    assert not any(
        constraint.key == "cluster" for resource in dispatched for constraint in convert_constraints(resource)
    )


def test_retained_runtime_lowering_preserves_grader_and_declared_resources():
    task = arithmetic_task()
    task = task.model_copy(
        update={
            "grader": task.grader.model_copy(
                update={
                    "environment": EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="s3://fixture/requirements.lock"
                    )
                }
            )
        }
    )
    row = {
        "task_spec": task.model_dump_json(),
        "runtime": json.dumps(
            {"task_machine": None, "verifier_machine": {"cpus": 3, "memory_mb": 6144, "storage_mb": 10240, "gpus": 0}}
        ),
    }
    defaults = REFERENCE.parents[4] / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
    environment = yaml.safe_load(defaults.read_text())["task_spec"]
    projected = bind_runtime(row, environment, "nonagentic")
    lowered = TaskRolloutSpec.model_validate_json(projected["task_rollout_spec"])
    assert lowered.task == task
    assert projected["task_spec"] == row["task_spec"]
    assert projected["runtime"] == row["runtime"]
    assert lowered.verifier_machine is not None
    assert lowered.verifier_machine.cpus == 3
    assert lowered.verifier_machine.memory_mb == 6144
    assert lowered.verifier_machine.backend == environment["package_verifier"]["backend"]


def test_null_catalog_sizes_keep_provider_defaults_and_declared_minima():
    task = arithmetic_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image="fixture@sha256:" + "a" * 64,
            ),
            "execution_requirements": ExecutionRequirements(
                cpus=4, memory_mb=8192, storage_mb=2048, agent_timeout_sec=600, verifier_timeout_sec=600
            ),
        }
    )
    row = {
        "task_spec": task.model_dump_json(),
        "runtime": json.dumps({"task_machine": {"cpus": None, "memory_mb": None, "storage_mb": None, "gpus": 0}}),
    }
    defaults = REFERENCE.parents[4] / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
    environment = yaml.safe_load(defaults.read_text())["task_spec"]
    environment["image_verifier"].update(cpus=2, memory_mb=4096, storage_mb=10240)
    projected = bind_runtime(row, environment, "nonagentic")
    lowered = TaskRolloutSpec.model_validate_json(projected["task_rollout_spec"])
    assert lowered.task == task
    assert projected["task_spec"] == row["task_spec"]
    assert lowered.task_machine.cpus == 4
    assert lowered.task_machine.memory_mb == 8192
    assert lowered.task_machine.storage_mb == 10240
    assert lowered.limits.total_turn_timeout == 600
