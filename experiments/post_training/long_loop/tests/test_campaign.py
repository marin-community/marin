# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import textwrap

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from fray.iris_backend import convert_constraints
from marin.execution.lazy import materialized_config
from marin.rl.curriculum import validation_payload
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from skyrl_train.dataset.harbor_import import harbor_trial_task, restore_harbor_trial_task
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ExecutionRequirements

from experiments.post_training.long_loop import campaign
from experiments.post_training.long_loop.campaign import (
    LOCAL_DISK_ROOT,
    LOCAL_REGISTRY_AUTH_SECRET,
    PER_TASK_IMAGE_MACHINE,
    REFERENCE,
    campaign_round,
)
from experiments.post_training.long_loop.campaign_data import (
    CampaignMix,
    load_mix,
    partition_mix,
    partition_parquets,
    retained_task,
    write_parquet,
)
from experiments.post_training.long_loop.campaign_runtime import bind_runtime, lower_parquet
from experiments.post_training.long_loop.execution_policy import RlStopping, rl_training_callbacks
from experiments.post_training.long_loop.round_runner import adopted_round_rl, load_round
from experiments.post_training.task_curation.curriculum_manifest import curriculum_from_manifest


def test_equal_mix_holdout_preserves_task_bytes_and_pins():
    pins = {
        label: {
            "catalog_status": "Available",
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
        "catalog_status": "Available",
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


def test_batched_pool_preparation_preserves_the_original_split_and_literal_rows(tmp_path):
    pins = {
        label: {
            "catalog_status": "Available",
            "name": label,
            "version": "3",
            "repo": f"tests/{label}",
            "revision": "a" * 40,
            "url": "https://example.org/pin",
            "recipe": "atlas:fixture/retained@3",
            "route": "nonagentic",
        }
        for label in ("first", "second")
    }
    mix = CampaignMix("v2-fixture", pins, "b" * 64, "c" * 64)
    defaults = REFERENCE.parents[4] / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
    environment = yaml.safe_load(defaults.read_text())["task_spec"]
    paths, rows = {}, {}
    for label in pins:
        rows[label] = []
        for index in range(180):
            task = arithmetic_task().model_copy(update={"id": f"{label}-{index}"})
            encoded = task.model_dump_json()
            rows[label].append(
                {
                    "task_spec": encoded,
                    "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    "runtime": "{}",
                    f"{label}_metadata": None if index < 64 else f"kept-{index}",
                }
            )
        original = tmp_path / f"{label}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[label]), original)
        paths[label] = tmp_path / f"{label}-lowered.parquet"
        lower_parquet(original, paths[label], environment)
        expected = [bind_runtime(row, environment) for row in rows[label]]
        assert pq.read_table(paths[label]).to_pylist() == expected
        rows[label] = expected
    reference = partition_mix(mix, rows, "2026.11.26.532")
    destination = tmp_path / "published"
    destination.mkdir()
    manifest = partition_parquets(mix, paths, "2026.11.26.532", StoragePath(str(destination)))
    assert manifest["curriculum"] == reference.curriculum.identity
    assert manifest["validation"] == validation_payload(reference.curriculum.validation)
    for label, name in manifest["training_files"].items():
        assert pq.read_table(destination / name).to_pylist() == reference.training[label]
    expected_held = {row["task_sha256"]: row for row in reference.validation}
    held = pq.read_table(destination / "validation.parquet").to_pylist()
    assert len(held) == 200
    for row in held:
        assert {key: value for key, value in row.items() if value is not None} == {
            key: value for key, value in expected_held[row["task_sha256"]].items() if value is not None
        }
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest


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
    for field in ("max_steps", "stopping", "native_checkpoint_storage", "pools", "role_plan", "megatron"):
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
    sessions, reference_sessions = (
        after["rl"]["recipe"]["environment"]["task_sessions"],
        before["rl"]["recipe"]["environment"]["task_sessions"],
    )
    assert sessions.pop("local") == {"disk_root": LOCAL_DISK_ROOT, "registry_auth_secret": LOCAL_REGISTRY_AUTH_SECRET}
    reference_sessions.pop("local")
    judge, reference_judge = sessions.pop("remote_judge"), reference_sessions.pop("remote_judge")
    assert {key: judge[key] for key in reference_judge} == reference_judge
    assert judge["substitute_models"]["glm-5.3"] == judge["model"]
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


@pytest.mark.parametrize("generation,candidate_id", [("v2", "fixture"), ("v3", "v3-fixture")])
def test_unpublished_source_does_not_fall_back_to_raw_converter(tmp_path, generation, candidate_id):
    candidates = f"mix_id,k,sources\n{candidate_id},1,tests:raw\n".encode()
    (tmp_path / f"mix_candidates_{generation}.csv").write_bytes(candidates)
    (tmp_path / "source_pins.json").write_text(
        json.dumps(
            {
                "catalog_load": {"load_id": "54405be9b4db4c2d9828549469abe759"},
                "candidate_sha256": {generation: hashlib.sha256(candidates).hexdigest()},
                "sources": {"tests:raw": {"status": "Pending", "intended_use": "train", "files": ["raw.parquet"]}},
            }
        )
    )
    with pytest.raises(ValueError, match=r"Served R40 catalog cannot admit.*tests:raw"):
        load_mix(f"{generation}-fixture", tmp_path)


def test_authorized_campaign_divergence_keeps_exact_retained_pins():
    mix = load_mix("v2-mix-033")
    assert {label: entry["status"] for label, entry in mix.catalog_divergence.items()} == {
        "TaskTrove:swesmith-verifier-repaired-v5": "Excluded",
        "TaskTrove:tasktrove-codeforces-rebuilt-v1": "Pending",
    }
    source = mix.sources["TaskTrove:swesmith-verifier-repaired-v5"]
    assert source["repo"] == "open-athena/campaign-release-swesmith-verifier-repaired-v5-0bfcd894-retained-20261116-v1"
    assert source["revision"] == "758b28bf4cc0d0e770329f2944c9451fd578566a"
    assert source["count"] == 7016
    assert mix.weights == {label: 1 / 7 for label in mix.sources}
    entry = mix.curriculum_catalog["TaskTrove:swesmith-verifier-repaired-v5"]
    assert (entry.pin.name, entry.pin.revision) == (source["repo"], source["revision"])


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
        {"grugmoe_artifact_schema_version": 1, "num_hidden_layers": 48, "num_attention_heads": 32, "num_experts": 128}
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
    # Reproduce the campaign reset through the native wire format: PP16 needs
    # 64 training GPUs while the five EP16 serving engines remain on80 GPUs.
    assert native["iris"]["allocation"]["num_nodes"] == 36
    assert native["iris"]["allocation"]["gpus_per_node"] == 4
    assert native["skyrl"]["trainer"]["placement"]["policy_num_nodes"] == 16
    geometry = native["skyrl"]["trainer"]["policy"]["megatron_config"]
    assert geometry["pipeline_model_parallel_size"] == 16
    assert geometry["context_parallel_size"] == 4
    assert geometry["tensor_model_parallel_size"] == 1
    assert geometry["expert_model_parallel_size"] == 4
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
    graph = campaign.campaign_graph(mix, spec)
    assert graph.plan["gpu_footprint"] == {"trainer": 64, "rollout": 80, "total": 144}
    # The production coordinator is already regional. A child cluster constraint
    # becomes a second federation request, which an 08a controller rejects.
    assert len(dispatched) == 2
    assert not any(
        constraint.key == "cluster" for resource in dispatched for constraint in convert_constraints(resource)
    )

    # Reproduce the regional CPU failure with the provider SDK absent. The pool
    # worker still needs a materialized, CPU-only graph payload before dispatch.
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import sys
        sys.modules['daytona'] = None
        sys.modules['daytona_api_client_async'] = None
        from marin.execution.lazy import materialized_config
        from experiments.post_training.long_loop import campaign
        from experiments.post_training.long_loop.campaign_data import load_mix
        campaign.REFERENCE = Path(sys.argv[1])
        mix = load_mix('v2-mix-032')
        spec = campaign.campaign_round(mix, '2026.11.26.501', max_steps=50)
        graph = campaign.campaign_graph(mix, spec)
        pool = next(dep for dep in graph.training.deps if dep.name.endswith('/pools'))
        config = materialized_config(pool, str(Path(sys.argv[1]).parent / 'outputs'))
        print(json.dumps({'arguments': sorted(config)}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(authored)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["arguments"] == ["launch_document", "mix", "output", "pool_version"]


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
    projected = bind_runtime(row, environment)
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
    projected = bind_runtime(row, environment)
    lowered = TaskRolloutSpec.model_validate_json(projected["task_rollout_spec"])
    assert lowered.task == task
    assert projected["task_spec"] == row["task_spec"]
    assert lowered.task_machine.cpus == 4
    assert lowered.task_machine.memory_mb == 8192
    assert lowered.task_machine.storage_mb == 10240
    assert lowered.limits.total_turn_timeout == 600


def test_mixed_validation_metadata_survives_parquet_publication(tmp_path):
    rows = [
        {"task_spec": '{"id": "first"}', "runtime": "{}", "teacher_route": "first", "first_metadata": "kept"},
        {"task_spec": '{"id":"second"}', "runtime": "{}", "teacher_route": "second", "second_metadata": "also kept"},
    ]
    digest = write_parquet(StoragePath(str(tmp_path)), "validation.parquet", rows)
    assert digest == hashlib.sha256((tmp_path / "validation.parquet").read_bytes()).hexdigest()
    actual = pq.read_table(tmp_path / "validation.parquet").to_pylist()
    assert actual == [
        {**rows[0], "second_metadata": None},
        {**rows[1], "first_metadata": None},
    ]


def test_original_harbor_package_survives_runtime_binding(tmp_path):
    package = tmp_path / "original"
    package.mkdir()
    (package / "instruction.md").write_text("Write the requested file.")
    (package / "task.toml").write_text('[environment]\ndocker_image = "fixture@sha256:' + "a" * 64 + '"\n')
    task = harbor_trial_task(package, source=arithmetic_task().source)
    row = {"task_spec": task.model_dump_json(), "runtime": "{}"}
    defaults = REFERENCE.parents[4] / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
    environment = yaml.safe_load(defaults.read_text())["task_spec"]
    bound = TaskRolloutSpec.model_validate_json(bind_runtime(row, environment)["task_rollout_spec"])
    assert bound.harness == "harbor_trial"
    assert bound.task == task
    restored = tmp_path / "restored"
    restore_harbor_trial_task(bound.task, restored)
    assert {p.name: p.read_bytes() for p in restored.iterdir()} == {p.name: p.read_bytes() for p in package.iterdir()}


def test_missing_cohort_binds_only_its_declared_release():
    mix = load_mix("v3-anchor-exp4-01")
    label = "AllenAI:Dolci-Think-SFT-Clean-v2/precise-if"
    assert mix.catalog_divergence[label] == {
        "status": "Missing",
        "repo": "open-athena/allenai-dolci-think-sft-clean-v2-precise-if-37ed6d9f-retained-20261125-b-dolci-v1",
        "revision": "3705ff47972eef44df33e4c160e7329b72af6021",
    }
    assert mix.sources[label]["count"] == 132151


def test_per_task_image_mix_places_actor_and_grader_on_local():
    mix = load_mix("v2-mix-011")
    assert "RST:laion/rts-1k-rootless-apptainer" in mix.sources
    task_spec = campaign_round(mix, "2026.11.27.990", max_steps=50).rl.recipe["environment"]["task_sessions"][
        "task_spec"
    ]
    for placement in ("task_machine_by_dataset", "verifier_machine_by_dataset"):
        assert task_spec[placement] == {"laion/rts-1k-rootless-apptainer": PER_TASK_IMAGE_MACHINE}
