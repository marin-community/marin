# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles
from marin.rl.curriculum import PoolRow, consumed_route_quotas, validation_payload
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.long_loop.archive import assert_best_rl_checkpoint
from experiments.post_training.long_loop.execution_policy import RL_TRAINING_BUDGET
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.openresearcher_service import OpenResearcherService, run_with_retrieval
from experiments.post_training.long_loop.rl_terminal_evaluation import FinalPolicy, ReusedEvaluation, run_distinct_final
from experiments.post_training.long_loop.round_runner import (
    DRAFTS,
    adopted_rl_evaluation,
    adopted_round_resolved_plan,
    adopted_round_rl,
    load_round,
    round_draft,
)
from experiments.post_training.long_loop.run_spec import FilePin
from experiments.post_training.task_curation.exp3_curriculum import exp3_rl_curriculum
from experiments.post_training.task_curation.exp4_curriculum import (
    SOURCE_ROUTES,
    exp4_rl_curriculum,
    exp4_validation_from_manifest,
    exp4_validation_split,
)
from experiments.post_training.task_curation.exp5_curriculum import TARGET_SHARES as EXP5_TARGET_SHARES
from experiments.post_training.task_curation.exp5_curriculum import exp5_rl_curriculum, exp5_validation_split


def _pools_rows() -> dict[str, tuple[PoolRow, ...]]:
    """Synthetic row identities over the eight Exp4 labels, enough for the 200-prompt draw."""
    counts = {
        "long-loop-smartie-swesmith": 80,
        "nemotron_if": 40,
        "open-thoughts/TaskTrove": 30,
        "Reasoning Gym:reasoning_gym": 30,
        "nvidia/OpenScience": 30,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": 20,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 20,
        "PRIME-RL/Eurus-2-RL-Data": 20,
    }
    return {
        label: tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(count))
        for label, count in counts.items()
    }


ROUND_DIR = Path(__file__).parents[1] / "rounds"
RL_ROUNDS = [
    path for path in sorted(ROUND_DIR.glob("*.yaml")) if "recipe" in yaml.safe_load(path.read_text()).get("rl", {})
]


@pytest.mark.parametrize("profile", ["one-rack"])
def test_adopted_rl_exports_selected_policy_before_frozen_evaluation(local_round, tmp_path):
    spec = dataclasses.replace(
        local_round,
        evaluation="frozen-micro210",
        evaluation_profile="mochi-gb200-cw-us-east-08a",
        rl=dataclasses.replace(
            local_round.rl,
            export_hf=True,
            retained_checkpoints=1,
            recipe=copy.deepcopy(local_round.rl.recipe),
        ),
    )
    spec.rl.recipe["trainer"]["hf_export_checkpoint"] = "best"
    training = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "rl"), str(tmp_path), deps=training.deps, runtime_args=training.runtime_args
    )
    config = training.build_config(context)
    launch = yaml.safe_load(config.launch_config_yaml)
    assert config.export_hf
    assert launch["skyrl"]["trainer"]["hf_export_checkpoint"] == "best"
    assert launch["artifacts"]["resume_checkpoint_count"] == 1
    evaluation = adopted_rl_evaluation(spec, training)
    assert training in graph_handles([evaluation])
    evaluation_context = StepContext.for_run(
        str(tmp_path / "eval"), str(tmp_path), deps=evaluation.deps, runtime_args=evaluation.runtime_args
    )
    run = evaluation.build_config(dataclasses.replace(evaluation_context, is_fingerprint=True))
    assert run.model_manifest_filename == ".marinskyrl-model-manifest.json"
    assert run.serving_profile.role_plan.num_nodes * run.serving_profile.role_plan.gpus_per_node == 16
    assert run.serving_profile.execution.target_cluster == "cw-us-east-08a"
    assert run.evaluation.record().policy_reference.revision == "2026.11.23-together"


@pytest.mark.parametrize("profile", ["one-rack"])
def test_final_evaluation_reuses_best_job_when_checkpoint_is_identical(local_round, tmp_path):
    spec = dataclasses.replace(
        local_round,
        evaluation="frozen-micro210",
        evaluation_profile="mochi-gb200-cw-us-east-08a",
        evaluation_checkpoints="best_and_final",
        rl=dataclasses.replace(local_round.rl, export_hf=True),
    )
    training = adopted_round_rl(spec)
    terminal = adopted_rl_evaluation(spec, training)
    stages = graph_handles([terminal])
    exported = next(step for step in stages if step.artifact_type is FinalPolicy)
    best = next(step for step in stages if step.name.endswith("/micro210"))
    assert exported.deps[0] == training
    assert exported.deps[1].deps == (best,)
    final = next(step for step in stages if step.name.endswith("/micro210-final"))
    context = StepContext.for_run(str(tmp_path / "final"), str(tmp_path), deps=final.deps)
    context._loaded[id(exported)] = FinalPolicy(
        uri="s3://marin-us-east-08a/model/policy",
        step=7,
        tokenizer_uri="tokenizer",
        tokenizer_revision="pinned",
        same_as_best=True,
    )
    result = SkyRLRun(
        hf_model_uri=None,
        global_step=None,
        tokenizer_uri=None,
        tokenizer_revision=None,
        checkpoint_root="unused",
        draft_checkpoint_root=None,
        terminal_manifest_uri="best/terminal.json",
        iris_job_id="/user/best-evaluation",
    )
    context._loaded[id(best)] = result
    config = final.build_config(context)
    assert isinstance(config, ReusedEvaluation)
    assert run_distinct_final(config) == result
    assert json.loads((tmp_path / "final" / "evaluation-reuse.json").read_text())["job_id"] == result.iris_job_id


@pytest.fixture
def round_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "output"))
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = tmp_path / "model"
    model.mkdir()
    manifest = json.dumps({"request": {"config": {"checkpoint": "s3://source/checkpoints/step-400407"}}}).encode()
    config = json.dumps(
        {"grugmoe_artifact_schema_version": 3, "num_hidden_layers": 96, "num_attention_heads": 96, "num_experts": 384}
    ).encode()
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_bytes(config)
    data = tmp_path / "data"
    data.mkdir()
    curriculum = exp4_rl_curriculum(exp4_validation_split(_pools_rows()).validation)
    rows = curriculum.validation.rows
    payload = {
        "size": curriculum.validation.size,
        "seed": curriculum.validation.seed,
        "rows": [{"label": row.label, "id": row.id, "sha256": row.sha256} for row in rows],
    }
    composition = Counter(row.label for row in rows)
    files = {"validation.parquet": hashlib.sha256(b"pinned validation fixture").hexdigest()}
    (data / "validation.parquet").write_bytes(b"pinned validation fixture")
    for label in sorted(composition):
        content = f"pinned training fixture for {label}".encode()
        (data / label).mkdir(parents=True)
        (data / label / "train.parquet").write_bytes(content)
        files[f"{label}/train.parquet"] = hashlib.sha256(content).hexdigest()
    manifest = {
        "curriculum": curriculum.identity,
        "validation": payload,
        "composition": dict(sorted(composition.items())),
        "routes": dict(sorted(SOURCE_ROUTES.items())),
        "files": files,
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    return model, data


@pytest.fixture
def local_round(round_inputs, profile):
    model, data = round_inputs
    spec = load_round(ROUND_DIR / f"mochi-{profile}.yaml")
    pins = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ("config.json", "export-manifest.json")
    }
    return dataclasses.replace(
        spec,
        adopted_model=dataclasses.replace(
            spec.adopted_model,
            uri=str(model),
            config_sha256=pins["config.json"],
            export_manifest_sha256=pins["export-manifest.json"],
        ),
        rl=dataclasses.replace(
            spec.rl,
            pools={"uri": str(data)},
            max_steps=15,
        ),
    )


@pytest.mark.parametrize("profile", ["one-rack"])
@pytest.mark.parametrize("round_path", RL_ROUNDS, ids=lambda path: path.stem)
def test_every_authored_rl_round_resolves_the_canonical_training_clock(local_round, tmp_path, round_path):
    authored = yaml.safe_load(round_path.read_text())["rl"]
    authored["recipe"]["trainer"]["rollout_buffer"]["source_routes"] = dict(SOURCE_ROUTES)
    spec = dataclasses.replace(
        local_round,
        cluster=authored["execution"]["cluster"],
        rl=dataclasses.replace(
            local_round.rl,
            recipe=authored["recipe"],
            role_plan=authored["role_plan"],
            megatron=authored["megatron"],
            execution=authored["execution"],
            gpu_variant=authored.get("gpu_variant", "GB200"),
        ),
    )
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    budgets = [
        callback for callback in launch["skyrl"]["trainer"]["callbacks"] if callback["type"] == "wall_time_budget"
    ]
    assert len(budgets) == 1
    assert {key: budgets[0][key] for key in RL_TRAINING_BUDGET} == RL_TRAINING_BUDGET
    assert budgets[0]["groups_per_step"] == spec.rl.role_plan["train_batch_size"]


@pytest.mark.parametrize("profile", ["one-rack"])
@pytest.mark.parametrize("field", ["seconds", "drain_margin"])
def test_round_refuses_training_clock_drift_before_allocation(local_round, tmp_path, field):
    document = dataclasses.asdict(local_round)
    budget = next(cb for cb in document["rl"]["recipe"]["trainer"]["callbacks"] if cb["type"] == "wall_time_budget")
    budget[field] += 1
    source = tmp_path / "changed-budget.yaml"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=f"RL {field}=.*canonical budget"):
        load_round(source)


@pytest.mark.parametrize("profile,gpus", [("one-rack", 60), ("three-rack", 192)])
@pytest.mark.parametrize("resume", [False, True])
def test_round_adopts_sft_or_native_rl_without_sft_and_keeps_absolute_steps(
    round_inputs, local_round, tmp_path, gpus, resume
):
    _, data = round_inputs
    spec = dataclasses.replace(
        local_round,
        rl=dataclasses.replace(local_round.rl, native_checkpoint=str(tmp_path / "global_step_13") if resume else None),
    )
    terminal = adopted_round_rl(spec)
    stages = graph_handles([terminal])
    assert all(step.adopt_source for step in stages if step is not terminal)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    assert launch["skyrl"]["trainer"]["max_steps"] == 15
    assert launch["iris"]["allocation"]["num_nodes"] * launch["iris"]["allocation"]["gpus_per_node"] == gpus
    assert launch["iris"]["timeout"] == 48 * 3600
    callbacks = {cb["type"]: cb for cb in launch["skyrl"]["trainer"]["callbacks"]}
    assert callbacks["checkpoint"]["save_steps"] == 2
    assert {key: callbacks["wall_time_budget"][key] for key in RL_TRAINING_BUDGET} == RL_TRAINING_BUDGET
    assert callbacks["best_checkpoint"]["expected_tasks"] == 200
    assert callbacks["best_checkpoint"]["minimum_scored_tasks"] == 180
    assert callbacks["best_checkpoint"]["selection_eligibility"] == "covered"
    if resume:
        assert launch["inputs"]["starting_checkpoint"]["uri"].endswith("/global_step_13")
    else:
        assert "starting_checkpoint" not in launch["inputs"]
    label = sorted(json.loads((data / "manifest.json").read_text())["composition"])[0]
    (data / label / "train.parquet").write_bytes(b"changed after adoption")
    with pytest.raises(ValueError, match="Round data changed"):
        terminal.build_config(context)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_preserves_response_budget_for_train_eval_and_pi(local_round, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["context_budget"].update(request_window_tokens=131072, max_new_tokens_per_turn=98304)
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    document = terminal.build_config(context).launch_config_yaml
    validate_launch(document, tmp_path / "composed", model_config=b"{}")
    launch = yaml.safe_load((tmp_path / "composed" / "compiled.yaml").read_text())["skyrl"]
    generator = launch["generator"]
    assert generator["sampling_params"]["max_generate_length"] == 98304
    assert generator["eval_sampling_params"]["max_generate_length"] == 98304
    assert generator["max_input_length"] == 32768
    agent = launch["terminal_bench_config"]["installed_agent"]["agent"]["kwargs"]
    assert agent["model_info"] == {"max_input_tokens": 131072, "max_output_tokens": 98304}
    assert agent["compaction_reserve_tokens"] == 98304


@pytest.mark.parametrize("profile", ["throughput-train-track1"])
def test_mochi_96k_round_aligns_overlong_window_with_response_cap(local_round, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    terminal = adopted_round_rl(local_round)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    validate_launch(terminal.build_config(context).launch_config_yaml, tmp_path / "composed", model_config=b"{}")
    generator = yaml.safe_load((tmp_path / "composed" / "compiled.yaml").read_text())["skyrl"]["generator"]
    assert generator["sampling_params"]["max_generate_length"] == 98304
    assert generator["eval_sampling_params"]["max_generate_length"] == 98304
    assert generator["trajectory_reward_shaping"]["overlong"] == {
        "l_max": 98304,
        "l_cache": 16384,
        "penalty_scale": 0.0,
    }


@pytest.mark.parametrize("profile", ["throughput-train-track1"])
def test_mochi_normalized_round_resolves_batch_optimizer_and_route_capacity(local_round, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    terminal = adopted_round_rl(local_round)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    validate_launch(terminal.build_config(context).launch_config_yaml, tmp_path / "composed", model_config=b"{}")
    config = yaml.safe_load((tmp_path / "composed" / "compiled.yaml").read_text())["skyrl"]
    trainer, generator = config["trainer"], config["generator"]
    assert trainer["train_batch_size"] == trainer["policy_mini_batch_size"] == 64
    assert generator["n_samples_per_prompt"] == 8
    optimizer = trainer["policy"]["optimizer_config"]
    assert optimizer["lr"] == 1e-5
    assert optimizer["optimizer_kwargs"]["adam_lr"] == 2.307692307692308e-6
    assert optimizer["max_grad_norm"] == 0.5
    buffer = trainer["rollout_buffer"]
    assert buffer["eval_drain"] == "priority"
    assert generator["engine_init_kwargs"]["scheduling_policy"] == "priority"
    assert buffer["max_staleness_steps"] == 3
    assert buffer["consumed_route_quotas"] == {"agentic": 22, "nonagentic": 42}
    assert buffer["max_in_flight"] >= 4 * trainer["train_batch_size"]
    two_batches = 2 * trainer["train_batch_size"] * generator["n_samples_per_prompt"]
    slots = (
        generator["num_inference_engines"] * generator["inference_engine_data_parallel_size"] * generator["max_num_seqs"]
    )
    assert two_batches == 1024 <= slots
    assert 2 * buffer["consumed_route_quotas"]["agentic"] * generator["n_samples_per_prompt"] <= (
        config["terminal_bench_config"]["harbor"]["n_concurrent_trials"]
    )
    wall_budget = next(callback for callback in trainer["callbacks"] if callback["type"] == "wall_time_budget")
    assert wall_budget["groups_per_step"] == 64
    assert generator["max_input_length"] == 32768
    assert generator["sampling_params"]["max_generate_length"] == 98304
    assert generator["eval_sampling_params"]["max_generate_length"] == 98304


def test_exp4_dspark_round_keeps_normalized_training_and_derives_curriculum_quotas(round_inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    model, data = round_inputs
    authored = load_round(ROUND_DIR / "mochi-exp4-dspark-rl.yaml")
    draft = round_draft(authored)
    baseline = load_round(ROUND_DIR / "mochi-v5-exp3-rl.yaml")
    assert authored.adopted_model == baseline.adopted_model
    assert authored.adopted_model.checkpoint_step == 2100
    assert draft.trained_for_initializer != authored.adopted_model.uri
    assert "step-400407/" in draft.trained_for_initializer
    assert authored.evaluation_checkpoints == "best_and_final"
    manifest_path = model / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fixture_checkpoint = "s3://source/checkpoints/step-" + str(authored.adopted_model.checkpoint_step)
    manifest["request"]["config"]["checkpoint"] = fixture_checkpoint
    manifest_path.write_text(json.dumps(manifest))
    recipe = copy.deepcopy(authored.rl.recipe)
    fixture_manifest = json.loads((data / "manifest.json").read_text())
    recipe["trainer"]["rollout_buffer"]["source_routes"] = fixture_manifest["routes"]
    spec = dataclasses.replace(
        authored,
        adopted_model=dataclasses.replace(
            authored.adopted_model,
            uri=str(model),
            config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            export_manifest_sha256=hashlib.sha256((model / "export-manifest.json").read_bytes()).hexdigest(),
        ),
        rl=dataclasses.replace(authored.rl, pools={"uri": str(data)}, recipe=recipe),
    )
    training = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=training.deps, runtime_args=training.runtime_args
    )
    built = training.build_config(context)
    document = yaml.safe_load(built.launch_config_yaml)
    document["inputs"]["model"]["uri"] = authored.adopted_model.uri
    validate_launch(yaml.safe_dump(document), tmp_path / "composed", model_config=b"{}")
    launch = yaml.safe_load((tmp_path / "composed/compiled.yaml").read_text())
    trainer, generator = launch["skyrl"]["trainer"], launch["skyrl"]["generator"]
    buffer = trainer["rollout_buffer"]
    assert (buffer["batch_policy"], buffer["eval_drain"], buffer["max_staleness_steps"]) == ("rolling", "priority", 3)
    assert buffer["consumed_route_quotas"] == {"agentic": 22, "nonagentic": 42}
    assert trainer["train_batch_size"] == trainer["policy_mini_batch_size"] == 64
    assert generator["n_samples_per_prompt"] == 8
    assert generator["gpu_memory_utilization"] == 0.775
    assert buffer["source_routes"]["nvidia/Llama-Nemotron-Post-Training-Dataset"] == "nonagentic"
    assert generator["speculative_decoding"] == draft.speculative_decoding()
    assert generator["engine_init_kwargs"]["scheduling_policy"] == "priority"
    assert generator["inference_engine_tensor_parallel_size"] == 1
    assert generator["model_dtype"] == "bfloat16"
    assert generator["sampling_params"]["max_generate_length"] == 98304
    assert generator["eval_sampling_params"]["max_generate_length"] == 98304
    assert generator["engine_init_kwargs"]["max_model_len"] == 131072
    agent = launch["skyrl"]["terminal_bench_config"]["installed_agent"]["agent"]["kwargs"]
    assert agent["model_info"] == {"max_input_tokens": 131072, "max_output_tokens": 24576}
    optimizer = trainer["policy"]["optimizer_config"]
    assert (optimizer["optimizer"], optimizer["lr"], optimizer["max_grad_norm"]) == ("MuonH", 1e-5, 0.5)
    assert optimizer["optimizer_kwargs"]["offload_momentum"]
    callbacks = {cb["type"]: cb for cb in trainer["callbacks"]}
    assert "inference_stats" in callbacks
    assert authored.rl.recipe["trainer"]["training_metrics"]
    assert (callbacks["best_checkpoint"]["expected_tasks"], callbacks["best_checkpoint"]["minimum_scored_tasks"]) == (
        200,
        180,
    )
    assert {key: callbacks["wall_time_budget"][key] for key in RL_TRAINING_BUDGET} == RL_TRAINING_BUDGET
    allocation = launch["iris"]["allocation"]
    assert allocation["num_nodes"] * allocation["gpus_per_node"] == 128
    assert built.export_hf and trainer["hf_export_checkpoint"] == "best"


def test_v5_exp3_round_preserves_route_mix_and_context_through_native_composition(round_inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    model, data = round_inputs
    authored = load_round(ROUND_DIR / "mochi-v5-exp3-rl.yaml")
    assert authored.adopted_model.checkpoint_step == 2100
    assert authored.evaluation_checkpoints == "best_and_final"
    curriculum = exp3_rl_curriculum(
        {
            route: tuple(PoolRow(id=f"{route}-{i}", sha256=f"{i:064x}") for i in range(180))
            for route in ("agentic", "nonagentic")
        }
    )
    files = {"validation.parquet": hashlib.sha256((data / "validation.parquet").read_bytes()).hexdigest()}
    for route in ("agentic", "nonagentic"):
        (data / route).mkdir()
        content = f"frozen {route} training fixture".encode()
        (data / route / "train.parquet").write_bytes(content)
        files[f"{route}/train.parquet"] = hashlib.sha256(content).hexdigest()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": curriculum.identity,
                "validation": validation_payload(curriculum.validation),
                "composition": {"agentic": 180, "nonagentic": 180},
                "routes": authored.rl.recipe["trainer"]["rollout_buffer"]["source_routes"],
                "files": files,
            }
        )
    )
    spec = dataclasses.replace(
        authored,
        adopted_model=dataclasses.replace(
            authored.adopted_model,
            uri=str(model),
            checkpoint_step=400407,
            config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            export_manifest_sha256=hashlib.sha256((model / "export-manifest.json").read_bytes()).hexdigest(),
        ),
        rl=dataclasses.replace(authored.rl, pools={"uri": str(data)}),
    )
    training = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=training.deps, runtime_args=training.runtime_args
    )
    built = training.build_config(context)
    document = yaml.safe_load(built.launch_config_yaml)
    # Metadata is local in this fixture; terminal export needs a replayable cloud locator.
    document["inputs"]["model"]["uri"] = authored.adopted_model.uri
    validate_launch(yaml.safe_dump(document), tmp_path / "composed", model_config=b"{}")
    launch = yaml.safe_load((tmp_path / "composed" / "compiled.yaml").read_text())
    config = launch["skyrl"]
    trainer, generator = config["trainer"], config["generator"]
    buffer = trainer["rollout_buffer"]
    assert buffer["batch_policy"] == "rolling"
    assert buffer["eval_drain"] == "priority"
    assert buffer["consumed_route_quotas"] == {"agentic": 32, "nonagentic": 32}
    assert buffer["max_staleness_steps"] == 3
    assert trainer["train_batch_size"] == trainer["policy_mini_batch_size"] == 64
    assert generator["n_samples_per_prompt"] == 8
    assert generator["engine_init_kwargs"]["scheduling_policy"] == "priority"
    optimizer = trainer["policy"]["optimizer_config"]
    assert optimizer["optimizer"] == "MuonH"
    assert optimizer["lr"] == 1e-5
    assert optimizer["max_grad_norm"] == 0.5
    assert optimizer["optimizer_kwargs"]["offload_momentum"]
    assert generator["sampling_params"]["max_generate_length"] == 98304
    assert generator["eval_sampling_params"]["max_generate_length"] == 98304
    assert generator["engine_init_kwargs"]["max_model_len"] == 131072
    agent = config["terminal_bench_config"]["installed_agent"]["agent"]["kwargs"]
    assert agent["model_info"] == {"max_input_tokens": 131072, "max_output_tokens": 24576}
    assert agent["compaction_reserve_tokens"] == 24576
    callbacks = {cb["type"]: cb for cb in trainer["callbacks"]}
    assert callbacks["best_checkpoint"]["expected_tasks"] == 200
    assert callbacks["best_checkpoint"]["minimum_scored_tasks"] == 180
    assert {key: callbacks["wall_time_budget"][key] for key in RL_TRAINING_BUDGET} == RL_TRAINING_BUDGET
    assert trainer["hf_export_checkpoint"] == "best"
    assert built.export_hf
    allocation = launch["iris"]["allocation"]
    assert allocation["num_nodes"] * allocation["gpus_per_node"] == 128
    assert launch["artifacts"]["resume_checkpoint_count"] == 1


def _write_schema1_model(tmp_path: Path, checkpoint_step: int) -> tuple[Path, dict[str, str]]:
    """A Snowball-style handoff export: schema-1 config and a direct step manifest."""
    model = tmp_path / "model-schema1"
    model.mkdir()
    manifest = json.dumps({"step": checkpoint_step}).encode()
    config = json.dumps(
        {"grugmoe_artifact_schema_version": 1, "num_hidden_layers": 96, "num_attention_heads": 96, "num_experts": 384}
    ).encode()
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_bytes(config)
    pins = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ("config.json", "export-manifest.json")
    }
    return model, pins


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_adopts_schema1_step_manifest_export(round_inputs, local_round):
    _, data = round_inputs
    model, pins = _write_schema1_model(Path(data).parent, 400407)
    spec = dataclasses.replace(
        local_round,
        adopted_model=dataclasses.replace(
            local_round.adopted_model,
            uri=str(model),
            config_sha256=pins["config.json"],
            export_manifest_sha256=pins["export-manifest.json"],
        ),
    )
    assert adopted_round_rl(spec) is not None


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_rejects_schema1_step_mismatch(round_inputs, local_round):
    _, data = round_inputs
    model, pins = _write_schema1_model(Path(data).parent, 400500)
    spec = dataclasses.replace(
        local_round,
        adopted_model=dataclasses.replace(
            local_round.adopted_model,
            uri=str(model),
            config_sha256=pins["config.json"],
            export_manifest_sha256=pins["export-manifest.json"],
        ),
    )
    with pytest.raises(ValueError, match="checkpoint step changed"):
        adopted_round_rl(spec)


@pytest.mark.parametrize("profile", ["one-rack", "three-rack"])
def test_round_derives_route_quotas_from_the_curriculum(local_round, tmp_path):
    """No declared quotas: the buffer's window quotas derive from the pools
    curriculum's own mixture at the run's batch size."""
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    buffer = launch["skyrl"]["trainer"]["rollout_buffer"]
    manifest = json.loads((Path(local_round.rl.pools["uri"]) / "manifest.json").read_text())
    curriculum = exp4_rl_curriculum(exp4_validation_from_manifest(manifest["validation"]))
    batch = launch["skyrl"]["trainer"]["train_batch_size"]
    derived = consumed_route_quotas(curriculum, buffer["source_routes"], batch)
    assert buffer["consumed_route_quotas"] == derived
    assert sum(buffer["consumed_route_quotas"].values()) == batch


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_accepts_a_declared_quota_that_matches_the_derivation(local_round, tmp_path):
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    plain_spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    plain = adopted_round_rl(plain_spec)
    context = StepContext.for_run(str(tmp_path / "run"), str(tmp_path), deps=plain.deps, runtime_args=plain.runtime_args)
    launch = yaml.safe_load(plain.build_config(context).launch_config_yaml)
    derived = launch["skyrl"]["trainer"]["rollout_buffer"]["consumed_route_quotas"]
    declared = copy.deepcopy(local_round.rl.recipe)
    declared["trainer"]["rollout_buffer"]["consumed_route_quotas"] = derived
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=declared))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "declared"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    assert launch["skyrl"]["trainer"]["rollout_buffer"]["consumed_route_quotas"] == derived


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_rejects_a_declared_quota_that_disagrees_with_the_derivation(local_round):
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"]["consumed_route_quotas"] = {"agentic": 22, "nonagentic": 42}
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    with pytest.raises(ValueError, match="disagrees with the quotas"):
        adopted_round_rl(spec)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_aliases_raw_row_sources_to_their_teacher_routes(local_round, tmp_path):
    """The stamped pools keep raw ``data_source`` values; the compiled launch must
    classify them so the runtime's route check passes."""
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    routes = launch["skyrl"]["trainer"]["rollout_buffer"]["source_routes"]
    manifest = json.loads((Path(local_round.rl.pools["uri"]) / "manifest.json").read_text())
    for label, route in manifest["routes"].items():
        assert routes[label] == route
    assert routes["nvidia/Llama-Nemotron-Post-Training-Dataset"] == routes["nemotron_if"]
    reasoning_gym_url = (
        "https://files.pythonhosted.org/packages/42/21/"
        "bb9f6d2f76424f8b516fe83f8026e52e466b0202341bb4ee7a9861e06277/reasoning_gym-0.1.25-py3-none-any.whl"
    )
    assert routes[reasoning_gym_url] == routes["Reasoning Gym:reasoning_gym"]


@pytest.mark.parametrize("profile", ["one-rack", "three-rack"])
def test_round_best_and_final_native_checkpoints_use_permanent_storage(
    local_round, tmp_path, permanent_path, monkeypatch
):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    terminal = adopted_round_rl(local_round)
    output = permanent_path / "round"
    context = StepContext.for_run(
        str(output), str(output.parent), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    config = terminal.build_config(context)
    launch = yaml.safe_load(config.launch_config_yaml)
    assert config.output.checkpoint_root == str(output / "checkpoints")
    assert launch["artifacts"]["checkpoint_root"] == config.output.checkpoint_root
    assert str(tmp_path / "temporary") in config.output.attempts_root

    validate_launch(config.launch_config_yaml, tmp_path / "composed", model_config=b"{}")
    compiled = yaml.safe_load((tmp_path / "composed" / "compiled.yaml").read_text())
    checkpoint_root = Path(compiled["skyrl"]["trainer"]["ckpt_path"])
    inventory = {"selected": str(checkpoint_root / "global_step_14"), "final": str(checkpoint_root / "global_step_15")}
    for uri in inventory.values():
        path = Path(uri)
        (path / "policy").mkdir(parents=True)
        (path / "trainer_state.pt").write_bytes(b"native optimizer state fixture")
        (path / "policy" / "weights.distcp").write_bytes(b"native policy fixture")
    receipt = assert_best_rl_checkpoint(str(tmp_path / "archive"), inventory)
    assert receipt["selected"]["uri"] == str(output / "checkpoints" / "global_step_14")
    assert receipt["final"]["uri"] == str(output / "checkpoints" / "global_step_15")


@pytest.mark.parametrize("profile", ["one-rack"])
def test_adopted_exp5_round_uses_its_eleven_source_curriculum(local_round, tmp_path):
    shares = dict(EXP5_TARGET_SHARES)
    rows = {label: tuple(PoolRow(id=f"{label}-{i}", sha256=f"{i:064x}") for i in range(80)) for label in shares}
    curriculum = exp5_rl_curriculum(exp5_validation_split(rows).validation)
    root = Path(local_round.rl.pools["uri"])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["curriculum"] = curriculum.identity
    manifest["validation"] = validation_payload(curriculum.validation)
    manifest["composition"] = {label: 80 for label in shares}
    routes = {label: "agentic" if label == "long-loop-smartie-swesmith" else "nonagentic" for label in shares}
    manifest["routes"] = routes
    for label in shares:
        path = root / label / "train.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"pinned {label} pool".encode())
        manifest["files"][f"{label}/train.parquet"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"]["source_routes"] = routes
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "exp5-run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    buffer = launch["skyrl"]["trainer"]["rollout_buffer"]
    assert buffer["consumed_route_quotas"] == consumed_route_quotas(
        curriculum, routes, launch["skyrl"]["trainer"]["train_batch_size"]
    )
    for raw in (
        "allenai/IF_multi_constraints_upto5",
        "OpenResearcher/OpenResearcher-Dataset",
        "pafitis/HARDMath_processed_training",
    ):
        assert buffer["source_routes"][raw] == "nonagentic"


_DRAFT_NAME = "dspark-dr-doom-datakit-sft-0921"


def _round_launch(round_spec, tmp_path):
    terminal = adopted_round_rl(round_spec)
    context = StepContext.for_run(
        str(tmp_path / "draft-run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    return yaml.safe_load(terminal.build_config(context).launch_config_yaml)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_draft_defaults_to_plain_decoding(local_round, tmp_path):
    launch = _round_launch(local_round, tmp_path)
    assert launch["skyrl"]["generator"].get("speculative_decoding") is None


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_draft_binding_injects_the_registered_dspark(local_round, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, draft=_DRAFT_NAME))
    draft = round_draft(spec)
    assert draft.trained_for_initializer.startswith("snowball-datakit-sft 2026.09.21 initializer")
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "draft-run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    document = terminal.build_config(context).launch_config_yaml
    validate_launch(document, tmp_path / "draft-composed")
    launch = yaml.safe_load((tmp_path / "draft-composed" / "compiled.yaml").read_text())["skyrl"]
    assert launch["generator"]["speculative_decoding"] == {
        "method": "dspark",
        "model": {"source_uri": draft.source_uri, "source_identity": draft.source_identity},
        "num_speculative_tokens": draft.num_speculative_tokens,
    }
    plan = adopted_round_resolved_plan(spec, source_revision="0" * 40, workspace=Path.cwd())
    assert plan["draft_binding"] == dataclasses.asdict(draft)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_rejects_unknown_draft_names(local_round):
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, draft="not-a-draft"))
    with pytest.raises(ValueError, match="Unknown round draft binding"):
        round_draft(spec)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_rejects_a_registered_draft_alongside_an_explicit_block(local_round):
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["generator"]["speculative_decoding"] = DRAFTS[_DRAFT_NAME].speculative_decoding()
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe, draft=_DRAFT_NAME))
    with pytest.raises(ValueError, match="registered draft and an explicit"):
        adopted_round_rl(spec)


_OPENRESEARCHER_PATH = "OpenResearcher:OpenResearcher-Dataset/train.parquet"


def _exp5_style_pools(root, *, with_openresearcher: bool) -> dict:
    """Rewrite the fixture pools to the eleven-source Exp5 composition with route metadata."""
    manifest = json.loads((root / "manifest.json").read_text())
    label = "OpenResearcher:OpenResearcher-Dataset"
    content = f"pinned {label} pool".encode()
    (root / label).mkdir(parents=True, exist_ok=True)
    (root / label / "train.parquet").write_bytes(content)
    manifest["files"][f"{label}/train.parquet"] = hashlib.sha256(content).hexdigest()
    manifest["composition"][label] = 283
    manifest["routes"][label] = "nonagentic"
    if with_openresearcher:
        manifest["openresearcher"] = {
            "dataset_id": "open-athena/openresearcher-openresearcher-dataset-service-20261121-v1",
            "dataset_revision": "0c4f39584d7f0121e97f6ad4f4761749add56fef",
            "parquet_sha256": hashlib.sha256(content).hexdigest(),
        }
    else:
        manifest.pop("openresearcher", None)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _exp5_style_round(round_spec, *, with_openresearcher: bool):
    """Rewrite the fixture pools to the eleven-source Exp5 composition and declare its routes."""
    root = Path(round_spec.rl.pools["uri"])
    manifest = _exp5_style_pools(root, with_openresearcher=with_openresearcher)
    recipe = copy.deepcopy(round_spec.rl.recipe)
    recipe["trainer"]["rollout_buffer"]["source_routes"] = dict(manifest["routes"])
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    return dataclasses.replace(round_spec, rl=dataclasses.replace(round_spec.rl, recipe=recipe))


def _service_binding() -> OpenResearcherService:
    return OpenResearcherService(
        dataset_id="open-athena/openresearcher-openresearcher-dataset-service-20261121-v1",
        dataset_revision="0c4f39584d7f0121e97f6ad4f4761749add56fef",
        index_uri="s3://marin-us-east-08a/marin/rl-data/openresearcher/bm25-index/2026.11.15.1",
        parquet_sha256=hashlib.sha256(b"pinned OpenResearcher:OpenResearcher-Dataset pool").hexdigest(),
    )


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_requires_openresearcher_service_when_pools_carry_the_route(round_inputs, local_round):
    spec = _exp5_style_round(local_round, with_openresearcher=True)
    with pytest.raises(ValueError, match="OpenResearcher service binding"):
        adopted_round_rl(spec)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_rejects_a_service_binding_without_openresearcher_pools(round_inputs, local_round):
    spec = _exp5_style_round(local_round, with_openresearcher=False)
    spec = dataclasses.replace(spec, rl=dataclasses.replace(spec.rl, openresearcher_service=_service_binding()))
    with pytest.raises(ValueError, match="OpenResearcher service binding"):
        adopted_round_rl(spec)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_openresearcher_binding_matches_the_published_admission(round_inputs, local_round):
    spec = _exp5_style_round(local_round, with_openresearcher=True)
    binding = dataclasses.replace(_service_binding(), dataset_revision="other-revision")
    spec = dataclasses.replace(spec, rl=dataclasses.replace(spec.rl, openresearcher_service=binding))
    with pytest.raises(ValueError, match="dataset_revision differs from the published pools"):
        adopted_round_rl(spec)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_openresearcher_service_wires_machine_contract_and_retrieval_run(round_inputs, local_round, tmp_path):
    spec = _exp5_style_round(local_round, with_openresearcher=True)
    spec = dataclasses.replace(
        spec, rl=dataclasses.replace(spec.rl, openresearcher_service=_service_binding(), max_steps=2)
    )
    terminal = adopted_round_rl(spec)
    assert terminal.run.func is run_with_retrieval
    assert terminal.run.keywords["parquet_relative_path"] == _OPENRESEARCHER_PATH
    context = StepContext.for_run(
        str(tmp_path / "service-run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    machine = launch["skyrl"]["environment"]["task_sessions"]["task_spec"]["task_machine_by_dataset"]
    assert machine["OpenResearcher/OpenResearcher-Dataset"]["backend"] == "daytona"
    buffer = launch["skyrl"]["trainer"]["rollout_buffer"]
    assert buffer["source_routes"]["OpenResearcher/OpenResearcher-Dataset"] == "nonagentic"


@pytest.mark.parametrize("profile", ["one-rack"])
def test_specdec_gate_arms_differ_only_in_the_draft_binding(round_inputs, local_round, tmp_path):
    """Load the real gate round YAMLs onto local fixtures and compare their launch configs."""
    _, data = round_inputs
    plain = load_round(ROUND_DIR / "snowball-specdec-gate-plain.yaml")
    dspark = load_round(ROUND_DIR / "snowball-specdec-gate-dspark.yaml")
    assert plain.rl.draft is None and dspark.rl.draft is not None
    manifest = _exp5_style_pools(data, with_openresearcher=True)
    manifest["routes"] = dict(plain.rl.recipe["trainer"]["rollout_buffer"]["source_routes"])
    (data / "manifest.json").write_text(json.dumps(manifest))
    _, hf, config_sha, manifest_sha, seal = _certified_export_fixture(tmp_path)
    arms = []
    for spec in (plain, dspark):
        local = dataclasses.replace(
            spec,
            adopted_model=dataclasses.replace(
                spec.adopted_model,
                uri=str(hf),
                export_manifest_sha256=None,
                config_sha256=config_sha,
                checkpoint_step=527,
                model_manifest_sha256=manifest_sha,
                certification_seal=seal,
            ),
            rl=dataclasses.replace(
                spec.rl,
                pools={"uri": str(data)},
                openresearcher_service=dataclasses.replace(
                    spec.rl.openresearcher_service,
                    parquet_sha256=hashlib.sha256((data / _OPENRESEARCHER_PATH).read_bytes()).hexdigest(),
                ),
            ),
        )
        arms.append(_round_launch(local, tmp_path))

    def comparable(launch):
        document = copy.deepcopy(launch)
        text = json.dumps(document, sort_keys=True)
        for arm in ("plain", "dspark"):
            text = text.replace(f"snowball-specdec-gate-{arm}", "ARM")
            text = text.replace(f"-4eng-{arm}", "-4eng-ARM")
            text = text.replace(f"2026.11.23.{'1' if arm == 'plain' else '2'}", "VERSION")
        document = json.loads(text)
        document["run"].pop("attempt_id", None)
        # The coordinator job name carries a per-arm fingerprint suffix; the draft makes the arms differ.
        document["iris"]["job_name"] = re.sub(r"-[0-9a-f]+$", "-FINGERPRINT", document["iris"]["job_name"])
        return document

    plain_doc, dspark_doc = comparable(arms[0]), comparable(arms[1])
    assert plain_doc["skyrl"]["generator"].get("speculative_decoding") is None
    draft = dspark_doc["skyrl"]["generator"].pop("speculative_decoding")
    assert draft["method"] == "dspark" and draft["num_speculative_tokens"] == 4

    def differences(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            return [
                d
                for key in sorted(set(left) | set(right))
                for d in differences(left.get(key), right.get(key), f"{prefix}.{key}")
            ]
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            pairs = [
                differences(l, r, f"{prefix}[{index}]") for index, (l, r) in enumerate(zip(left, right, strict=False))
            ]
            return [d for group in pairs for d in group]
        return [] if left == right else [(prefix, left, right)]

    assert differences(plain_doc, dspark_doc) == []


def _certified_export_fixture(tmp_path):
    """Build a certified step-527 export: seal, receipts, staging manifest and an HF export."""
    root = tmp_path / "sft"
    checkpoint = root / "checkpoints/step-527"
    checkpoint.mkdir(parents=True)
    hf = root / "hf/step-527"
    hf.mkdir(parents=True)
    receipts = root / "recovery/finalization-527"
    receipts.mkdir(parents=True)

    def write(path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(document).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    config_sha = write(
        hf / "config.json",
        {"grugmoe_artifact_schema_version": 3, "num_hidden_layers": 96, "num_attention_heads": 96, "num_experts": 384},
    )
    manifest_sha = write(
        hf / ".marinskyrl-model-manifest.json",
        {
            "format_version": 1,
            "model_id": None,
            "revision": None,
            "tokenizer_mode": "embedded",
            "identity": "sha256:" + "a" * 64,
            "files": [{"path": "config.json", "sha256": config_sha, "size": 1}],
        },
    )
    completion = root / "telemetry/completion.json"
    completion_sha = write(completion, {"checkpoint": str(checkpoint), "step": 527, "completed_updates": 528})
    native_sha = write(
        receipts / "native-checkpoint.json",
        {"checkpoint": str(checkpoint), "step": 527, "is_temporary": False, "verdict": "full"},
    )
    hf_sha = write(
        receipts / "hf-export.json",
        {
            "native_checkpoint": str(checkpoint),
            "source": str(hf),
            "step": 527,
            "verdict": "full_hash_and_index",
            "files": [{"path": "config.json", "sha256": config_sha}],
        },
    )
    seal_sha = write(
        receipts / "seal.json",
        {
            "status": "COMPLETE",
            "step": 527,
            "completed_updates": 528,
            "native_checkpoint": str(checkpoint),
            "hf_export": str(hf),
            "completion_receipt": str(completion),
            "completion_sha256": completion_sha,
            "native_receipt_sha256": native_sha,
            "hf_receipt_sha256": hf_sha,
        },
    )
    return root, hf, config_sha, manifest_sha, FilePin(str(receipts / "seal.json"), seal_sha)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_adopts_a_certified_export_through_the_seal(round_inputs, local_round, tmp_path):
    model_dir, _ = round_inputs
    _, hf, config_sha, manifest_sha, seal = _certified_export_fixture(tmp_path)
    spec = dataclasses.replace(
        local_round,
        adopted_model=dataclasses.replace(
            local_round.adopted_model,
            uri=str(hf),
            export_manifest_sha256=None,
            config_sha256=config_sha,
            checkpoint_step=527,
            model_manifest_sha256=manifest_sha,
            certification_seal=seal,
        ),
    )
    assert adopted_round_rl(spec) is not None
    # A seal pin that no longer matches fails the pinned read, not the step check.
    tampered = dataclasses.replace(
        spec.adopted_model,
        certification_seal=FilePin(seal.path, hashlib.sha256(b"tampered").hexdigest()),
    )
    with pytest.raises(ValueError, match="Pinned file changed"):
        adopted_round_rl(dataclasses.replace(spec, adopted_model=tampered))
    # A seal certifying a different step fails the strict step check.
    moved = dataclasses.replace(
        spec.adopted_model,
        certification_seal=FilePin(seal.path, seal.sha256),
        checkpoint_step=400407,
    )
    with pytest.raises(ValueError, match="does not certify the adopted checkpoint step"):
        adopted_round_rl(dataclasses.replace(spec, adopted_model=moved))
    # The legacy route still rejects the foreign export manifest's step.
    legacy = dataclasses.replace(
        spec,
        adopted_model=dataclasses.replace(
            spec.adopted_model,
            uri=str(model_dir),
            checkpoint_step=527,
            certification_seal=None,
            export_manifest_sha256=pins_export_manifest(model_dir),
        ),
    )
    with pytest.raises(ValueError, match="checkpoint step changed"):
        adopted_round_rl(legacy)


def pins_export_manifest(model_dir):
    return hashlib.sha256((model_dir / "export-manifest.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("profile", ["one-rack"])
def test_exp6_pinned_patch_resolves_exact_context_staleness_and_snowball_optimizer(local_round, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    source = Path(__file__).parents[1] / "specs/exp6-rl.yaml"
    patch = tmp_path / "exp6-rl.yaml"
    patch.write_bytes(source.read_bytes())
    pin = FilePin(str(patch), hashlib.sha256(patch.read_bytes()).hexdigest())
    roles = {
        **local_round.rl.role_plan,
        "policy_num_nodes": 4,
        "policy_num_gpus_per_node": 8,
        "num_inference_engines": 1,
        "inference_engine_data_parallel_size": 8,
        "inference_engine_expert_parallel_size": 8,
        "train_batch_size": 64,
        "policy_mini_batch_size": 64,
        "n_samples_per_prompt": 8,
    }
    authored = dataclasses.asdict(local_round)
    authored["cluster"] = "cw-us-east-02a"
    authored["rl"]["execution"].update(
        cluster="cw-us-east-02a",
        target_cluster="cw-us-east-02a",
        cluster_config="lib/iris/config/cw-us-east-02a.yaml",
        role_topology=None,
        excluded_nvlink_domains=[],
    )
    authored["rl"].update(
        gpu_variant="H100",
        recipe_patch=dataclasses.asdict(pin),
        role_plan=roles,
        megatron={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 4,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
        },
    )
    authored["rl"]["recipe"]["trainer"]["policy"]["megatron_config"]["optimizer_config_kwargs"] = {
        "optimizer_cpu_offload": True,
        "optimizer_offload_fraction": 1.0,
        "overlap_cpu_optimizer_d2h_h2d": True,
    }
    round_file = tmp_path / "exp6-round.yaml"
    round_file.write_text(yaml.safe_dump(authored))
    spec = load_round(round_file)
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    validate_launch(terminal.build_config(context).launch_config_yaml, tmp_path / "composed", model_config=b"{}")
    config = yaml.safe_load((tmp_path / "composed/compiled.yaml").read_text())["skyrl"]
    trainer, generator = config["trainer"], config["generator"]
    assert trainer["rollout_buffer"]["max_staleness_steps"] == 3
    assert trainer["max_prompt_length"] == 32768
    assert trainer["algorithm"]["max_seq_len"] == 131072
    assert generator["max_input_length"] == 32768
    assert generator["sampling_params"]["max_generate_length"] == 98304
    assert generator["eval_sampling_params"]["max_generate_length"] == 98304
    assert generator["engine_init_kwargs"]["max_model_len"] == 131072
    agent = config["terminal_bench_config"]["installed_agent"]["agent"]["kwargs"]
    assert agent["model_info"] == {"max_input_tokens": 131072, "max_output_tokens": 24576}
    assert agent["compaction_reserve_tokens"] == 24576
    assert (trainer["train_batch_size"], generator["n_samples_per_prompt"]) == (64, 8)
    assert trainer["policy"]["optimizer_config"]["max_grad_norm"] == 0.5
    optimizer = trainer["policy"]["optimizer_config"]
    mochi = yaml.safe_load((source.parents[1] / "rounds/mochi-throughput-train-track1.yaml").read_text())
    reference = mochi["rl"]["recipe"]["trainer"]["policy"]["optimizer_config"]
    assert optimizer == {**reference, "max_grad_norm": 0.5, "lr_decay_steps": None}
    offload = trainer["policy"]["megatron_config"]["optimizer_config_kwargs"]
    assert not offload["optimizer_cpu_offload"]
    assert offload["optimizer_offload_fraction"] == 0.0
    assert not offload["overlap_cpu_optimizer_d2h_h2d"]
    placement = trainer["placement"]
    assert (placement["policy_num_nodes"], placement["policy_num_gpus_per_node"]) == (4, 8)
    for role in ("policy", "ref"):
        megatron = trainer[role]["megatron_config"]
        assert (
            megatron["tensor_model_parallel_size"],
            megatron["pipeline_model_parallel_size"],
            megatron["context_parallel_size"],
            megatron["expert_model_parallel_size"],
            megatron["expert_tensor_parallel_size"],
        ) == (1, 2, 4, 16, 1)
        assert megatron["transformer_config_kwargs"]["cp_comm_type"] == "all_gather"
    megatron = trainer["policy"]["megatron_config"]
    assert megatron["output_projection_chunk_size"] == 1024
    assert megatron["memory_profile"] == {
        "enabled": True,
        "max_sequence_length": 131072,
        "valid_token_counts": [85557],
        "minimum_headroom_fraction": 0.1,
    }
    patch.write_text(patch.read_text().replace("max_staleness_steps: 3", "max_staleness_steps: 4"))
    with pytest.raises(ValueError, match="Pinned file changed"):
        adopted_round_rl(spec)
    adversarial = yaml.safe_load(source.read_text())
    callbacks = copy.deepcopy(local_round.rl.recipe["trainer"]["callbacks"])
    for callback in callbacks:
        if callback["type"] == "best_checkpoint":
            callback.update(expected_tasks=1, minimum_scored_tasks=1)
    adversarial["trainer"]["callbacks"] = callbacks
    patch.write_text(yaml.safe_dump(adversarial))
    updated = dataclasses.replace(
        spec,
        rl=dataclasses.replace(
            spec.rl, recipe_patch=FilePin(str(patch), hashlib.sha256(patch.read_bytes()).hexdigest())
        ),
    )
    terminal = adopted_round_rl(updated)
    context = StepContext.for_run(
        str(tmp_path / "adversarial"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    callbacks = yaml.safe_load(terminal.build_config(context).launch_config_yaml)["skyrl"]["trainer"]["callbacks"]
    best = next(callback for callback in callbacks if callback["type"] == "best_checkpoint")
    assert (best["expected_tasks"], best["minimum_scored_tasks"]) == (200, 180)


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_uses_manifest_train_files_without_changing_sampling_labels(local_round, tmp_path):
    data = Path(local_round.rl.pools["uri"])
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mapping = {label: f"source-{index:03d}/train.parquet" for index, label in enumerate(sorted(manifest["composition"]))}
    for label, relative in mapping.items():
        original = f"{label}/train.parquet"
        target = data / relative
        target.parent.mkdir(parents=True)
        target.write_bytes((data / original).read_bytes())
        manifest["files"][relative] = manifest["files"].pop(original)
    manifest["training_files"] = mapping
    manifest_path.write_text(json.dumps(manifest))
    step = adopted_round_rl(local_round)
    context = StepContext.for_run(
        str(tmp_path / "mapped"), str(tmp_path), deps=step.deps, runtime_args=step.runtime_args
    )
    document = yaml.safe_load(step.build_config(context).launch_config_yaml)
    assert {item["relative_path"] for item in document["inputs"]["train_data"]} == set(mapping.values())
    assert set(document["skyrl"]["data"]["sampling"]["domain_weights"]) == set(mapping)
