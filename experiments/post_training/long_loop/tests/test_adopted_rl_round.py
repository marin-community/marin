# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles
from marin.rl.curriculum import PoolRow, consumed_route_quotas, validation_payload

from experiments.post_training.long_loop.archive import assert_best_rl_checkpoint
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.openresearcher_service import OpenResearcherService, run_with_retrieval
from experiments.post_training.long_loop.round_runner import (
    DRAFTS,
    adopted_round_resolved_plan,
    adopted_round_rl,
    load_round,
    round_draft,
)
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
    assert callbacks["wall_time_budget"]["seconds"] == 36000
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
            "dataset_id": "OpenResearcher/OpenResearcher-Dataset",
            "dataset_revision": "test-revision",
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
        dataset_id="OpenResearcher/OpenResearcher-Dataset",
        dataset_revision="test-revision",
        index_uri="s3://marin-us-east-08a/marin/users/benfeuer/openresearcher/index",
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
