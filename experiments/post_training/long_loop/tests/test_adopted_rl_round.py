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

from experiments.post_training.long_loop.archive import assert_best_rl_checkpoint
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.round_runner import adopted_round_rl, load_round
from experiments.post_training.task_curation.exp4_curriculum import (
    SOURCE_ROUTES,
    exp4_rl_curriculum,
    exp4_validation_from_manifest,
    exp4_validation_split,
)
from marin.rl.curriculum import PoolRow


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
    assert callbacks["best_checkpoint"]["minimum_scored_tasks"] == 108
    assert callbacks["best_checkpoint"]["selection_eligibility"] == "covered"
    if resume:
        assert launch["inputs"]["starting_checkpoint"]["uri"].endswith("/global_step_13")
    else:
        assert "starting_checkpoint" not in launch["inputs"]
    label = sorted(json.loads((data / "manifest.json").read_text())["composition"])[0]
    (data / label / "train.parquet").write_bytes(b"changed after adoption")
    with pytest.raises(ValueError, match="Round data changed"):
        terminal.build_config(context)


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


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_preserves_explicit_route_quotas(local_round, tmp_path):
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"]["consumed_route_quotas"] = {"agentic": 11, "nonagentic": 21}
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    buffer = launch["skyrl"]["trainer"]["rollout_buffer"]
    assert buffer["consumed_route_quotas"] == {"agentic": 11, "nonagentic": 21}


@pytest.mark.parametrize("profile", ["one-rack"])
def test_round_rejects_route_quotas_that_do_not_partition_the_batch(local_round):
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"]["consumed_route_quotas"] = {"agentic": 22, "nonagentic": 42}
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    with pytest.raises(ValueError, match="sum to the training batch groups"):
        adopted_round_rl(spec)


@pytest.mark.parametrize("profile", ["one-rack", "three-rack"])
def test_round_preserves_disabled_route_quotas(local_round, tmp_path):
    recipe = copy.deepcopy(local_round.rl.recipe)
    recipe["trainer"]["rollout_buffer"]["consumed_route_quotas"] = {}
    spec = dataclasses.replace(local_round, rl=dataclasses.replace(local_round.rl, recipe=recipe))
    terminal = adopted_round_rl(spec)
    context = StepContext.for_run(
        str(tmp_path / "run"), str(tmp_path), deps=terminal.deps, runtime_args=terminal.runtime_args
    )
    launch = yaml.safe_load(terminal.build_config(context).launch_config_yaml)
    buffer = launch["skyrl"]["trainer"]["rollout_buffer"]
    assert buffer["consumed_route_quotas"] == {}
    assert buffer["batch_policy"] == "full_batch"


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
