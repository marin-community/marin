"""Contract checks for the three-teacher Snowball MOPD smoke on the Nemotron Ultra blend."""

import copy
from pathlib import Path

import pytest
import yaml

from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from cloud.iris.role_plan import derive_num_nodes, derive_role_plan
from infra.rl_data.nemotron_ultra_mopd_subset import TEACHER_ROUTES
from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    LocalInferenceTeacherSpec,
    TeacherEvidenceKind,
    TeacherPlacement,
    validate_distillation_runtime_support,
)
from skyrl_train.config.training_budget import training_budget

CONFIG = Path(__file__).parents[1] / "configs" / "snowball_mopd_ultra_smoke.yaml"
OPD_SMOKE_CONFIG = Path(__file__).parents[1] / "configs" / "snowball_opd_math_smoke.yaml"
STUDENT = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"


class _HPCStub:
    gpus_per_node = 8


@pytest.mark.parametrize(
    ("config", "expected_routes"),
    [
        (CONFIG, {"math": "math", "swe": "swe", "terminal": "terminal"}),
        (OPD_SMOKE_CONFIG, {"default": "math"}),
    ],
    ids=lambda value: value.stem if isinstance(value, Path) else None,
)
def test_snowball_distillation_smoke_pins_one_expert_parallel_teacher_per_route(config, expected_routes):
    parsed = parse_rl_config(str(config), model_override=STUDENT)
    plan = parsed.distillation_plan

    assert plan is not None
    validate_distillation_runtime_support(plan)
    assert plan.objective is DistillationObjectiveKind.SAMPLED_REVERSE_KL
    assert plan.reward_mode is DistillationRewardMode.REPLACE
    assert {route.key: route.teacher_id for route in plan.routing.routes} == expected_routes
    assert len(plan.teachers) == len(expected_routes)
    for teacher in plan.teachers:
        assert isinstance(teacher, LocalInferenceTeacherSpec)
        assert teacher.placement is TeacherPlacement.PINNED
        assert teacher.evidence is TeacherEvidenceKind.CHOSEN_TOKEN
        assert teacher.resources is not None
        # GrugMoE serves only at tensor-parallel 1, so each teacher is one EP8 engine on its own node.
        assert (teacher.resources.tensor_parallel_size, teacher.resources.data_parallel_size) == (1, 8)
        assert teacher.resources.expert_parallel_size == 8
        assert teacher.resources.num_nodes * teacher.resources.gpus_per_node == teacher.resources.gpus_per_engine
    assert len({teacher.resources.colocation_group for teacher in plan.teachers}) == len(plan.teachers)


def test_snowball_mopd_smoke_needs_eight_nodes():
    config = yaml.safe_load(CONFIG.read_text())
    plan = derive_role_plan(config)

    assert derive_num_nodes(plan) == 8


def test_snowball_mopd_smoke_sampler_weights_match_the_hardcoded_routes():
    config = yaml.safe_load(CONFIG.read_text())
    route_keys = set(config["teacher_routing"]["opd"]["routes"])

    assert config["data"]["sampling"]["kind"] == "domain-weighted"
    assert set(config["data"]["sampling"]["domain_weights"]) == route_keys
    assert set(TEACHER_ROUTES.values()) == route_keys


def test_snowball_mopd_smoke_route_weights_survive_config_composition():
    """The base config declares domain_weights as an empty map, so route keys must be allowed in."""
    parsed = parse_rl_config(str(CONFIG), model_override=STUDENT)
    cfg = compose_skyrl_config(
        parsed, {"job_name": "mopd-smoke-test", "experiments_dir": "/tmp/exp", "num_nodes": 8}, _HPCStub()
    ).config

    assert cfg.data.sampling.kind == "domain-weighted"
    assert dict(cfg.data.sampling.domain_weights) == {"math": 1.0, "swe": 1.0, "terminal": 1.0}


def _compose_launch(path: Path):
    parsed = parse_rl_config(str(path), model_override=STUDENT)
    return compose_skyrl_config(
        parsed, {"job_name": "budget-test", "experiments_dir": "/tmp/exp", "num_nodes": 8}, _HPCStub()
    ).config


def _fixed_completion_recipe(rollout_recipe: dict, *, train_batch_size: int, policy_mini_batch_size: int) -> dict:
    """The M1 arm of a rollout recipe: the same routes and context budget, read from fixed rows."""
    recipe = copy.deepcopy(rollout_recipe)
    for section in ("teachers", "teacher_routing"):
        del recipe[section]
    recipe["environment"] = {"env_class": "fixed_completion"}
    algorithm = recipe["trainer"]["algorithm"]
    del algorithm["distillation"]
    algorithm.update(policy_loss_type="sft", advantage_estimator="uniform")
    recipe["trainer"].update(train_batch_size=train_batch_size, policy_mini_batch_size=policy_mini_batch_size)
    recipe["generator"]["n_samples_per_prompt"] = 1
    return recipe


def test_fixed_completion_recipe_matches_the_snowball_mopd_training_budget(tmp_path):
    """An M1 recipe spends M4's budget only when it trains rollouts-per-prompt times as many rows per step."""
    rollout_recipe = yaml.safe_load(CONFIG.read_text())
    rollouts = rollout_recipe["generator"]["n_samples_per_prompt"]
    trainer = rollout_recipe["trainer"]
    m4 = training_budget(_compose_launch(CONFIG))

    matched = tmp_path / "m1_matched.yaml"
    matched.write_text(
        yaml.safe_dump(
            _fixed_completion_recipe(
                rollout_recipe,
                train_batch_size=trainer["train_batch_size"] * rollouts,
                policy_mini_batch_size=trainer["policy_mini_batch_size"] * rollouts,
            )
        )
    )
    assert training_budget(_compose_launch(matched)) == m4

    copied = tmp_path / "m1_copied_batch.yaml"
    copied.write_text(
        yaml.safe_dump(
            _fixed_completion_recipe(
                rollout_recipe,
                train_batch_size=trainer["train_batch_size"],
                policy_mini_batch_size=trainer["policy_mini_batch_size"],
            )
        )
    )
    assert training_budget(_compose_launch(copied)) != m4
