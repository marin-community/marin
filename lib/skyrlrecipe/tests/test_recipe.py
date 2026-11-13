# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import pickle

import pytest
import skyrlrecipe as schema
from skyrlrecipe import ContextBudget, declared, sections
from skyrlrecipe.documents import set_path
from skyrlrecipe.ownership import DERIVED_PATHS, OWNER_MESSAGES, REMOVED


@pytest.fixture
def base_recipe():
    return schema.SkyRLRecipe(
        context_budget=ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=2),
        data=sections.Data(sampling=sections.Sampling(domain_weights={"math": 2, "code": 1})),
    )


def test_recipe_round_trip_preserves_immutable_containers_and_mapping_order():
    context = ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=2)
    supplied = {
        "context_budget": context.to_skyrl(),
        "generator": {"engine_init_kwargs": {"schedule": {"windows": [1, 2]}, "enabled": True}},
    }
    recipe = schema.SkyRLRecipe.model_validate_json(json.dumps(supplied))
    original = recipe.to_skyrl()
    rendered = recipe.to_skyrl()
    rendered["generator"]["engine_init_kwargs"]["schedule"]["windows"].append(4)
    assert recipe.to_skyrl() == original
    changed = recipe.merge(
        schema.RecipePatch(generator=sections.Generator(engine_init_kwargs={"schedule": {}, "enabled": False}))
    )
    assert changed.to_skyrl()["generator"] == {"engine_init_kwargs": {"schedule": {"windows": [1, 2]}, "enabled": False}}
    for invalid in (float("nan"), float("inf"), object()):
        with pytest.raises(ValueError, match="finite JSON value"):
            schema.SkyRLRecipe(
                context_budget=context, generator=sections.Generator(engine_init_kwargs={"schedule": [invalid]})
            )
    routes = schema.SkyRLRecipe(
        context_budget=context, generator=sections.Generator(engine_init_kwargs={"math": 1.0, "code": 1.0, "chat": 1.0})
    )
    updated = routes.merge(schema.RecipePatch(trainer=sections.Trainer(max_steps=2)))
    assert list(updated.to_skyrl()["generator"]["engine_init_kwargs"]) == ["math", "code", "chat"]
    base = schema.SkyRLRecipe(
        context_budget=context,
        generator=recipe.generator,
        data=sections.Data(sampling=sections.Sampling(domain_weights={"math": 1.0, "code": 1, "chat": 1.0})),
        teacher_routing={
            "domains": declared.TeacherRouting(
                revision="routes-v1", routes={"math": declared.TeacherRoute(teacher="local", weight=1)}
            )
        },
    )
    updated = base.with_settings(
        [
            "data.sampling.domain_weights.math=2",
            "data.sampling.domain_weights.code=2.0",
            "teacher_routing.domains.routes.math.weight=4",
        ]
    )
    assert updated.teacher_routing["domains"].routes["math"].to_skyrl() == {"teacher": "local", "weight": 4}
    assert list(updated.to_skyrl()["data"]["sampling"]["domain_weights"]) == ["math", "code", "chat"]
    assert type(updated.data.sampling.domain_weights["math"]) is int
    assert type(updated.data.sampling.domain_weights["code"]) is float
    callback = declared.EvaluationCallback(
        type="evaluation", eval_on_train_end=True, eval_steps=4, metric_groups={"eval/score": ("eval/a/score",)}
    )
    replaced = callback.replace(metric_groups={})
    assert replaced.to_skyrl() == {"type": "evaluation", "eval_steps": 4, "eval_on_train_end": True, "metric_groups": {}}
    assert callback.metric_groups["eval/score"] == ("eval/a/score",)
    with pytest.raises(ValueError, match="eval_steps"):
        callback.replace(eval_steps="invalid")
    restored = pickle.loads(pickle.dumps(updated))
    assert restored == updated
    with pytest.raises(TypeError):
        restored.generator.engine_init_kwargs["schedule"]["windows"] = (9,)
    for routes in ({1: 2}, {1: 2, "1": 3}, {"nested": {1: 2, "1": 3}}):
        document = {"context_budget": context.to_skyrl(), "generator": {"engine_init_kwargs": routes}}
        original_routes = list(routes.items())
        with pytest.raises(ValueError, match="mapping keys must be strings"):
            schema.SkyRLRecipe.from_document(document)
        with pytest.raises(ValueError, match="mapping keys must be strings"):
            recipe.generator.replace(engine_init_kwargs=routes)
        assert list(document["generator"]["engine_init_kwargs"].items()) == original_routes


def test_settings_merge_required_sections_and_preserve_empty_mappings(base_recipe):
    base = base_recipe
    authored_trainer = {
        "logger": ["console", "wandb"],
        "policy": {
            "optimizer_config": {"scheduler": "constant_with_warmup"},
            "megatron_config": {"optimizer_checkpoint_sharding_type": "dp_reshardable"},
            "model": {"lora": {"dropout": 0.05}},
        },
        "critic": {"optimizer_config": {"scheduler": "constant_with_warmup"}},
        "ref": {"megatron_config": {"optimizer_checkpoint_sharding_type": "fully_reshardable"}},
        "algorithm": {"resolved_group_advantage": {"kind": "none"}},
    }
    part = schema.RecipePatch.from_document(
        {
            "trainer": authored_trainer,
            "generator": {"chat_template": None, "engine_init_kwargs": {"user_options": {"math": 1, "code": [2, 3]}}},
        }
    )
    computed = schema.RecipePatch(trainer=sections.Trainer(ckpt_interval=4))
    expected = base.merge(part, computed)
    grouped = expected.with_settings(
        ["config_groups.algorithm_recipe=grpo", "config_groups.terminal_bench_config=terminal_bench"]
    )
    assert grouped.to_skyrl()["config_groups"] == {
        "terminal_bench_config": "terminal_bench",
        "algorithm_recipe": "grpo",
    }
    local = schema.RecipePatch(environment=sections.Environment(task_sessions={"local": {}}))
    for enabled in (
        base.merge(local),
        base.with_settings(["environment.task_sessions.local={}"]),
    ):
        assert enabled.to_skyrl()["environment"]["task_sessions"]["local"] == {}
    edited = expected.with_settings(["context_budget.max_turns=4", "generator.engine_init_kwargs.user_options.math=2"])
    assert edited.context_budget.to_skyrl() == {
        "request_window_tokens": 256,
        "max_new_tokens_per_turn": 64,
        "max_turns": 4,
    }
    assert list(edited.to_skyrl()["generator"]["engine_init_kwargs"]["user_options"]) == ["math", "code"]
    assert edited.to_skyrl()["trainer"] == {**authored_trainer, "ckpt_interval": 4}
    named_option = edited.with_settings(
        ["generator.engine_init_kwargs.user_options.backend=custom-engine", "trainer.run_name=123"]
    )
    assert named_option.to_skyrl()["generator"]["engine_init_kwargs"]["user_options"]["backend"] == "custom-engine"
    assert named_option.to_skyrl()["trainer"]["run_name"] == "123"
    for setting in ('generator.engine_init_kwargs.user_options={"items":', "trainer.max_steps=lots"):
        with pytest.raises(ValueError) as error:
            edited.with_settings([setting])
        assert f"setting {setting!r}" in str(error.value)
    structured = expected.with_settings(['context_budget={"max_turns":4}'])
    assert structured.context_budget == edited.context_budget
    with pytest.raises(ValueError, match="request_window_tokens must exceed"):
        expected.with_settings(["context_budget.request_window_tokens=64"])


def test_recipe_validation_reports_owned_removed_and_nonnullable_paths(base_recipe):
    base = base_recipe
    for paths, owner in ((set(OWNER_MESSAGES) - DERIVED_PATHS, "launch document"), (DERIVED_PATHS, "context_budget")):
        for path in paths:
            document = base.to_skyrl()
            set_path(document, path, "authored-sentinel")
            with pytest.raises(ValueError) as error:
                schema.SkyRLRecipe.from_document(document)
            assert f"{path}:" in str(error.value)
            assert owner in str(error.value)
            with pytest.raises(ValueError) as setting_error:
                base.with_settings([f"{path}=authored-sentinel"])
            assert f"{path}:" in str(setting_error.value)
            assert owner in str(setting_error.value)
    for path, message in REMOVED.items():
        document = base.to_skyrl()
        set_path(document, path, {})
        with pytest.raises(ValueError) as error:
            schema.SkyRLRecipe.from_document(document)
        assert f"{path}: {message}" in str(error.value)
        with pytest.raises(ValueError) as setting_error:
            base.with_settings([f"{path}={{}}"])
        assert f"{path}: {message}" in str(setting_error.value)
    removed = {"harbor": {"max_episodes": 1}}
    with pytest.raises(ValueError, match=r"use context_budget.max_turns; Harbor reads max_turns"):
        schema.RecipePatch(terminal_bench=removed)
    with pytest.raises(ValueError, match="initial_model_uri"):
        base.with_settings(['trainer.callbacks=[{"type":"best_checkpoint"}]'])
    with pytest.raises(ValueError, match="micro_forward_batch_size_per_gpu"):
        base.with_settings(["trainer.micro_forward_batch_size_per_gpu=null"])
