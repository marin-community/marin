# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from itertools import permutations

import pytest
import skyrlrecipe as schema
from skyrlrecipe import ContextBudget, declared, sections
from skyrlrecipe.documents import set_path
from skyrlrecipe.ownership import DERIVED_PATHS, LAUNCH_PATHS, REMOVED


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
    numeric = schema.RecipePatch(generator=sections.Generator(engine_init_kwargs={"samples": [1], "weight": 1}))
    equivalent = schema.RecipePatch(generator=sections.Generator(engine_init_kwargs={"samples": [1.0], "weight": 1.0}))
    schema.SkyRLRecipe.combine(base=recipe, integer=numeric, decimal=equivalent)
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
    restored = pickle.loads(pickle.dumps(updated))
    assert restored == updated
    with pytest.raises(TypeError):
        restored.generator.engine_init_kwargs["schedule"]["windows"] = (9,)
    for routes in ({1: 2}, {1: 2, "1": 3}):
        document = {"context_budget": context.to_skyrl(), "generator": {"engine_init_kwargs": routes}}
        original_routes = list(routes.items())
        with pytest.raises(ValueError, match="mapping keys must be strings"):
            schema.SkyRLRecipe.from_document(document)
        assert list(document["generator"]["engine_init_kwargs"].items()) == original_routes
    assert "hf_save_interval" not in sections.Trainer().model_dump(mode="json")


def test_named_parts_are_order_independent_and_settings_merge_required_sections(base_recipe):
    base = base_recipe
    part = schema.RecipePatch.from_document(
        {
            "trainer": {
                "logger": ["console", "wandb"],
                "policy": {
                    "optimizer_config": {"scheduler": "constant_with_warmup"},
                    "megatron_config": {"optimizer_checkpoint_sharding_type": "dp_reshardable"},
                    "model": {"lora": {"dropout": 0.05}},
                },
                "critic": {"optimizer_config": {"scheduler": "constant_with_warmup"}},
                "ref": {"megatron_config": {"optimizer_checkpoint_sharding_type": "fully_reshardable"}},
                "algorithm": {"resolved_group_advantage": {"kind": "none"}},
            },
            "generator": {"chat_template": None, "engine_init_kwargs": {"user_options": {"math": 1, "code": [2, 3]}}},
        }
    )
    computed = schema.RecipePatch(trainer=sections.Trainer(ckpt_interval=4))
    options = schema.RecipePatch(generator=sections.Generator(engine_init_kwargs={"buckets": [1, 2]}))
    parts = {"policy": part, "computed": computed, "options": options}
    expected = schema.SkyRLRecipe.combine(base=base, **parts)
    grouped = expected.with_settings(
        ["config_groups.algorithm_recipe=grpo", "config_groups.terminal_bench_config=terminal_bench"]
    )
    assert grouped.to_skyrl()["config_groups"] == {
        "terminal_bench_config": "terminal_bench",
        "algorithm_recipe": "grpo",
    }
    assert schema.SkyRLRecipe.from_document(grouped.to_skyrl()) == grouped
    for order in permutations(parts.items()):
        combined = schema.SkyRLRecipe.combine(base=base, **dict(order))
        assert combined == expected
        assert list(combined.to_skyrl()["generator"]["engine_init_kwargs"]) == ["buckets", "user_options"]
    local = schema.RecipePatch(environment=sections.Environment(task_sessions={"local": {}}))
    for enabled in (
        schema.SkyRLRecipe.combine(base=base, local=local),
        base.merge(local),
        base.with_settings(["environment.task_sessions.local={}"]),
    ):
        assert enabled.to_skyrl()["environment"]["task_sessions"]["local"] == {}
    with_parent = base.merge(schema.RecipePatch(generator=sections.Generator(engine_init_kwargs={"parent": None})))
    nested_empty = schema.RecipePatch(generator=sections.Generator(engine_init_kwargs={"parent": {"child": {}}}))
    assert schema.SkyRLRecipe.combine(base=with_parent, empty=nested_empty).to_skyrl()["generator"][
        "engine_init_kwargs"
    ]["parent"] == {"child": {}}
    for first, second in (
        ({"buckets": [1, 2]}, {"buckets": [1, 3]}),
        ({"parent": None}, {"parent": {"child": 2}}),
        ({"parent": {}}, {"parent": {"child": 2}}),
        ({"enabled": True}, {"enabled": 1}),
    ):
        conflicts = (
            ("first", schema.RecipePatch(generator=sections.Generator(engine_init_kwargs=first))),
            ("second", schema.RecipePatch(generator=sections.Generator(engine_init_kwargs=second))),
            ("unrelated", computed),
        )
        for order in permutations(conflicts):
            with pytest.raises(ValueError, match="parts 'first' and 'second'"):
                schema.SkyRLRecipe.combine(base=base, **dict(order))
    edited = expected.with_settings(["context_budget.max_turns=4", "generator.engine_init_kwargs.user_options.math=2"])
    assert edited.context_budget.to_skyrl() == {
        "request_window_tokens": 256,
        "max_new_tokens_per_turn": 64,
        "max_turns": 4,
    }
    assert list(edited.to_skyrl()["generator"]["engine_init_kwargs"]["user_options"]) == ["math", "code"]
    assert edited.to_skyrl()["trainer"] == {**part.to_skyrl()["trainer"], "ckpt_interval": 4}
    named_option = edited.with_settings(["generator.engine_init_kwargs.user_options.backend=custom-engine"])
    assert named_option.to_skyrl()["generator"]["engine_init_kwargs"]["user_options"]["backend"] == "custom-engine"
    with pytest.raises(json.JSONDecodeError):
        edited.with_settings(['generator.engine_init_kwargs.user_options={"items":'])
    structured = expected.with_settings(['context_budget={"max_turns":4}'])
    assert structured.context_budget == edited.context_budget
    with pytest.raises(ValueError, match="request_window_tokens must exceed"):
        expected.with_settings(["context_budget.request_window_tokens=64"])


def test_recipe_validation_reports_owned_removed_and_nonnullable_paths(base_recipe):
    base = base_recipe
    edited = base
    for paths, owner in ((LAUNCH_PATHS, "launch document"), (DERIVED_PATHS, "context_budget")):
        for path in paths:
            if not path.startswith(("terminal_bench.", "environment.task_sessions.", "generator.engine_init_kwargs.")):
                continue
            document = base.to_skyrl()
            set_path(document, path, "authored-sentinel")
            with pytest.raises(ValueError) as error:
                schema.SkyRLRecipe.from_document(document)
            assert f"{path}:" in str(error.value)
            assert owner in str(error.value)
    for path, message in REMOVED.items():
        document = base.to_skyrl()
        set_path(document, path, {})
        with pytest.raises(ValueError) as error:
            schema.SkyRLRecipe.from_document(document)
        assert f"{path}: {message}" in str(error.value)
    removed = {"harbor": {"max_episodes": 1}}
    for construct in (
        lambda: schema.RecipePatch(terminal_bench=removed),
        lambda: schema.SkyRLRecipe(context_budget=base.context_budget, terminal_bench=removed),
        lambda: schema.SkyRLRecipe.model_validate_json(json.dumps({**base.to_skyrl(), "terminal_bench": removed})),
    ):
        with pytest.raises(ValueError, match=r"use context_budget.max_turns; Harbor reads max_turns"):
            construct()
    for following in ("hf_save_interval", "micro_forward_batch_size_per_gpu"):
        with pytest.raises(ValueError, match=following):
            edited.with_settings([f"trainer.{following}=null"])


def test_artifact_draft_identity_is_validated_after_parts_and_settings(base_recipe):
    base = base_recipe
    uri_part = schema.RecipePatch.from_document(
        {"generator": {"speculative_decoding": {"model": {"source_uri": "s3://models/draft"}}}}
    )
    artifact = schema.SkyRLRecipe.combine(
        base=base,
        uri=uri_part,
        details=schema.RecipePatch.from_document(
            {
                "generator": {
                    "speculative_decoding": {
                        "method": "eagle3",
                        "model": {"source_identity": "sha256:" + "a" * 64},
                        "num_speculative_tokens": 3,
                    }
                }
            }
        ),
    )
    assert artifact.to_skyrl()["generator"]["speculative_decoding"] == {
        "method": "eagle3",
        "model": {"source_uri": "s3://models/draft", "source_identity": "sha256:" + "a" * 64},
        "num_speculative_tokens": 3,
    }
    with pytest.raises(ValueError, match="artifact sources require sha256"):
        schema.SkyRLRecipe.combine(base=base, uri=uri_part)
    with pytest.raises(ValueError, match="artifact sources require sha256"):
        artifact.with_settings(["generator.speculative_decoding.model.source_identity=author-identity"])
    for source, identity in (("hf://org/draft", "b" * 40), ("/local/draft", "author-identity")):
        changed = artifact.with_settings(
            [
                f"generator.speculative_decoding.model.source_uri={source}",
                f"generator.speculative_decoding.model.source_identity={identity}",
            ]
        )
        assert changed.to_skyrl()["generator"]["speculative_decoding"]["model"] == {
            "source_uri": source,
            "source_identity": identity,
        }
