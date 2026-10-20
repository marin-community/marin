import json
from itertools import permutations
import pickle

import pytest
from pydantic import ValidationError

import marinskyrl.recipe_schema as schema
from marinskyrl.recipe_schema import ContextBudget
from marinskyrl.recipe_schema.ownership import DERIVED_PATHS, LAUNCH_PATHS, REMOVED
from marinskyrl.recipe_schema.documents import set_path


def test_public_recipe_round_trip_preserves_parts_and_reports_owned_paths():
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
        schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"schedule": {}, "enabled": False}))
    )
    assert changed.to_skyrl()["generator"] == {
        "engine_init_kwargs": {"schedule": {"windows": [1, 2]}, "enabled": False}
    }
    numeric = schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"samples": [1], "weight": 1}))
    equivalent = schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"samples": [1.0], "weight": 1.0}))
    schema.SkyRLRecipe.combine(base=recipe, integer=numeric, decimal=equivalent)
    for invalid in (float("nan"), float("inf"), object()):
        with pytest.raises(ValueError, match="finite JSON value"):
            schema.SkyRLRecipe(
                context_budget=context, generator=schema.Generator(engine_init_kwargs={"schedule": [invalid]})
            )
    routes = schema.SkyRLRecipe(
        context_budget=context, generator=schema.Generator(engine_init_kwargs={"math": 1.0, "code": 1.0, "chat": 1.0})
    )
    updated = routes.merge(schema.RecipePatch(trainer=schema.Trainer(max_steps=2)))
    assert list(updated.to_skyrl()["generator"]["engine_init_kwargs"]) == ["math", "code", "chat"]
    base = schema.SkyRLRecipe(
        context_budget=context,
        generator=recipe.generator,
        data=schema.Data(sampling=schema.Sampling(domain_weights={"math": 1.0, "code": 1, "chat": 1.0})),
        teacher_routing={
            "domains": schema.TeacherRouting(
                revision="routes-v1", routes={"math": schema.TeacherRoute(teacher="local", weight=1)}
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
    assert "hf_save_interval" not in schema.Trainer().model_dump(mode="json")
    base = schema.SkyRLRecipe(
        context_budget=ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=2),
        data=schema.Data(sampling=schema.Sampling(domain_weights={"math": 2, "code": 1})),
    )
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
    computed = schema.RecipePatch(trainer=schema.Trainer(ckpt_interval=4))
    options = schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"buckets": [1, 2]}))
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
    assert (
        schema.SkyRLRecipe.combine(
            base=base, empty=schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"unused": {}}))
        )
        == base
    )
    with_parent = base.merge(schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"parent": None})))
    nested_empty = schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"parent": {"child": {}}}))
    assert schema.SkyRLRecipe.combine(base=with_parent, empty=nested_empty) == with_parent
    for first, second in (
        ({"buckets": [1, 2]}, {"buckets": [1, 3]}),
        ({"parent": None}, {"parent": {"child": 2}}),
        ({"enabled": True}, {"enabled": 1}),
    ):
        conflicts = (
            ("first", schema.RecipePatch(generator=schema.Generator(engine_init_kwargs=first))),
            ("second", schema.RecipePatch(generator=schema.Generator(engine_init_kwargs=second))),
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
    for paths, owner in ((LAUNCH_PATHS, "launch document"), (DERIVED_PATHS, "context_budget")):
        for path in paths:
            document = base.to_skyrl()
            set_path(document, path, "authored-sentinel")
            with pytest.raises(ValueError) as error:
                schema.SkyRLRecipe.from_document(document)
            if path.startswith(("terminal_bench.", "environment.task_sessions.", "generator.engine_init_kwargs.")):
                assert f"{path}:" in str(error.value)
                assert owner in str(error.value)
            else:
                assert isinstance(error.value, ValidationError)
                assert any(
                    item["loc"] == tuple(path.split(".")) and item["type"] == "extra_forbidden"
                    for item in error.value.errors()
                )
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
        with pytest.raises(ValueError, match="use context_budget.max_turns; Harbor reads max_turns"):
            construct()
    for following in ("hf_save_interval", "micro_forward_batch_size_per_gpu"):
        with pytest.raises(ValueError, match=following):
            edited.with_settings([f"trainer.{following}=null"])
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
