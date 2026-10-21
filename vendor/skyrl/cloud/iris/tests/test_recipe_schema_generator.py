import copy
import json
from pathlib import Path

import pytest

import marinskyrl.recipe_schema as schema
from marinskyrl.recipe_schema.model import Section, thaw
from marinskyrl.recipe_schema.ownership import DERIVED_PATHS, LAUNCH_PATHS
from scripts import generate_recipe_schema as generator

ROOT = Path(__file__).resolve().parents[3]

OUTPUT_PATH_RECIPES = frozenset(
    {
        "128GPU_80B_A3B_next_cp1",
        "32GPU_qwen3_coder_30b_a3b_ep4",
        "32GPU_qwen3_coder_30b_a3b_ep4_nooffload",
        "56GPU_qwen3_8b",
        "64GPU_qwen3_6_35b_a3b",
        "opencode_smoke_literal",
        "tasktrove_dq_sweep_30b",
        "tasktrove_dq_sweep_30b_cp6",
        "tasktrove_dq_sweep_30b_ncclnet",
        "tasktrove_dq_sweep_30b_terminus2",
    }
)
BUNDLED_NORMALIZATION = {
    "trainer.ckpt_path": OUTPUT_PATH_RECIPES,
    "trainer.export_path": OUTPUT_PATH_RECIPES,
    "trainer.seed": frozenset(
        {
            "nemotron_ultra_rlvr_acceptance",
            "snowball_ultra_rlvr1_colocated64",
            "snowball_ultra_rlvr1_split64",
            "snowball_ultra_rlvr2_colocated64",
            "snowball_ultra_rlvr2_split64",
        }
    ),
    "trainer.max_ckpts_to_keep": frozenset(
        {
            "snowball_mopd_ultra_32k",
            "snowball_mopd_ultra_async_32k_smoke",
            "snowball_ultra_rlvr1_colocated64",
            "snowball_ultra_rlvr1_split64",
            "snowball_ultra_rlvr2_colocated64",
            "snowball_ultra_rlvr2_split64",
        }
    ),
    "policy_chat_template": frozenset({"delphi_math_rl", "delphi_math_rl_ifeval"}),
    "trainer.ref.model.path": OUTPUT_PATH_RECIPES - {"56GPU_qwen3_8b", "opencode_smoke_literal"},
    "generator.engine_init_kwargs.served_model_name": frozenset(
        {"nemotron_ultra_rlvr_acceptance", "opencode_smoke_literal"}
    ),
}
BASE = """defaults:
  - _self_
data:
  sampling:
    domain_weights:
      math-reasoning: 1
      code: 2.0
trainer:
  callbacks: null
  strategy: megatron
  max_steps: 0
  placement:
    # The all-role colocation flag.
    colocate_all: true
    # The policy's physical footprint.
    policy_num_nodes: null # Filled from allocation when unset.
  algorithm:
    cispo:
      cispo_eps_clip_high: 5
    ftpo: {}
  policy:
    model:
      lora:
        dropout: 0
  critic:
    model:
      lora:
        dropout: 0
  ckpt_interval: 2
  hf_save_interval: ${trainer.ckpt_interval}
  micro_train_batch_size_per_gpu: 2
  micro_forward_batch_size_per_gpu: ${trainer.micro_train_batch_size_per_gpu}
generator:
  engine_init_kwargs: {}
  adapter: null
  sampling_params:
    temperature: 1
"""
SIDECAR = """TYPES = {
    'trainer.callbacks': 'tuple[CheckpointCallback, ...] | None',
    'data.sampling.domain_weights': 'NumberMap',
    'trainer.strategy': 'Literal["megatron"]',
    'trainer.max_steps': 'NonNegativeInt',
    'trainer.placement.policy_num_nodes': 'PositiveInt | None',
    'trainer.policy.model.lora.dropout': 'Annotated[int | float, Field(ge=0, le=1)]',
    'trainer.critic.model.lora.dropout': 'Annotated[int | float, Field(ge=0, le=1)]',
}
OPEN = frozenset({'generator.engine_init_kwargs'})
UNDECLARED = {
    'trainer.algorithm.ftpo.lambda_mse': ('int | float', ...),
    'generator.adapter.name': ('str', ...),
    'generator.adapter.parameters': ('Parameters | None', None),
    'generator.adapter.parameters.count': ('int', ...),
}
NAMES = {"trainer.policy.model": "PolicyModel", "trainer.critic.model": "CriticModel", "trainer.policy.model.lora": "PolicyLora", "trainer.critic.model.lora": "CriticLora"}
"""


def _author_document(document):
    authored = {}

    def visit(mapping, target, prefix=""):
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in DERIVED_PATHS | LAUNCH_PATHS or generator.following_source(value):
                continue
            if isinstance(value, dict):
                target[key] = {}
                visit(value, target[key], path)
            else:
                target[key] = value

    visit(document, authored)
    return authored


def test_repository_generation_matches_hydra_author_defaults_and_all_group_options():
    for name, generated in generator.generate().items():
        assert (ROOT / "marinskyrl/recipe_schema" / name).read_text() == generated, f"regenerate {name}"
    sources = generator.source_documents(generator.CONFIG_DIR)
    defaults = schema.RecipePatch()

    def compare(mapping, actual, prefix=""):
        for key, expected in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in DERIVED_PATHS | LAUNCH_PATHS or generator.following_source(expected):
                continue
            value = getattr(actual, key)
            if isinstance(expected, dict) and isinstance(value, Section):
                compare(expected, value, path)
            else:
                assert thaw(value) == expected, path

    compare(sources.base, defaults)
    observed = set()
    for group, options in sources.groups.items():
        for name, option in options.items():
            observed.add((group.partition("@")[0], name))
            document = _author_document(option)
            assert schema.RecipePatch.from_document(document).to_skyrl() == document, (group, name)
    assert observed == {(path.parent.name, path.stem) for path in generator.CONFIG_DIR.glob("*/*.yaml")}
    paths = sorted((ROOT / "cloud/iris/configs").glob("*.yaml"))
    recipes = generator.recipe_documents(ROOT / "cloud/iris/configs", generator.CONFIG_DIR, sources.groups)
    for path, document in zip(paths, recipes, strict=True):
        for key, names in BUNDLED_NORMALIZATION.items():
            if path.stem in names:
                parent = document
                for part in key.split(".")[:-1]:
                    parent = parent.get(part, {})
                parent.pop(key.rsplit(".", 1)[-1], None)
        assert schema.SkyRLRecipe.from_document(document).to_skyrl() == document, path.name


def test_generator_preserves_observed_types_comments_following_nulls_and_sidecar_drift(tmp_path):
    configs = tmp_path / "config"
    configs.mkdir()
    (configs / "ppo_base_config.yaml").write_text(BASE)
    group = configs / "algorithm_recipe"
    group.mkdir()
    (group / "cispo.yaml").write_text(
        "# @package trainer.algorithm\n# A file header, not a field.\ncispo:\n  cispo_eps_clip_high: 5.0\n"
    )
    sidecar = {}
    exec(SIDECAR, sidecar)
    sources = generator.source_documents(configs)
    owners = {"DERIVED_PATHS": frozenset(), "LAUNCH_PATHS": frozenset()}
    code = generator.render_sections(sources.base, sidecar, owners, sources.comments, sources.groups)
    namespace = {"__name__": "marinskyrl.recipe_schema._fixture", "__package__": "marinskyrl.recipe_schema"}
    exec(compile(code, "generated_fixture.py", "exec"), namespace)
    model = namespace["RecipeSections"]
    for value in (5, 5.0):
        document = {"trainer": {"algorithm": {"cispo": {"cispo_eps_clip_high": value}}}}
        recipe = model.model_validate_json(json.dumps(document))
        assert type(recipe.to_skyrl()["trainer"]["algorithm"]["cispo"]["cispo_eps_clip_high"]) is type(value)
    assert '"The policy\'s physical footprint.\\nFilled from allocation when unset."' in code
    assert "A file header, not a field." not in code
    unnamed = copy.deepcopy(sidecar)
    del unnamed["NAMES"]["trainer.critic.model"]
    with pytest.raises(ValueError, match="trainer.critic.model"):
        generator.render_sections(sources.base, unnamed, owners, sources.comments, sources.groups)
    for name, key, value in (
        ("TYPES", "trainer.misspelled", "int"),
        ("UNDECLARED", "trainer.ckpt_interval", ("int", 99)),
    ):
        changed = copy.deepcopy(sidecar)
        changed[name][key] = value
        with pytest.raises(ValueError):
            generator.render_sections(sources.base, changed, owners, sources.comments, sources.groups)
