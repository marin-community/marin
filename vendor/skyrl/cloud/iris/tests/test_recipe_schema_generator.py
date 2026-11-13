import copy
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import skyrlrecipe as schema
from skyrlrecipe.model import Section, thaw
from skyrlrecipe.ownership import OWNER_MESSAGES

from scripts import generate_recipe_schema as generator
from scripts.recipe_schema_sidecar import OPEN

ROOT = Path(__file__).resolve().parents[3]

BASE = """defaults:
  - _self_
trainer:
  placement:
    # The all-role colocation flag.
    colocate_all: true
    # The policy's physical footprint.
    policy_num_nodes: null # Filled from allocation when unset.
  algorithm:
    cispo:
      cispo_eps_clip_high: 5
  policy:
    model:
      lora:
        dropout: 0
  critic:
    model:
      lora:
        dropout: 0
  ckpt_interval: 2
"""
SIDECAR = """TYPES = {
    'trainer.placement.policy_num_nodes': 'PositiveInt | None',
    'trainer.policy.model.lora.dropout': 'Annotated[int | float, Field(ge=0, le=1)]',
    'trainer.critic.model.lora.dropout': 'Annotated[int | float, Field(ge=0, le=1)]',
}
OPEN = frozenset()
UNDECLARED = {}
NAMES = {"trainer.policy.model": "PolicyModel", "trainer.critic.model": "CriticModel", "trainer.policy.model.lora": "PolicyLora", "trainer.critic.model.lora": "CriticLora"}
"""


def _skipped(path, value):
    return path in frozenset(OWNER_MESSAGES) or generator.following_source(value) is not None


def _author_document(document):
    authored = {}

    def visit(mapping, target, prefix=""):
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if _skipped(path, value):
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
        assert (generator.OUTPUT_DIR / name).read_text() == generated, f"regenerate {name}"
    sources = generator.source_documents(generator.CONFIG_DIR)
    defaults = schema.RecipePatch()

    def compare(mapping, actual, prefix=""):
        for key, expected in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if _skipped(path, expected):
                continue
            if path in OPEN and expected:
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
        assert schema.SkyRLRecipe.from_document(document).to_skyrl() == document, path.name


def test_generator_preserves_observed_types_comments_and_sidecar_drift(tmp_path, monkeypatch):
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
    owned = frozenset()
    code = generator.render_sections(generator.render_schema(sources, sidecar, owned))
    module = ModuleType("skyrlrecipe._fixture")
    module.__package__ = "skyrlrecipe"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(code, "generated_fixture.py", "exec"), vars(module))
    model = module.RecipeSections
    for value in (5, 5.0):
        document = {"trainer": {"algorithm": {"cispo": {"cispo_eps_clip_high": value}}}}
        recipe = model.model_validate_json(json.dumps(document))
        assert type(recipe.to_skyrl()["trainer"]["algorithm"]["cispo"]["cispo_eps_clip_high"]) is type(value)
    assert '"The policy\'s physical footprint.\\nFilled from allocation when unset."' in code
    assert "A file header, not a field." not in code
    unnamed = copy.deepcopy(sidecar)
    del unnamed["NAMES"]["trainer.critic.model"]
    with pytest.raises(ValueError, match="trainer.critic.model"):
        generator.render_schema(sources, unnamed, owned)
    for name, key, value in (
        ("TYPES", "trainer.misspelled", "int"),
        ("UNDECLARED", "trainer.ckpt_interval", ("int", 99)),
    ):
        changed = copy.deepcopy(sidecar)
        changed[name][key] = value
        with pytest.raises(ValueError):
            generator.render_schema(sources, changed, owned)
