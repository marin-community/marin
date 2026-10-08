# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from taskforge.llm.client import Pool
from taskforge.queue.config import LaptopGlm, load_run_config, run_config
from taskforge.review.rules import BandChoice, BandRule

EXAMPLE = Path(__file__).parents[2] / "docs" / "policy.example.json"


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


def test_the_committed_example_queues_on_the_bulk_pool():
    config = load_run_config(EXAMPLE)

    # User decision D3: the committed unattended configuration names the bulk pool.
    assert config.glm.pool is Pool.BULK


def test_the_committed_example_accepts_a_too_easy_task_after_one_revision_and_rejects_a_too_hard_one():
    policy = load_run_config(EXAMPLE).policy

    # The capability run labels a too-easy task with its pass rate instead of losing it; Taskforge's
    # own default rejects both kinds.
    assert policy.band_rules.too_easy == BandRule(1, BandChoice.ACCEPT)
    assert policy.band_rules.too_hard == BandRule(1, BandChoice.REJECT)
    assert (policy.validation.adversary_submissions, policy.validation.adversary_repair_submissions) == (10, 3)


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda p: p.pop("band_rules"), "missing fields \\['band_rules'\\]"),
        (lambda p: p["band_rules"]["too_easy"].update(then="note"), "note"),
        (
            lambda p: p["validation"].update(adversary_output_tokens=32768),
            "unknown fields \\['adversary_output_tokens'\\]",
        ),
        (lambda p: p["validation"].pop("adversary_submissions"), "missing fields \\['adversary_submissions'\\]"),
    ],
)
def test_a_policy_must_state_its_band_rules_and_adversary_budget(edit, problem):
    obj = example()
    edit(obj["policy"])

    with pytest.raises(ValueError, match=problem):
        run_config(obj)


@pytest.mark.parametrize(
    ("path", "removed"),
    [
        ((), "width"),
        ((), "image_cache"),
        (("glm",), "kind"),
        (("glm",), "pool"),
        (("web",), "kind"),
        (("engine",), "max_turns"),
        (("engine",), "tool_turn_timeout"),
    ],
)
def test_every_field_is_required(path, removed):
    obj = example()
    target = obj
    for key in path:
        target = target[key]
    del target[removed]

    with pytest.raises(ValueError, match=f"missing \\['{removed}'\\]"):
        run_config(obj)


def test_an_unknown_convention_type_is_a_config_error():
    obj = example()
    obj["engine"]["conventions"][0]["type"] = "no_such_convention"

    with pytest.raises(ValueError, match="unknown type 'no_such_convention'"):
        run_config(obj)


def test_a_config_cannot_carry_a_token_inline():
    obj = example()
    obj["glm"]["token"] = "secret"

    with pytest.raises(ValueError, match="unknown \\['token'\\]"):
        run_config(obj)


def test_a_relay_names_the_token_variable_not_the_token():
    obj = example()
    obj["glm"]["token_env"] = "sk-not-a-variable-name"

    with pytest.raises(ValueError, match="names an environment variable"):
        run_config(obj)


def test_the_pool_is_explicit():
    obj = example()
    obj["glm"] = {"kind": "laptop", "base_url": "http://127.0.0.1:18000/v1", "token_file": "~/token.txt"}

    with pytest.raises(ValueError, match="missing \\['pool'\\]"):
        run_config(obj)


def laptop_example() -> dict:
    obj = example()
    obj |= {"host": "laptop", "root": "/runs/a", "image_cache": "/cache/images"}
    obj["glm"] = {"kind": "laptop", "base_url": "http://127.0.0.1:18000/v1", "token_file": "/keys/t.txt", "pool": "high"}
    obj["web"] = {"kind": "key_file", "path": "/keys/parallel"}
    return obj


def test_a_laptop_config_reads_a_token_file_path_and_a_local_root():
    config = run_config(laptop_example())

    assert config.glm == LaptopGlm("http://127.0.0.1:18000/v1", Path("/keys/t.txt"), Pool.HIGH)
    assert config.root == Path("/runs/a")
    assert config.image_cache == Path("/cache/images")


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        ((), "root", "runs/a"),
        ((), "image_cache", "~/images"),
        (("glm",), "token_file", "~/t.txt"),
        (("web",), "path", "keys/parallel"),
    ],
)
def test_a_laptop_config_takes_only_absolute_local_paths(path, field, value):
    obj = laptop_example()
    target = obj
    for key in path:
        target = target[key]
    target[field] = value

    # A "~" or a relative path would mean something different for each launching shell.
    with pytest.raises(ValueError, match="is an absolute path"):
        run_config(obj)


@pytest.mark.parametrize(("host", "image_cache"), [("laptop", None), ("iris", "/images")])
def test_only_a_laptop_run_names_an_image_cache(host, image_cache):
    obj = example()
    obj |= {"host": host, "root": "run", "image_cache": image_cache}

    with pytest.raises(ValueError, match="image_cache is a directory on a laptop and null on Iris"):
        run_config(obj)


def test_an_iris_root_is_relative_to_the_attempt_output_dir():
    obj = example()
    obj["root"] = "/abs/run"

    with pytest.raises(ValueError, match="relative to \\$IRIS_OUTPUT_DIR"):
        run_config(obj)
