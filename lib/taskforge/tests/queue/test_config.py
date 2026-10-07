# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from taskforge.llm.client import Pool
from taskforge.queue.config import LaptopGlm, load_run_config, run_config

EXAMPLE = Path(__file__).parents[2] / "docs" / "policy.example.json"


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


def test_the_committed_example_queues_on_the_bulk_pool():
    config = load_run_config(EXAMPLE)

    # User decision D3: the committed unattended configuration names the bulk pool.
    assert config.glm.pool is Pool.BULK


@pytest.mark.parametrize(
    ("path", "removed"),
    [
        ((), "width"),
        ((), "image_cache"),
        (("glm",), "kind"),
        (("glm",), "pool"),
        (("web",), "kind"),
        (("engine",), "max_turns"),
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


def test_a_laptop_config_reads_a_token_file_path_and_a_local_root():
    obj = example()
    obj |= {"host": "laptop", "root": "~/runs/a", "image_cache": "~/images"}
    obj["glm"] = {"kind": "laptop", "base_url": "http://127.0.0.1:18000/v1", "token_file": "~/t.txt", "pool": "high"}

    config = run_config(obj)

    assert config.glm == LaptopGlm("http://127.0.0.1:18000/v1", Path("~/t.txt").expanduser(), Pool.HIGH)
    assert config.root == Path("~/runs/a").expanduser()
    assert config.image_cache == Path("~/images").expanduser()


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
