# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from taskforge.llm.client import Pool
from taskforge.loop.policy import POLICY
from taskforge.queue.config import LaptopGlm, ParallelKeyEnv, RelayGlm, load_run_config, run_config
from taskforge.sandbox.factories import MachineHost
from taskforge.validate.adversary import AdversaryRole

EXAMPLE = Path(__file__).parents[2] / "docs" / "policy.example.json"


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


def test_the_committed_example_is_an_unattended_iris_run_on_the_bulk_pool_with_the_first_run_policy():
    config = load_run_config(EXAMPLE)

    assert config.host is MachineHost.IRIS
    assert config.glm == RelayGlm(relay_job="/muchanem/glm53-relay-rno2a", token_env="GLM_API_TOKEN", pool=Pool.BULK)
    assert config.web == ParallelKeyEnv("PARALLEL_KEY")
    assert config.width == 256
    validation = config.policy.validation
    assert (validation.k, validation.adversary_k) == (8, 2)
    assert set(validation.roles) == set(AdversaryRole)
    assert (validation.band.min_solve_rate, validation.band.max_solve_rate) == (0.125, 0.875)
    assert validation.sampling.max_continuations == 0
    assert [c.id for c in config.engine.conventions] == ["plain_text", "json_answer", "json_value_answer"]


def test_the_example_policy_round_trips_through_policy_json():
    config = load_run_config(EXAMPLE)

    assert POLICY.validate_json(POLICY.dump_json(config.policy)).digest == config.policy.digest


@pytest.mark.parametrize("path", [(), ("glm",), ("engine",)])
def test_every_field_is_required(path):
    obj = example()
    target = obj
    for key in path:
        target = target[key]
    removed = sorted(k for k in target if k != "kind")[0]
    del target[removed]

    with pytest.raises(ValueError, match=f"missing \\['{removed}'\\]"):
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
    obj |= {"host": "laptop", "root": "~/runs/a"}
    obj["glm"] = {"kind": "laptop", "base_url": "http://127.0.0.1:18000/v1", "token_file": "~/t.txt", "pool": "high"}

    config = run_config(obj)

    assert config.glm == LaptopGlm("http://127.0.0.1:18000/v1", Path("~/t.txt").expanduser(), Pool.HIGH)
    assert config.root == Path("~/runs/a").expanduser()


def test_an_iris_root_is_relative_to_the_attempt_output_dir():
    obj = example()
    obj["root"] = "/abs/run"

    with pytest.raises(ValueError, match="relative to \\$IRIS_OUTPUT_DIR"):
        run_config(obj)
