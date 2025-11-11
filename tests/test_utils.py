# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
from collections.abc import Callable
from dataclasses import dataclass

import draccus
import pytest
import ray
from marin.utils import asdict_excluding, remove_tpu_lockfile_on_exit


def test_remove_tpu_lockfile_on_exit_works_with_ray_remote(ray_tpu_cluster):
    @ray.remote
    @remove_tpu_lockfile_on_exit
    def test_fn():
        return 1

    assert ray.get(test_fn.remote()) == 1


def parameterize_with_configs(pattern: str, config_path: str | None = None) -> Callable:
    """
    A decorator to parameterize a test function with configuration files.

    Args:
        pattern (str): A glob pattern to match configuration files.
        config_path (Optional[str]): The base path to look for config files.
                                    If None, uses "../config" relative to the test file.

    Returns:
        Callable: A pytest.mark.parametrize decorator that provides config files to the test function.

    """
    test_path = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(test_path, "..", "config")
    configs = glob.glob(os.path.join(config_path, "**", pattern), recursive=True)
    return pytest.mark.parametrize("config_file", configs, ids=lambda x: f"{os.path.basename(x)}")


def check_load_config(config_class: type, config_file: str) -> None:
    """
    Attempt to load and parse a configuration file using a specified config class.

    Args:
        config_class (Type): The configuration class to use for parsing.
        config_file (str): Path to the configuration file to be parsed.

    Raises:
        Exception: If the configuration file fails to parse.
    """
    try:
        draccus.parse(config_class, config_file, args=[])
    except Exception as e:
        raise Exception(f"failed to parse {config_file}") from e


def skip_if_module_missing(module: str):
    def try_import_module(module):
        try:
            __import__(module)
        except ImportError:
            return False
        else:
            return True

    return pytest.mark.skipif(not try_import_module(module), reason=f"{module} not installed")


def skip_in_ci(fn_or_msg):
    if isinstance(fn_or_msg, str):

        def decorator(fn):
            return pytest.mark.skipif("CI" in os.environ, reason=fn_or_msg)(fn)

        return decorator

    return pytest.mark.skipif("CI" in os.environ, reason="skipped in CI")(fn_or_msg)


@dataclass
class NestedConfig:
    name: str
    value: int
    runtime_env: str = "test"


def test_asdict_excluding_simple():
    """Test asdict_excluding with a simple dataclass."""
    config = NestedConfig(name="test", value=42)
    result = asdict_excluding(config, exclude={"runtime_env"})
    assert result == {"name": "test", "value": 42}
    assert "runtime_env" not in result


def test_asdict_excluding_invalid():
    """Test asdict_excluding with non-dataclass input."""
    with pytest.raises(ValueError, match="Only dataclasses are supported"):
        asdict_excluding({"key": "value"}, exclude=set())
