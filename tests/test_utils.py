import glob
import os
from collections.abc import Callable

import draccus
import pytest
import ray

from marin.utils import remove_tpu_lockfile_on_exit


def setup_module(module):
    ray.init("local", num_cpus=8, ignore_reinit_error=True)


def teardown_module(module):
    ray.shutdown()


def test_remove_tpu_lockfile_on_exit_works_with_ray_remote():
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


def check_wandb_api_functionality():
    from marin.utilities.metrics_utils import get_flops_usage_over_period, get_wandb_run_metrics

    # Test get_flops_usage_over_period
    flops_usage = get_flops_usage_over_period(num_days=7)
    assert isinstance(flops_usage, dict)
    assert "num_days" in flops_usage
    assert "num_runs_counted" in flops_usage
    assert "total_flops" in flops_usage

    # Test get_wandb_run_metrics with a specific run ID
    TEST_RUN_ID = "exp446-fineweb-edu-1.4b-9e4be7"
    metrics = get_wandb_run_metrics(run_id=TEST_RUN_ID)
    assert isinstance(metrics, dict)
    assert "run_id" in metrics
    assert metrics["run_id"] == TEST_RUN_ID
