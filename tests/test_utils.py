import glob
import os

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


def parameterize_with_configs(pattern, config_path=None):
    test_path = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(test_path, "..", "config")
    configs = glob.glob(os.path.join(config_path, pattern))
    return pytest.mark.parametrize("config_file", configs, ids=lambda x: f"{os.path.basename(x)}")


def check_load_config(config_class, config_file):
    try:
        draccus.parse(config_class, config_file, args=[])
    except Exception as e:
        raise Exception(f"failed to parse {config_file}") from e
