import dataclasses
from unittest.mock import patch

import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.main import train_lm
from levanter.trainer import TrainerConfig

from marin.resources import TpuPodConfig
from marin.training.training import (
    TrainLmOnPodConfig,
    _doublecheck_paths,
)


@pytest.fixture
def trainer_config():
    """Create a basic trainer config for tests."""
    return TrainerConfig(
        id="test-run",
        checkpointer=CheckpointerConfig(),
        ray=RayConfig(),
    )


@dataclasses.dataclass
class MockDataConfig:
    """Mock data config for testing."""

    cache_dir: str


@dataclasses.dataclass
class MockNestedDataConfig:
    """Mock nested data config for testing."""

    cache_dir: str
    subdir: dict


@dataclasses.dataclass
class MockNestedConfig:
    """Mock nested config for testing."""

    path: str


@pytest.mark.parametrize("tpu_type", [None, "v3-8"])
def test_lm_config_with_local_paths(trainer_config, tpu_type):
    """Test that local paths are allowed when running locally."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-central1"

        # Create a config with local paths
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="local/path"),
                trainer=trainer_config,
            ),
            resources=TpuPodConfig(tpu_type=tpu_type),
        )

        # This should not raise an exception
        _doublecheck_paths(config)


def test_lm_config_with_gcs_paths_same_region(trainer_config):
    """Test that GCS paths in the same region are allowed."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-central1"

        # Create a config with GCS paths in the same region
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=TpuPodConfig(tpu_type="v3-8"),  # TPU mode
        )

        # This should not raise an exception
        _doublecheck_paths(config)


def test_lm_config_with_gcs_paths_different_region(trainer_config):
    """Test that GCS paths in a different region raise an exception."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with GCS paths in a different region
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=TpuPodConfig(tpu_type="v3-8"),  # TPU mode
        )

        # This should raise an exception
        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)


# New tests for JAX_COMPILATION_CACHE_DIR
import os
import unittest
from unittest.mock import MagicMock, call

from levanter.config import WandbConfig # WandbConfig needed for TrainerConfig
from marin.resources import CpuOnlyConfig # For a simple ResourceConfig
from marin.training.training import run_levanter_train_lm


class TestJaxCompilationCacheDir(unittest.TestCase): # Renamed class for clarity
    def _get_minimal_config(self):
        """Helper to create a minimal TrainLmOnPodConfig for testing."""
        return TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                trainer=TrainerConfig(
                    id="test-jax-compilation-cache",
                    checkpointer=CheckpointerConfig(base_path="gs://dummy/checkpoints"),
                    ray=RayConfig(auto_start_cluster=False, start_workers=False),
                    wandb=WandbConfig(mode="disabled"),
                    mp=None, # type: ignore
                    tracker=None, # type: ignore
                ),
                data=None, # type: ignore
                model=None, # type: ignore
                optimizer=None, # type: ignore
                project_name = "test-project",
                hf_save_path = "gs://dummy/hf_save",
                hf_upload = None,
                hf_save_steps = 100,
                initialize_from_hf = "",
                load_checkpoint = None,
                load_dataset_params = None,
                loss_fn = None, # type: ignore
                max_completion_len = None, # type: ignore
                max_eval_completion_len = None, # type: ignore
                max_examples = None, # type: ignore
                num_train_steps = 1,
                num_warmup_steps = 0,
                per_device_eval_parallelism = 1,
                per_device_train_parallelism = 1,
                save_interval = 1.0,
                skip_eval_on_resume = False,
                skip_final_eval = False,
                skip_initial_eval = False,
                steps_per_eval = 1,
                steps_per_save = 1,
                stream_eval_to_wandb = False,
                train_batch_size = 1,
                train_dl_workers = 0,
                use_cpu_for_eval = False,
                use_wandb = False,
                validation_batch_size = 1,
                validation_dl_workers = 0,
                validation_max_examples = None, # type: ignore
                validation_steps_per_eval = 1,
                wandb = None, # type: ignore
            ),
            resources=CpuOnlyConfig(),
            output_path="gs://dummy/output",
        )

    @patch("marin.training.training._doublecheck_paths")
    @patch("marin.training.training._update_config_to_use_out_path", side_effect=lambda x: x)
    @patch("marin.training.training._suppress_ray_config", side_effect=lambda x: x)
    @patch("marin.training.training._enforce_run_id", side_effect=lambda x: x)
    @patch("marin.training.training._check_for_wandb_key")
    @patch("levanter.infra.cli_helpers.load_config")
    @patch("marin.training.training.ray.get")
    @patch("marin.training.training.ray.remote")
    @patch("marin.training.training.train_lm")
    @patch("marin.training.training.logger")
    @patch.dict(os.environ, {"MARIN_PREFIX": "/test/prefix"}, clear=True)
    def test_marin_prefix_set_jax_compilation_cache_dir_configured(
        self,
        mock_logger,
        mock_train_lm_main,
        mock_ray_remote,
        mock_ray_get,
        mock_load_config,
        mock_check_wandb,
        mock_enforce_run_id,
        mock_suppress_ray,
        mock_update_config,
        mock_doublecheck_paths,
    ):
        mock_ray_remote.return_value = lambda func: func
        mock_ray_get.side_effect = lambda x: x

        mock_launch_config_instance = MagicMock()
        mock_launch_config_instance.env_for_accel.return_value = {}
        mock_load_config.return_value = mock_launch_config_instance

        test_config = self._get_minimal_config()

        with patch.object(CpuOnlyConfig, 'with_env_vars', wraps=test_config.resources.with_env_vars) as mock_with_env_vars:
            test_config.resources.with_env_vars = mock_with_env_vars

            run_levanter_train_lm(test_config)

            mock_train_lm_main.main.assert_called_once()
            mock_logger.info.assert_any_call("JAX compilation cache enabled at: /test/prefix/compilation-cache")

            found_jax_cache_in_env_vars = False
            for call_args in mock_with_env_vars.call_args_list:
                args, _ = call_args
                if args:
                    env_dict = args[0]
                    if env_dict.get("JAX_COMPILATION_CACHE_DIR") == "/test/prefix/compilation-cache":
                        found_jax_cache_in_env_vars = True
                        break
            self.assertTrue(found_jax_cache_in_env_vars, "JAX_COMPILATION_CACHE_DIR was not set correctly in env for with_env_vars")

    @patch("marin.training.training._doublecheck_paths")
    @patch("marin.training.training._update_config_to_use_out_path", side_effect=lambda x: x)
    @patch("marin.training.training._suppress_ray_config", side_effect=lambda x: x)
    @patch("marin.training.training._enforce_run_id", side_effect=lambda x: x)
    @patch("marin.training.training._check_for_wandb_key")
    @patch("levanter.infra.cli_helpers.load_config")
    @patch("marin.training.training.ray.get")
    @patch("marin.training.training.ray.remote")
    @patch("marin.training.training.train_lm")
    @patch("marin.training.training.logger")
    @patch.dict(os.environ, {}, clear=True)
    def test_marin_prefix_not_set_jax_compilation_cache_dir_not_configured(
        self,
        mock_logger,
        mock_train_lm_main,
        mock_ray_remote,
        mock_ray_get,
        mock_load_config,
        mock_check_wandb,
        mock_enforce_run_id,
        mock_suppress_ray,
        mock_update_config,
        mock_doublecheck_paths,
    ):
        mock_ray_remote.return_value = lambda func: func
        mock_ray_get.side_effect = lambda x: x

        mock_launch_config_instance = MagicMock()
        mock_launch_config_instance.env_for_accel.return_value = {}
        mock_load_config.return_value = mock_launch_config_instance

        test_config = self._get_minimal_config()

        with patch.object(CpuOnlyConfig, 'with_env_vars', wraps=test_config.resources.with_env_vars) as mock_with_env_vars:
            test_config.resources.with_env_vars = mock_with_env_vars

            run_levanter_train_lm(test_config)

            mock_train_lm_main.main.assert_called_once()
            mock_logger.warning.assert_any_call("MARIN_PREFIX environment variable not set. JAX compilation cache will not be configured.")

            found_jax_cache_in_env_vars = False
            for call_args in mock_with_env_vars.call_args_list:
                args, _ = call_args
                if args:
                    env_dict = args[0]
                    if "JAX_COMPILATION_CACHE_DIR" in env_dict:
                        found_jax_cache_in_env_vars = True
                        break
            self.assertFalse(found_jax_cache_in_env_vars, "JAX_COMPILATION_CACHE_DIR was unexpectedly set in env for with_env_vars")

    @patch("marin.training.training._doublecheck_paths")
    @patch("marin.training.training._update_config_to_use_out_path", side_effect=lambda x: x)
    @patch("marin.training.training._suppress_ray_config", side_effect=lambda x: x)
    @patch("marin.training.training._enforce_run_id", side_effect=lambda x: x)
    @patch("marin.training.training._check_for_wandb_key")
    @patch("levanter.infra.cli_helpers.load_config")
    @patch("marin.training.training.ray.get")
    @patch("marin.training.training.ray.remote")
    @patch("marin.training.training.train_lm")
    @patch("marin.training.training.logger")
    # Simulate JAX_COMPILATION_CACHE_DIR being pre-set in the environment the function reads from
    @patch.dict(os.environ, {"JAX_COMPILATION_CACHE_DIR": "/custom/jax/cache/from_os_env"}, clear=True)
    def test_jax_compilation_cache_dir_already_set_in_os_environ(
        self,
        mock_logger,
        mock_train_lm_main,
        mock_ray_remote,
        mock_ray_get,
        mock_load_config,
        mock_check_wandb,
        mock_enforce_run_id,
        mock_suppress_ray,
        mock_update_config,
        mock_doublecheck_paths,
    ):
        mock_ray_remote.return_value = lambda func: func
        mock_ray_get.side_effect = lambda x: x

        mock_launch_config_instance = MagicMock()
        # Simulate the case where the default_env from LevanterLaunchConfig also has it set
        # This is to ensure that even if it's in default_env, os.environ takes precedence if
        # the main logic `if "JAX_COMPILATION_CACHE_DIR" not in env:` relies on `env`
        # which is first populated by os.environ then merged with default_env.
        # However, the current logic in `run_levanter_train_lm` is:
        # 1. env = _add_default_env_variables(config_runtime_env, default_levanter_env)
        # 2. _check_for_wandb_key(env)
        # 3. env = _add_run_env_variables(env) (adds GIT_COMMIT etc from os.environ if not in env)
        # 4. NEW JAX CACHE LOGIC (checks os.environ directly for MARIN_PREFIX, adds to env if JAX_COMPILATION_CACHE_DIR not in env)
        # So, if JAX_COMPILATION_CACHE_DIR is in `config.resources.runtime_env.env_vars` OR
        # in `default_launch_config.env_for_accel`, it will be in `env` before our new block.

        # Test 1: JAX_COMPILATION_CACHE_DIR is in os.environ, which populates initial env
        # This is covered by @patch.dict above.
        # The `env` dict passed to _add_default_env_variables will have it from os.environ.

        test_config = self._get_minimal_config()
        # We want to ensure that the value from os.environ is present.
        # The `env` in `run_levanter_train_lm` is built from:
        # `config.resources.runtime_env.get("env_vars", {})` then `default_launch_config.env_for_accel`
        # then `_add_run_env_variables` (which adds from os.environ).
        # The test for "already set" means it's in `env` *before* our specific cache logic block.

        # To correctly test this, we should ensure it's in one of the sources for `env`
        # *before* our logic block.
        # Option A: it's in config.resources.runtime_env.env_vars
        # Option B: it's in default_launch_config.env_for_accel
        # Option C: it's added by _add_run_env_variables (but this is for specific vars like GIT_COMMIT)

        # Let's use Option A: pre-set it in the config's runtime_env
        pre_set_value = "/custom/jax/cache/from_config_runtime_env"
        test_config.resources.runtime_env = {"env_vars": {"JAX_COMPILATION_CACHE_DIR": pre_set_value}}
        mock_launch_config_instance.env_for_accel.return_value = {} # Ensure no conflict from here
        mock_load_config.return_value = mock_launch_config_instance

        with patch.object(CpuOnlyConfig, 'with_env_vars', wraps=test_config.resources.with_env_vars) as mock_with_env_vars:
            test_config.resources.with_env_vars = mock_with_env_vars

            run_levanter_train_lm(test_config)

            mock_train_lm_main.main.assert_called_once()

            # Ensure the specific JAX cache setting logs were NOT called
            for log_call in mock_logger.info.call_args_list:
                args, _ = log_call
                self.assertNotIn("JAX compilation cache enabled at:", args[0])
            for log_call in mock_logger.warning.call_args_list:
                args, _ = log_call
                self.assertNotIn("MARIN_PREFIX environment variable not set", args[0])

            # Check that with_env_vars was called and JAX_COMPILATION_CACHE_DIR was the original one
            found_jax_cache_in_env_vars = False
            correct_value_found = False
            for call_args in mock_with_env_vars.call_args_list:
                args, _ = call_args
                if args:
                    env_dict = args[0]
                    if "JAX_COMPILATION_CACHE_DIR" in env_dict:
                        found_jax_cache_in_env_vars = True
                        if env_dict["JAX_COMPILATION_CACHE_DIR"] == pre_set_value:
                            correct_value_found = True
                        break
            self.assertTrue(found_jax_cache_in_env_vars, "JAX_COMPILATION_CACHE_DIR was not in env for with_env_vars")
            self.assertTrue(correct_value_found, f"JAX_COMPILATION_CACHE_DIR was not the pre-set config value '{pre_set_value}'")


if __name__ == "__main__":
    unittest.main()


def test_lm_config_with_allowed_out_of_region_paths(trainer_config):
    """Test that paths in allow_out_of_region are allowed to be in different regions."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with GCS paths in a different region but allowed
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=TpuPodConfig(tpu_type="v3-8"),  # TPU mode
            allow_out_of_region=("data.cache_dir",),
        )

        # This should not raise an exception
        _doublecheck_paths(config)


def test_recursive_path_checking(trainer_config):
    """Test that paths are checked recursively in nested structures."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with nested GCS paths in a different region
        nested_data = MockNestedDataConfig(
            cache_dir="gs://bucket/path", subdir={"file": "gs://bucket/other/path", "list": ["gs://bucket/another/path"]}
        )

        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=nested_data,
                trainer=trainer_config,
            ),
            resources=TpuPodConfig(tpu_type="v3-8"),  # TPU mode
        )

        # This should raise an exception
        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)


def test_dataclass_recursive_checking(trainer_config):
    """Test that paths are checked recursively in dataclass objects."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with a dataclass containing a GCS path
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=MockNestedConfig(path="gs://bucket/path")),  # type: ignore
                trainer=trainer_config,
            ),
            resources=TpuPodConfig(tpu_type="v3-8"),  # TPU mode
        )

        # This should raise an exception
        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)
