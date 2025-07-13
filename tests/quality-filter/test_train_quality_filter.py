import os
from dataclasses import replace

import pytest

from experiments.datashop.default_configs import default_quality_filter_train_config_kwargs
from marin.classifiers.hf.launch_ray_training import LaunchConfig, launch_training_with_ray
from tests.conftest import SINGLE_GPU_CONFIG


@pytest.mark.skipif(os.getenv("GPU_CI") != "true", reason="Skip this test if not running with a GPU in CI.")
def test_quality_filter_training_gpu(ray_cluster, text_and_label_dataset_path):
    training_config = default_quality_filter_train_config_kwargs["training_config"]
    training_config = replace(
        training_config,
        train_dataset=text_and_label_dataset_path,
        output_dir="gs://marin-us-east1/classifiers/test-quality-filter",
        run_name="test-classifier-gpu",
    )
    launch_config = LaunchConfig(training_config=training_config, resource_config=SINGLE_GPU_CONFIG)

    launch_training_with_ray(launch_config)
