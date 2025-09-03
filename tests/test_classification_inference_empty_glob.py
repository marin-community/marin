import pytest
import ray

from marin.processing.classification.inference import InferenceConfig, run_inference


def test_run_inference_raises_for_empty_glob(tmp_path, ray_tpu_cluster):
    config = InferenceConfig(
        input_path=str(tmp_path),
        output_path=str(tmp_path / "out"),
        model_name="dummy",
        model_type="fasttext",
        attribute_name="test",
    )

    with pytest.raises(FileNotFoundError):
        ray.get(run_inference.remote(config))
