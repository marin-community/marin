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

from __future__ import annotations

from pathlib import Path

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator


def test_model_config_destroy_removes_downloaded_dir(tmp_path: Path) -> None:
    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    (downloaded_dir / "marker.txt").write_text("ok")

    config = ModelConfig(name="unit-test-model", path=None, engine_kwargs={})
    config.downloaded_to = str(downloaded_dir)
    config.destroy()

    assert not downloaded_dir.exists()


def test_vllm_auto_enables_streaming_for_gcs_paths() -> None:
    model = ModelConfig(
        name="unit-test-model",
        path="gs://bucket/some-model",
        engine_kwargs={},
    )

    model_name_or_path, model = VllmTpuEvaluator.resolve_model_name_or_path(model)

    assert model.engine_kwargs["load_format"] == "runai_streamer"
    assert model_name_or_path == "gs://bucket/some-model"
    assert model.downloaded_to is None


def test_engine_kwargs_to_cli_args_threads_max_model_len() -> None:
    args = VllmTpuEvaluator._engine_kwargs_to_cli_args({"max_model_len": 8192})
    assert args == ["--max-model-len", "8192"]
