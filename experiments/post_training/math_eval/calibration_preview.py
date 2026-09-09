# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose the complete native calibration argv without starting a server."""

from marin.inference.backend import ModelSpec
from marin.inference.vllm_backend import VllmBackend, vllm_launcher
from marin.inference.vllm_server import VllmEnvironment, _vllm_serve_command


def calibration_command(model, engine):
    """Use the same model and environment lowering as local_inference."""
    if model.chat_template_content is not None:
        raise ValueError("Calibration uses token prompts, not a server chat template")
    spec = ModelSpec(
        weights=model.weights,
        api_model=model.model_id,
        num_chips=1,
        tensor_parallel_size=model.tensor_parallel_size,
        dtype=model.dtype,
        max_model_len=model.max_model_len,
        chat_template_content=model.chat_template_content,
        revision=model.revision,
    )
    backend = VllmBackend(engine, host="127.0.0.1", port=8000)
    environment = VllmEnvironment(
        model=backend._model_config(spec),
        host=backend.host,
        port=backend.port,
        timeout_seconds=engine.startup_timeout_seconds,
        extra_args=backend._serve_args(spec, (), ()),
        launcher=vllm_launcher(engine),
        compilation_cache_mode=engine.compilation_cache,
        extra_metric_families=engine.extra_metric_families,
        wait_for_ready=False,
    )
    return _vllm_serve_command(
        launcher=environment.launcher,
        model_name_or_path=environment.model_name_or_path,
        host=environment.host,
        port=environment.port,
        extra_cli_args=environment.extra_cli_args,
    )
