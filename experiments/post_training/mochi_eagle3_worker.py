# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native vLLM worker extension for one regional Mochi capture interval."""

import json
import tempfile
from pathlib import Path

from rigging.filesystem.storage_path import prefix_join
from skyrl_train.inference_engines.base import request_group_from_id
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    capture_rank_directory,
    capture_rank_name,
    per_worker_capture_token_credit,
)
from skyrl_train.io import io
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class MochiCaptureWorker:
    model_runner: GPUModelRunner

    def begin_mochi_capture(self, payload: str) -> dict:
        """Reserve bounded host capture credit on each data-parallel rank."""
        config = json.loads(payload)
        parallel = self.model_runner.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Mochi capture requires TP1/PP1")
        rank = parallel.data_parallel_rank
        config["max_tokens"] = per_worker_capture_token_credit(
            global_max_tokens=config["max_tokens"],
            max_window_tokens=config["max_window_tokens"],
            worker_count=parallel.data_parallel_size,
            worker_index=rank,
        )
        config["capture_target_snapshot"] = rank == 0
        self.model_runner.refresh_online_eagle_target_owned_weights()
        return self.model_runner.begin_online_eagle_capture(config)

    def seal_mochi_capture(self, destination: str) -> dict:
        """Publish one immutable rank capture directly to regional object storage."""
        rank = self.model_runner.parallel_config.data_parallel_rank
        rank_destination = prefix_join(destination, capture_rank_name(rank))
        with tempfile.TemporaryDirectory(prefix="mochi-eagle3-capture-") as scratch:
            directory = capture_rank_directory(Path(scratch), rank)
            result = self.model_runner.seal_online_eagle_capture(str(directory))
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for window in manifest["windows"]:
                # Native completions prepend cmpl-, while SkyRL's shared grouping
                # helper handles the chatcmpl- prefix used by rollout engines.
                window["group_id"] = request_group_from_id(window["request_id"].removeprefix("cmpl-"))
            manifest_path.write_text(json.dumps(manifest, indent=2))
            io.upload_directory(str(directory), rank_destination)
        return {**result, "path": rank_destination}
