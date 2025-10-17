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

"""
Shared utilities for probextraction experiments.

- list_books: enumerate .txt book files from a GCS/local path (dir or single file)
- run_eval_sliding_on_tpu: launch Levanter marin_eval_sliding_total on a TPU slice via Ray
- make_run_eval_sliding_fn: curry TPU params into a callable suitable for ExecutorStep.fn
- run_eval_pz_on_tpu / make_run_eval_pz_fn: analogous helpers for P(z) evaluation
"""

from __future__ import annotations

from pathlib import Path
import os

import ray

from marin.resources import TpuPodConfig
from marin.utils import fsspec_glob
from levanter.infra.ray_tpu import run_on_pod_resumable
from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, main as eval_sliding_main
from levanter.main.eval_pz import PzEvalConfig, main as eval_pz_main


def list_books(gcp_path: str) -> list[tuple[str, str]]:
    """Return a list of (book_title, txt_path) for a directory or a single .txt file.

    - If `gcp_path` ends with .txt, returns just that file.
    - Else, glob `*.txt` under the directory (works with gs:// via fsspec).
    """
    txt_files = [gcp_path] if gcp_path.endswith(".txt") else fsspec_glob(f"{gcp_path.rstrip('/')}/*.txt")
    out: list[tuple[str, str]] = []
    for txt_path in txt_files:
        filename = Path(txt_path).stem
        book_title = filename
        out.append((book_title, txt_path))
    return out


def run_eval_sliding_on_tpu(
    config: EvalSlidingTotalConfig,
    tpu_type: str = "v4-128",
    slice_count: int = 1,
) -> None:
    """Run Levanter's marin_eval_sliding_total on a TPU slice via Ray.

    The function signature matches ExecutorStep.fn; configure TPU resources
    with `tpu_type` and `slice_count` when creating the step function via
    `make_run_eval_sliding_fn`.
    """

    hw_config = TpuPodConfig(tpu_type=tpu_type, slice_count=slice_count, runtime_env={"env_vars": {}})

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def eval_lm_task():
        eval_sliding_main(config)

    return run_on_pod_resumable(eval_lm_task, hw_config.accelerator_descriptor(), max_retries_failure=10)


def make_run_eval_sliding_fn(tpu_type: str = "v4-128", slice_count: int = 1):
    """Return a callable(config) that runs eval_sliding on the specified TPU configuration."""

    def _runner(cfg: EvalSlidingTotalConfig):
        return run_eval_sliding_on_tpu(cfg, tpu_type=tpu_type, slice_count=slice_count)

    return _runner


def run_eval_pz_on_tpu(
    config: PzEvalConfig,
    tpu_type: str = "v4-128",
    slice_count: int = 1,
) -> None:
    """Run Levanter's eval_pz on a TPU slice via Ray."""

    hw_config = TpuPodConfig(tpu_type=tpu_type, slice_count=slice_count, runtime_env={"env_vars": {}})

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def eval_task():
        eval_pz_main(config)

    return run_on_pod_resumable(eval_task, hw_config.accelerator_descriptor(), max_retries_failure=10)


def make_run_eval_pz_fn(tpu_type: str = "v4-128", slice_count: int = 1):
    """Return a callable(config) that runs eval_pz on the specified TPU configuration."""

    def _runner(cfg: PzEvalConfig):
        return run_eval_pz_on_tpu(cfg, tpu_type=tpu_type, slice_count=slice_count)

    return _runner


# -----------------------------------------------------------------------------
# Hardware selection helper for probextraction jobs
# -----------------------------------------------------------------------------

# Conservative mapping from model size (billions of parameters) to TPU type and
# a reasonable eval batch size for P(z). Tweak as needed per environment.
HW_PRESETS: list[tuple[float, str, int]] = [
    # (max_params_b, tpu_type, eval_batch_size)
    (8.0, "v4-64", 256),
    (15.0, "v4-128", 512),
    (35.0, "v4-128", 512),
    (80.0, "v4-256", 256),
]


def choose_hw_and_batch(params_b: float, *, override_env: str = "TPU_TYPE_OVERRIDE") -> tuple[str, int]:
    """Select TPU type and eval batch size for a given model size.

    - If environment variable `override_env` is set (default: `TPU_TYPE_OVERRIDE`),
      force that TPU type and pick a conservative batch: 512 on v4-128, 256 on v4-64.
    - Otherwise, use HW_PRESETS based on `params_b`.
    """
    override = os.environ.get(override_env)
    if override:
        tp = override.strip()
        if tp == "v4-64":
            return tp, 256
        elif tp == "v4-128":
            return tp, 512
        # Default conservative fallback
        return tp, 256

    for max_b, tpu, batch in HW_PRESETS:
        if params_b <= max_b:
            return tpu, batch
    # Fallback to largest preset
    _, tpu, batch = HW_PRESETS[-1]
    return tpu, batch
