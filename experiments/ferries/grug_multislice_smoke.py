# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightly Grug multislice smoke run.

Config is driven by env vars set in the GitHub Actions workflow env block and
forwarded to the Iris parent job.

    RUN_ID                         unique run identifier
    GRUG_MULTISLICE_BATCH_SIZE     global train batch size
    GRUG_MULTISLICE_CE_IMPL        optional loss implementation override
    GRUG_MULTISLICE_MAX_SEQ_LEN    sequence length
    GRUG_MULTISLICE_OUTPUT_PATH    optional explicit output path
    GRUG_MULTISLICE_REGION         GCP region for TPU slices
    GRUG_MULTISLICE_SLICE_COUNT    TPU slice count
    GRUG_MULTISLICE_STEPS          training steps
    GRUG_MULTISLICE_TPU_TYPE       TPU type per slice
"""

import dataclasses
import logging
import os

import fsspec
from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import compute_output_path, unwrap_versioned_value
from marin.execution.types import this_output_path, versioned

from experiments.grug.base.launch import GrugBaseLaunchConfig, grug_base_launch, train_grug

logger = logging.getLogger(__name__)

CANARY_STEP_NAME = "grug/multislice-smoke"

DEFAULT_REGION = "us-east5"
DEFAULT_TPU_TYPE = "v6e-8"
DEFAULT_SLICE_COUNT = 2
DEFAULT_STEPS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SEQ_LEN = 4096
CHILD_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_ENDPOINT_URL",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "MARIN_PREFIX",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_PROJECT",
)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def _run_id() -> str:
    return os.environ.get("RUN_ID") or os.environ.get("GRUG_RUN_ID") or "grug-multislice-smoke"


def multislice_smoke_resources() -> ResourceConfig:
    region = os.environ.get("GRUG_MULTISLICE_REGION", DEFAULT_REGION)
    tpu_type = os.environ.get("GRUG_MULTISLICE_TPU_TYPE", DEFAULT_TPU_TYPE)
    slice_count = _env_int("GRUG_MULTISLICE_SLICE_COUNT", DEFAULT_SLICE_COUNT)
    return ResourceConfig.with_tpu(tpu_type, slice_count=slice_count, regions=[region])


def multislice_smoke_launch():
    steps = _env_int("GRUG_MULTISLICE_STEPS", DEFAULT_STEPS)
    batch_size = _env_int("GRUG_MULTISLICE_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    max_seq_len = _env_int("GRUG_MULTISLICE_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN)
    loss_implementation = os.environ.get("GRUG_MULTISLICE_CE_IMPL") or None
    model = dataclasses.replace(unwrap_versioned_value(grug_base_launch.model), max_seq_len=max_seq_len)
    grug_trainer = dataclasses.replace(
        unwrap_versioned_value(grug_base_launch.grug_trainer),
        loss_implementation=loss_implementation,
    )

    return dataclasses.replace(
        grug_base_launch,
        run_id=_run_id(),
        model=versioned(model),
        resources=versioned(multislice_smoke_resources()),
        steps=versioned(steps),
        batch_size=versioned(batch_size),
        tracker=WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or None,
            project=os.environ.get("WANDB_PROJECT", "marin"),
            tags=["grug", "multislice", "smoke"],
            group="grug-multislice-smoke",
            name=None,
            mode=os.environ.get("WANDB_MODE") or None,
            replicate_path=this_output_path(),
        ),
        grug_trainer=versioned(grug_trainer),
        eval=None,
    )


def _child_env_vars() -> dict[str, str]:
    return {key: value for key in CHILD_ENV_KEYS if (value := os.environ.get(key))}


def _override_output_path() -> str | None:
    """Read ``GRUG_MULTISLICE_OUTPUT_PATH`` and validate it.

    If the env var is unset, returns ``None`` and the canary falls back to
    its deterministic content-addressed path. If the env var is set, it
    must be non-empty: an empty value would be interpreted by
    :func:`compute_output_path` as a relative override joined with
    ``MARIN_PREFIX``, which would point the pre-submit wipe at the prefix
    root (e.g. ``gs://marin-us-east5``) and recursively delete it.
    """
    if "GRUG_MULTISLICE_OUTPUT_PATH" not in os.environ:
        return None
    value = os.environ["GRUG_MULTISLICE_OUTPUT_PATH"]
    if not value:
        raise ValueError(
            "GRUG_MULTISLICE_OUTPUT_PATH is set but empty; "
            "unset it to use the default canary path, or provide an absolute output path."
        )
    return value


def wipe_path_if_exists(path: str) -> None:
    """Recursively delete ``path`` via fsspec if it exists; no-op otherwise."""
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    if not fs.exists(plain_path):
        logger.info("Canary output path %s does not exist; nothing to wipe.", path)
        return
    logger.info("Wiping canary output directory %s before submit.", path)
    fs.rm(plain_path, recursive=True)


def _wipe_canary_output(
    name: str,
    launch: GrugBaseLaunchConfig,
    override_output_path: str | None,
) -> None:
    """Delete the canary's deterministic output directory if it exists.

    The canary writes to a content-addressed path that is stable across runs
    (e.g. ``gs://marin-us-east5/grug/multislice-smoke-<hash>/``). When a
    previous run left a checkpoint behind, Levanter resumes it and then the
    final forced checkpoint write trips TensorStore's ``chunk_shape`` lock if
    the sharding has drifted since the prior write (see issue #6253). The
    canary is meant to be self-contained, so wipe the directory before
    submitting the training job — each invocation starts from an empty path.
    """
    output_path = compute_output_path(name, launch, override_output_path=override_output_path)
    wipe_path_if_exists(output_path)


def main() -> None:
    launch = multislice_smoke_launch()
    override_output_path = _override_output_path()
    _wipe_canary_output(CANARY_STEP_NAME, launch, override_output_path)
    train_grug(
        CANARY_STEP_NAME,
        launch,
        override_output_path=override_output_path,
        env_vars=_child_env_vars(),
    )


if __name__ == "__main__":
    main()
