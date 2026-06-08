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
import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import unwrap_versioned_value
from marin.execution.types import this_output_path, versioned

from experiments.grug.base.launch import grug_base_launch, train_grug

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


def main() -> None:
    train_grug(
        "grug/multislice-smoke",
        multislice_smoke_launch(),
        override_output_path=os.environ.get("GRUG_MULTISLICE_OUTPUT_PATH"),
        env_vars=_child_env_vars(),
    )


if __name__ == "__main__":
    main()
