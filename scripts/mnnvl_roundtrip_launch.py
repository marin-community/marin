# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the MNNVL round-trip probe as four single-GPU JAX tasks."""

import os

from fray.cluster import ResourceConfig

from experiments.grug.dispatch import dispatch_grug_training_run
from scripts.mnnvl_roundtrip_probe import main as run_roundtrip_probe


def _run_probe(_: None) -> None:
    run_roundtrip_probe()


def main() -> None:
    run_id = os.environ.get("RUN_ID", "mnnvl-roundtrip")
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=1,
        cpu=8,
        ram="64g",
        disk="64g",
        replicas=4,
    )
    dispatch_grug_training_run(
        run_id=run_id,
        config=None,
        local_entrypoint=_run_probe,
        resources=resources,
        processes_per_task=1,
        max_retries_failure=0,
    )


if __name__ == "__main__":
    main()
