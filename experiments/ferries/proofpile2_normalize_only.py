# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the proof-pile-2 v2 normalize step.

Filed because the parent DCLM normalize originally configured
``file_extensions=(".parquet",)`` for both DCLM components, while
EleutherAI/proof-pile-2 actually ships as ``.jsonl.zst``. After fixing
``file_extensions`` in ``DCLM_COMPONENTS``, this ferry retries proof-pile-2
in isolation so it doesn't conflict with any in-flight DCLM normalize run.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.executor import executor_main

from experiments.ferries.nemotron_data import (
    DCLM_COMPONENTS,
    MAX_WORKERS,
    NORMALIZE_WORKER_RESOURCES,
)


def main() -> None:
    component = next(c for c in DCLM_COMPONENTS if c["name"] == "proofpile_2")
    download_spec = download_hf_step(
        f"raw/{component['name']}",
        hf_dataset_id=component["hf_dataset_id"],
        revision=component["revision"],
        override_output_path=component["download_override_path"],
    )
    normalized_step = normalize_step(
        name=f"normalized/{component['name']}",
        download=download_spec,
        text_field=component["text_field"],
        id_field="id",
        file_extensions=component["file_extensions"],
        drop_fields=component["drop_fields"],
        bare=component["bare"],
        max_workers=MAX_WORKERS,
        worker_resources=NORMALIZE_WORKER_RESOURCES,
        version="v2",
    ).as_executor_step()
    executor_main(steps=[normalized_step])


if __name__ == "__main__":
    main()
