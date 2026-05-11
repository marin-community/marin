# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only DCLM (starcoderdata + proofpile_2) v2 normalize.

Both DCLM normalize steps previously failed inside the parent
``nemotron_data`` run. Use this entrypoint to retry them in isolation
without restarting the in-flight Nemotron normalize/tokenize work.
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
    steps = []
    for component in DCLM_COMPONENTS:
        name = component["name"]
        download_spec = download_hf_step(
            f"raw/{name}",
            hf_dataset_id=component["hf_dataset_id"],
            revision=component["revision"],
            override_output_path=component["download_override_path"],
        )
        normalized_step = normalize_step(
            name=f"normalized/{name}",
            download=download_spec,
            text_field=component["text_field"],
            id_field="id",
            file_extensions=component["file_extensions"],
            bare=component["bare"],
            max_workers=MAX_WORKERS,
            worker_resources=NORMALIZE_WORKER_RESOURCES,
            version="v2",
        ).as_executor_step()
        steps.append(normalized_step)
    print(f"Submitting {len(steps)} DCLM normalize steps")
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
