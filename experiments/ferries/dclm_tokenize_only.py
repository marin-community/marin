# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize-only entrypoint for the DCLM components (starcoderdata + proofpile_2).

Submit this once both DCLM normalize steps have ``STATUS_SUCCESS`` (use
``dclm_normalize_only`` / ``proofpile2_normalize_only`` to produce those).
The script skips any component whose upstream normalize is not yet SUCCESS
so it can be re-submitted incrementally as components come online.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.execution.executor_step_status import STATUS_SUCCESS, get_status_path
from marin.processing.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import url_to_fs

from experiments.ferries.nemotron_data import (
    DCLM_COMPONENTS,
    NORMALIZE_WORKER_RESOURCES,
    _with_worker_caps,
)
from experiments.llama import llama3_tokenizer


def _is_succeeded(output_path: str) -> bool:
    fs, resolved = url_to_fs(get_status_path(output_path))
    if not fs.exists(resolved):
        return False
    with fs.open(resolved, "r") as f:
        return f.read().strip() == STATUS_SUCCESS


def main() -> None:
    steps: list[ExecutorStep] = []
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
            worker_resources=NORMALIZE_WORKER_RESOURCES,
            version="v2",
        ).as_executor_step()
        if not _is_succeeded(normalized_step.override_output_path):
            print(f"Skipping DCLM tokenize for {name}: upstream normalize not yet SUCCESS")
            continue
        tokenize_step = _with_worker_caps(
            ExecutorStep(
                name=f"tokenized/{name}",
                fn=tokenize,
                config=TokenizeConfig(
                    train_paths=[normalized_step / "outputs/main/*.parquet"],
                    validation_paths=versioned([]),
                    cache_path=this_output_path(),
                    tokenizer=versioned(llama3_tokenizer),
                ),
            )
        )
        steps.append(tokenize_step)
    print(f"Submitting {len(steps)} DCLM tokenize steps")
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
