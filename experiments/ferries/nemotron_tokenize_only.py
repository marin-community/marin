# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize-only entrypoint for the Nemotron canary data prep.

Runs ``nemotron_data.py``'s tokenize steps but skips normalize entirely:
includes a tokenize step only if its upstream normalize already wrote a
``SUCCESS`` status. Use this when normalize is still running under a parallel
job and you want to make forward progress on tokenize without racing the
in-flight normalize coords on the same output paths.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.nemotron_v1 import (
    NEMOTRON_V1_SPLITS,
    download_nemotron_v1_step,
    normalize_nemotron_v1_step,
)
from marin.datakit.normalize import normalize_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.execution.executor_step_status import STATUS_SUCCESS, get_status_path
from marin.processing.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import url_to_fs

from experiments.defaults import default_validation_sets
from experiments.ferries.nemotron_data import (
    DCLM_COMPONENTS,
    MAX_WORKERS,
    NORMALIZE_WORKER_RESOURCES,
    _with_worker_caps,
)
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron import nemotron_mix, tokenize_nemotron


def _is_succeeded(output_path: str) -> bool:
    """True iff the executor wrote ``STATUS_SUCCESS`` for the step at this output_path."""
    fs, resolved = url_to_fs(get_status_path(output_path))
    if not fs.exists(resolved):
        return False
    with fs.open(resolved, "r") as f:
        return f.read().strip() == STATUS_SUCCESS


def main() -> None:
    nemotron_download = download_nemotron_v1_step()
    nemotron_normalize_by_split = {
        split: (
            normalize_nemotron_v1_step(
                nemotron_download,
                split=split,
                max_workers=MAX_WORKERS,
                worker_resources=NORMALIZE_WORKER_RESOURCES,
            ).as_executor_step()
        )
        for split in NEMOTRON_V1_SPLITS
    }
    ready_nemotron_splits = sorted(
        split for split, step in nemotron_normalize_by_split.items() if _is_succeeded(step.override_output_path)
    )
    print(f"Tokenize-ready Nemotron splits: {ready_nemotron_splits}")

    input_paths_by_split = {
        split: [nemotron_normalize_by_split[split] / "outputs/main/*.parquet"] for split in ready_nemotron_splits
    }
    nemotron_steps = [
        _with_worker_caps(step)
        for split_key, step in tokenize_nemotron(
            max_workers=MAX_WORKERS,
            cache_copy_max_workers=MAX_WORKERS,
            input_paths_by_split=input_paths_by_split,
        ).items()
        if split_key.removeprefix("nemotron_cc/") in ready_nemotron_splits
    ]

    dclm_tokenize_steps: list[ExecutorStep] = []
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
            file_extensions=(".parquet",),
            max_workers=MAX_WORKERS,
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
        dclm_tokenize_steps.append(tokenize_step)

    validation_steps = [
        _with_worker_caps(step) for step in default_validation_sets(tokenizer=nemotron_mix.tokenizer).values()
    ]
    steps = nemotron_steps + dclm_tokenize_steps + validation_steps
    print(f"Submitting {len(steps)} tokenize steps")

    executor_main(steps=steps)


if __name__ == "__main__":
    main()
