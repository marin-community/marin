# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the data-preparation steps upstream of the Nemotron canary ferry.

Runs v2 normalize for every Nemotron CC split and the DCLM code/math
components, then tokenizes everything that NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
depends on (Nemotron CC splits, DCLM components, default validation sets).
"""

import dataclasses

from fray import ResourceConfig
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.nemotron_v1 import (
    NEMOTRON_V1_SPLITS,
    download_nemotron_v1_step,
    normalize_nemotron_v1_step,
)
from marin.datakit.normalize import normalize_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.tokenize import TokenizeConfigBase

from experiments.defaults import default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron import nemotron_mix, tokenize_nemotron

MAX_WORKERS = 42
CACHE_COPY_MAX_WORKERS = 42
# Defaults bake in disk=5g (tokenize) / disk=10g (normalize) per worker pod,
# which evicts large splits mid-pipeline as ephemeral storage fills. iris
# treats ``disk="0"`` as "no ephemeral-storage limit on the pod" (k8s only
# applies node-level pressure). Cluster nodes have ~7 TiB ephemeral storage,
# so this is the simplest fix.
TOKENIZE_WORKER_RESOURCES = ResourceConfig(ram="12g", disk="0")
NORMALIZE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="0")

# DCLM components consumed by the Nemotron canary mix. Each entry mirrors the
# ``simple._dl(...)`` download wiring, plus the raw text column used by the HF
# parquet shards. Override paths point at the existing raw downloads so this
# script doesn't redownload.
DCLM_COMPONENTS = [
    {
        "name": "starcoderdata",
        "hf_dataset_id": "bigcode/starcoderdata",
        "revision": "9fc30b5",
        "download_override_path": "raw/starcoderdata-720c8c",
        "text_field": "content",
        "file_extensions": (".parquet",),
    },
    {
        "name": "proofpile_2",
        "hf_dataset_id": "EleutherAI/proof-pile-2",
        "revision": "901a927",
        "download_override_path": "raw/proof-pile-2-f1b1d8",
        "text_field": "text",
        # HF proof-pile-2 ships as compressed JSONL, not Parquet.
        "file_extensions": (".jsonl.zst",),
    },
]


def _with_worker_caps(step: ExecutorStep) -> ExecutorStep:
    """Override max_workers, cache_copy_max_workers, and per-worker resources on a TokenizeConfig step."""
    config = step.config
    if not isinstance(config, TokenizeConfigBase):
        return step
    return dataclasses.replace(
        step,
        config=dataclasses.replace(
            config,
            max_workers=MAX_WORKERS,
            cache_copy_max_workers=CACHE_COPY_MAX_WORKERS,
            worker_resources=TOKENIZE_WORKER_RESOURCES,
        ),
    )


def main() -> None:
    nemotron_download = download_nemotron_v1_step()
    normalize_by_split = {
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
    normalize_steps = list(normalize_by_split.values())
    # Tokenize splits read the normalized parquet output instead of the raw
    # jsonl.zst dump under data-jsonl/.
    input_paths_by_split = {split: [step / "outputs/main/*.parquet"] for split, step in normalize_by_split.items()}
    nemotron_steps = [
        _with_worker_caps(step)
        for step in tokenize_nemotron(
            max_workers=MAX_WORKERS,
            cache_copy_max_workers=CACHE_COPY_MAX_WORKERS,
            input_paths_by_split=input_paths_by_split,
        ).values()
    ]
    dclm_normalize_steps: list[ExecutorStep] = []
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
            file_extensions=component["file_extensions"],
            max_workers=MAX_WORKERS,
            worker_resources=NORMALIZE_WORKER_RESOURCES,
            version="v2",
        ).as_executor_step()
        dclm_normalize_steps.append(normalized_step)
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
    steps = normalize_steps + nemotron_steps + dclm_normalize_steps + dclm_tokenize_steps + validation_steps

    executor_main(steps=steps)


if __name__ == "__main__":
    main()
