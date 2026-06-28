# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron v2 pre-training dataset tokenization as lazy Dataset handles.

One Dataset handle per (family, subset) combination, reading from each subset's
normalized outputs. Families come from the NEMOTRON_V2_DATASETS registry in
marin.datakit.download.nemotron_v2; subsets and their globs are defined there.
"""

from marin.datakit.download.nemotron_v2 import NEMOTRON_V2_DATASETS
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.artifact import Dataset
from marin.execution.lazy import Lazy, derived
from marin.experiment.data import hf_download, tokenized

from experiments.llama import llama3_tokenizer


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        text_field=cfg["text_field"],
        id_field=cfg["id_field"],
        file_extensions=tuple(cfg["file_extensions"]),
        worker_resources=cfg.get("worker_resources"),
    )


def nemotron_v2_family_datasets(family: str, *, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """One Dataset handle per subset of a Nemotron v2 family, keyed by ``{family}/{subset}``."""
    info = NEMOTRON_V2_DATASETS[family]
    dl = hf_download(f"raw/{family}", hf_id=info.hf_dataset_id, revision=info.revision, pin=info.override_output_path)

    result: dict[str, Lazy[Dataset]] = {}
    for subset, glob_pattern in info.subsets.items():
        subset_dir = glob_pattern.split("/**")[0]
        text_field = info.subset_text_fields.get(subset, "text")
        worker_resources = info.subset_normalize_worker_resources.get(subset)

        norm = derived(
            f"normalized/{family}/{subset}",
            fn=_run_normalize,
            build_config=lambda ctx, _dl=dl, _sd=subset_dir, _tf=text_field, _wr=worker_resources: {
                "input_path": f"{ctx.path(_dl)}/{_sd}",
                "output_path": ctx.out,
                "text_field": _tf,
                "id_field": "id",
                "file_extensions": [".parquet"],
                "worker_resources": _wr,
            },
            deps=(dl,),
            kind=Dataset,
        )
        result[f"{family}/{subset}"] = tokenized(
            f"{family}/{subset}",
            tokenizer=tokenizer,
            raw=norm,
            glob="outputs/main/*.parquet",
        )
    return result


def nemotron_v2_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """One Dataset handle per (family, subset) for all Nemotron v2 families."""
    all_datasets: dict[str, Lazy[Dataset]] = {}
    for family in NEMOTRON_V2_DATASETS:
        all_datasets.update(nemotron_v2_family_datasets(family, tokenizer=tokenizer))
    return all_datasets
