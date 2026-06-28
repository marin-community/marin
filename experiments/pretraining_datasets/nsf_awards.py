# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NSF awards dataset download, normalization, and tokenization as a lazy Dataset handle."""

from marin.datakit.download.nsf_awards import MAX_YEAR, MIN_YEAR, download_nsf_awards
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import Dataset
from marin.experiment.data import derived, raw_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer


def _run_download(cfg: dict) -> None:
    download_nsf_awards(min_year=cfg["min_year"], max_year=cfg["max_year"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        text_field=cfg["text_field"],
        id_field=cfg["id_field"],
        file_extensions=tuple(cfg["file_extensions"]),
    )


def nsf_awards_datasets(*, tokenizer: str = marin_tokenizer) -> Dataset:
    """NSF awards corpus as a tokenized Dataset handle."""
    dl = raw_download(
        "raw/nsf-awards",
        fn=_run_download,
        build_config=lambda ctx: {
            "min_year": MIN_YEAR,
            "max_year": MAX_YEAR,
            "output_path": ctx.out,
        },
    )
    norm = derived(
        "normalized/nsf-awards",
        fn=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.path(dl),
            "output_path": ctx.out,
            "text_field": "text",
            "id_field": "awd_id",
            "file_extensions": [".parquet"],
        },
        deps=(dl,),
        kind=Dataset,
    )
    return tokenized("nsf_awards", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet")
