# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ClimbLab-Ja pre-training dataset as a lazy Dataset handle.

~300B Japanese tokens from KantaHayashiAI/ClimbLab-Ja (ODC-BY licensed),
normalized through the standard Marin pipeline and tokenized via the
default Marin tokenizer.
"""

from marin.datakit.download.climblab_ja import HF_DATASET_ID, HF_REVISION
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import Dataset
from marin.experiment.data import derived, hf_download, tokenized

from experiments.marin_tokenizer import marin_tokenizer


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        file_extensions=tuple(cfg["file_extensions"]),
    )


def climblab_ja_datasets(*, tokenizer: str = marin_tokenizer) -> Dataset:
    """ClimbLab-Ja as a tokenized Dataset handle."""
    dl = hf_download("raw/climblab-ja", hf_id=HF_DATASET_ID, revision=HF_REVISION)
    norm = derived(
        "normalized/climblab-ja",
        fn=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.path(dl),
            "output_path": ctx.out,
            "file_extensions": [".parquet"],
        },
        deps=(dl,),
        kind=Dataset,
    )
    return tokenized("climblab-ja", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet")
