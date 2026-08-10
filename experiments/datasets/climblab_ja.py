# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ClimbLab-Ja pre-training dataset as a lazy Dataset handle.

~300B Japanese tokens from KantaHayashiAI/ClimbLab-Ja (ODC-BY licensed),
normalized through the standard Marin pipeline and tokenized via the
default Marin tokenizer.
"""

from marin.datakit.download.climblab_ja import HF_DATASET_ID, HF_REVISION
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        file_extensions=tuple(cfg["file_extensions"]),
    )


def climblab_ja_dataset(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """ClimbLab-Ja as a tokenized Dataset handle."""
    dl = hf_download("raw/climblab-ja", hf_id=HF_DATASET_ID, revision=HF_REVISION, version="2026.06.28")
    norm = ArtifactStep(
        name="normalized/climblab-ja",
        version="2026.06.28",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(dl),
            "output_path": ctx.output_path,
            "file_extensions": [".parquet"],
        },
        deps=(dl,),
    )
    return tokenized("climblab-ja", tokenizer=tokenizer, raw=norm, glob="outputs/main/*.parquet", version="2026.06.28")


if __name__ == "__main__":
    dataset_main({"climblab-ja": climblab_ja_dataset()})
