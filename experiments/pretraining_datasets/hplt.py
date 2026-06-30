# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HPLT v3.0 dataset definitions and tokenization.

HPLT v3.0 English data filtered to non-Common Crawl sources only (WIDE, survey crawls),
with register-based quality filtering. This avoids redundancy with Nemotron CC while
adding ~612.7B unique tokens from European web crawls.
"""

from fray.types import ResourceConfig
from marin.datakit.download.hplt import HPLT_BASE_URL, download_hplt_v3
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import raw_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer

HPLT_DATASETS = {
    "all": ["*.parquet"],
}


def _run_hplt_download(cfg: dict) -> None:
    download_hplt_v3(output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        text_field=cfg["text_field"],
        id_field=cfg["id_field"],
        file_extensions=tuple(cfg["file_extensions"]),
    )


def hplt_datasets(*, tokenizer: str = marin_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """One tokenized Dataset handle per HPLT v3.0 split, keyed by split name."""
    dl = raw_download(
        "raw/hplt_v3",
        fn=_run_hplt_download,
        build_config=lambda ctx: {"base_url": HPLT_BASE_URL, "output_path": ctx.output_path},
        version="2026.06.28",
    )
    norm = ArtifactStep(
        name="normalized/hplt_v3",
        version="2026.06.28",
        artifact_type=Artifact,
        run=_run_normalize,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(dl),
            "output_path": ctx.output_path,
            "text_field": "text",
            "id_field": "id",
            "file_extensions": [".parquet"],
        },
        deps=(dl,),
    )
    return {
        "hplt_v3/all": tokenized(
            "hplt_v3/all",
            tokenizer=tokenizer,
            raw=norm,
            glob="outputs/main/*.parquet",
            resources=ResourceConfig(ram="20g", disk="5g"),
            version="2026.06.28",
        )
    }
