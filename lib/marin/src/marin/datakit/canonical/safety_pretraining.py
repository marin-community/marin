# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit canonical pipelines for the locuslab Safety Pretraining datasets.

HuggingFace collection: https://huggingface.co/collections/locuslab/safety-pretraining-datasets

The collection contains four ODC-BY licensed Parquet datasets from Maini et
al.'s safety-pretraining work. Every dataset shares the same row schema
(``id: string``, ``text: string``, ``metadata: struct``) and is organized as
top-level score-bucket directories of sharded Parquet files.

Subsets per dataset (each a top-level directory in the raw repo):

* ``locuslab/moral_education``: ``score_4_morals``, ``score_5_morals``
* ``locuslab/safeweb``: ``score_1_rephrased``, ``score_3_rephrased``,
  ``score_4_rephrased``, ``score_5_rephrased``
* ``locuslab/refuseweb``: ``score_4_refusal``, ``score_5_refusal``
* ``locuslab/fineweb_annotated``: ``score_0`` ... ``score_5``
"""

from fray import ResourceConfig

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

# Mapping of canonical family name → (HF repo id, default revision, subsets).
# Revisions pinned 2026-05-14 from https://huggingface.co/api/datasets/<repo>.
SAFETY_PRETRAINING_FAMILIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "moral_education": (
        "locuslab/moral_education",
        "579300f",
        ("score_4_morals", "score_5_morals"),
    ),
    "safeweb": (
        "locuslab/safeweb",
        "f73f213",
        (
            "score_1_rephrased",
            "score_3_rephrased",
            "score_4_rephrased",
            "score_5_rephrased",
        ),
    ),
    "refuseweb": (
        "locuslab/refuseweb",
        "a2cb2e1",
        ("score_4_refusal", "score_5_refusal"),
    ),
    "fineweb_annotated": (
        "locuslab/fineweb_annotated",
        "d96861c",
        ("score_0", "score_1", "score_2", "score_3", "score_4", "score_5"),
    ),
}


def _download(
    family: str,
    *,
    revision: str | None = None,
    hf_urls_glob: list[str] | None = None,
    worker_resources: ResourceConfig | None = None,
) -> StepSpec:
    hf_dataset_id, default_revision, _ = SAFETY_PRETRAINING_FAMILIES[family]
    resolved_revision = revision or default_revision
    return download_hf_step(
        f"raw/{family}",
        hf_dataset_id=hf_dataset_id,
        revision=resolved_revision,
        hf_urls_glob=hf_urls_glob,
        override_output_path=f"raw/{family}-{resolved_revision}",
        worker_resources=worker_resources,
    )


def download_moral_education(
    *,
    revision: str | None = None,
    hf_urls_glob: list[str] | None = None,
    worker_resources: ResourceConfig | None = None,
) -> StepSpec:
    """Download ``locuslab/moral_education`` from HuggingFace."""
    return _download(
        "moral_education",
        revision=revision,
        hf_urls_glob=hf_urls_glob,
        worker_resources=worker_resources,
    )


def download_safeweb(
    *,
    revision: str | None = None,
    hf_urls_glob: list[str] | None = None,
    worker_resources: ResourceConfig | None = None,
) -> StepSpec:
    """Download ``locuslab/safeweb`` from HuggingFace."""
    return _download(
        "safeweb",
        revision=revision,
        hf_urls_glob=hf_urls_glob,
        worker_resources=worker_resources,
    )


def download_refuseweb(
    *,
    revision: str | None = None,
    hf_urls_glob: list[str] | None = None,
    worker_resources: ResourceConfig | None = None,
) -> StepSpec:
    """Download ``locuslab/refuseweb`` from HuggingFace."""
    return _download(
        "refuseweb",
        revision=revision,
        hf_urls_glob=hf_urls_glob,
        worker_resources=worker_resources,
    )


def download_fineweb_annotated(
    *,
    revision: str | None = None,
    hf_urls_glob: list[str] | None = None,
    worker_resources: ResourceConfig | None = None,
) -> StepSpec:
    """Download ``locuslab/fineweb_annotated`` from HuggingFace."""
    return _download(
        "fineweb_annotated",
        revision=revision,
        hf_urls_glob=hf_urls_glob,
        worker_resources=worker_resources,
    )
