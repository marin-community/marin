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

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

# Mapping of canonical family name → (HF repo id, revision, subsets).
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


def _download(family: str) -> StepSpec:
    hf_dataset_id, revision, _ = SAFETY_PRETRAINING_FAMILIES[family]
    return download_hf_step(
        f"raw/{family}",
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        override_output_path=f"raw/{family}-{revision}",
    )


def safety_pretraining_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every Safety Pretraining subset.

    Keyed by the registry name ``"safety_pt/<family>/<subset>"``. The family
    download is materialized once per family and reused across its subsets;
    each subset gets its own ``normalize`` step pointed at the corresponding
    score-bucket directory via ``relative_input_path``.
    """
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for family, (_, _, subsets) in SAFETY_PRETRAINING_FAMILIES.items():
        download = _download(family)
        for subset in subsets:
            marin_name = f"safety_pt/{family}/{subset}"
            normalize = normalize_step(
                name=f"normalized/{marin_name}",
                download=download,
                relative_input_path=subset,
            )
            chains[marin_name] = (download, normalize)
    return chains
