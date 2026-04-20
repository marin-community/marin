# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical base-model pointer for the per-statement DPO Bug-1 campaign.

Future per_stmt_dpo experiments should use ``LLAMA_3_1_8B_INSTRUCT_GCS_PATH``
as their base model instead of the ``marin-community/marin-8b-instruct``
HuggingFace identifier wired into older experiments in this directory.

Why
---

Loading from same-region GCS (~288 MB/s, VM-network-bound) is roughly 3-5x
faster than pulling the 16 GB of safetensor shards from the HuggingFace Hub
CDN, and is not subject to HF rate limiting or intermittent network flakes.

Bug-1 (the topology-sensitive canonical-vs-reverse training-basin split) was
confirmed model-agnostic on 2026-04-19 by reproducing the BL step-9 split on
this base:

- Llama-3.1 canonical step 9: 0.427
- Llama-3.1 reverse   step 9: 0.111  (gap 0.316)
- marin-8b   canonical step 9: 0.662
- marin-8b   reverse   step 9: 0.308  (gap 0.354)

Same sign, same order of magnitude. Absolute loss values differ because
Llama-3.1-8B-Instruct is a different starting distribution for this DPO data,
but the relative canonical-vs-reverse gap is preserved.

Reference implementation:
  ``experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py``

Context:
  ``.agents/logbooks/bug_1_dpo_lora_physical_topology.md``
  (2026-04-19T Base Model Swap section).

Usage
-----

Old:

    model_name_or_path="marin-community/marin-8b-instruct",

New:

    from experiments.posttrain.per_stmt_dpo.base_model import (
        LLAMA_3_1_8B_INSTRUCT_GCS_PATH,
    )

    model_name_or_path=LLAMA_3_1_8B_INSTRUCT_GCS_PATH,

Do NOT retroactively change ``model_name_or_path`` in historical experiment
scripts — the logged W&B losses and HLO artifacts are tied to the base model
that produced them. Change only in new experiments. Historical scripts keep a
docstring note (added 2026-04-19) pointing readers here.
"""

# Same-region GCS copies exist in both us-central1 and us-east5. Revision pin
# ``0e9e39f`` matches the canonical entry in ``experiments/models.py``. The
# iris worker image does not ship ``marin.download``, so the path is declared
# as a plain string rather than via ``output_path_of(llama_3_1_8b_instruct)``.
LLAMA_3_1_8B_INSTRUCT_GCS_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

# Per-region GCS copies. Marin enforces a same-region check on
# ``initialize_from_hf`` to prevent cross-region egress billing surprises, so
# experiments launching in ``us-east5`` must pick the ``us-east5`` copy even
# though contents are byte-identical.
LLAMA_3_1_8B_INSTRUCT_GCS_PATH_BY_REGION = {
    "us-central1": "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
    "us-east5": "gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
}


def llama_3_1_8b_instruct_gcs_path_for(regions: list[str]) -> str:
    """Pick the same-region GCS copy for the first region in ``regions``.

    Marin's ``check_gcs_paths_same_region`` rejects cross-region model loads,
    so scripts that accept a list of acceptable regions must pick a model
    path matching the region where the VM actually schedules. Callers that
    constrain to a single region via ``REGIONS_OVERRIDE`` should pass that
    single-region list here.
    """
    for region in regions:
        path = LLAMA_3_1_8B_INSTRUCT_GCS_PATH_BY_REGION.get(region)
        if path is not None:
            return path
    raise ValueError(
        f"No same-region Llama-3.1-8B-Instruct GCS copy for regions={regions}; "
        f"known regions: {sorted(LLAMA_3_1_8B_INSTRUCT_GCS_PATH_BY_REGION)}"
    )
