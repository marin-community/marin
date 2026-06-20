# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime vLLM/TPU-inference registry patches for Marin GrugMoE."""

from __future__ import annotations


def register_grugmoe_for_tpu_inference() -> None:
    from tpu_inference.models.common import model_loader  # noqa: PLC0415
    from tpu_inference.models.jax.grugmoe import GrugMoeForCausalLM  # noqa: PLC0415

    if model_loader._MODEL_REGISTRY.get("GrugMoeForCausalLM") is GrugMoeForCausalLM:
        return
    model_loader.register_model("GrugMoeForCausalLM", GrugMoeForCausalLM)


register_grugmoe_for_tpu_inference()
