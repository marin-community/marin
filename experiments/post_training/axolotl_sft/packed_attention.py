# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl plugin: packed conversations never attend to each other (pod-side, image environment).

Axolotl's multipack collator numbers each packed sample in ``attention_mask`` (1, 2, 3, ...).
Under transformers 5.12 the sdpa and flex paths read that as a plain padding mask, so a
packed conversation attends to the ones before it (measured by ``packing_check.py``: segment
logits differ from the unpacked forward by ~100%). Transformers isolates packed sequences
itself when the batch carries only ``position_ids`` that restart at every sample, no
attention mask, and no KV cache. This plugin's collator drops the attention mask, and the
model's KV cache is turned off for training; packed and unpacked logits then agree to 1e-7.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.monkeypatch.models.qwen3_5.modeling import patch_qwen3_5_moe_modeling_packing
from axolotl.utils.collators.batching import V2BatchSamplerDataCollatorForSeq2Seq


def apply_packing_patch(model_type: str) -> None:
    """Apply the architecture patch that Axolotl training uses for packed segments."""
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        patch_qwen3_5_moe_modeling_packing()


class SegmentPositionCollator(V2BatchSamplerDataCollatorForSeq2Seq):
    """The multipack collator without ``attention_mask``: segments come from position-id resets."""

    def __call__(self, features, return_tensors=None):
        packs = features if isinstance(features[0], list) else [features]
        stripped = [[{k: v for k, v in item.items() if k != "attention_mask"} for item in pack] for pack in packs]
        return super().__call__(stripped, return_tensors=return_tensors)


class PackedAttentionPlugin(BasePlugin):
    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        return SegmentPositionCollator, {}

    def post_model_load(self, cfg, model):
        # A KV cache disables transformers' packed-sequence detection.
        model.config.use_cache = False
