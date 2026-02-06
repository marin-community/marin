# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
Preference learning dataset support for DPO/SimPO training.

This module provides dataset formats, processors, and datasets for preference
learning where each example contains chosen and rejected responses.
"""

import functools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Literal, TypedDict

import equinox as eqx
import haliax as hax
import jax
import numpy as np
from haliax import Axis

from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.packing import GreedyPrepackedDataset
from levanter.models.lm_model import LmExample
from levanter.store.cache import TreeCache
from levanter.utils.hf_utils import HfTokenizer

from .formats import ChatProcessor, LmDatasetFormatBase

logger = logging.getLogger("levanter.data.text.preference")


class ProcessedPreferenceChatDict(TypedDict):
    """Processed preference pair with chosen and rejected tokenized sequences."""

    chosen_input_ids: np.ndarray
    chosen_assistant_masks: np.ndarray
    rejected_input_ids: np.ndarray
    rejected_assistant_masks: np.ndarray


@LmDatasetFormatBase.register_subclass("preference_chat")
@dataclass(frozen=True)
class PreferenceChatLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for preference chat transcripts.

    Attributes:
        chosen_field: Field name containing the preferred chat transcript.
        rejected_field: Field name containing the rejected chat transcript.
        chat_template: Overrides the tokenizer's chat template when provided.
        system_prompt: Field name carrying an optional system instruction to prepend.
        chat_template_kwargs: Field name containing optional keyword arguments passed to the chat template.
        pack: Whether to allow example packing for efficient batching.
        mask_user_turns: Mask user tokens from the training loss when True.
        slice_strategy: How to handle sequences longer than the max length.
    """

    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    chat_template: str | None = None
    system_prompt: str | None = None
    chat_template_kwargs: str | None = "chat_template_kwargs"
    pack: bool = False
    mask_user_turns: bool = True
    slice_strategy: Literal["left", "right", "raise", "drop"] = "raise"

    @property
    def token_data_key(self) -> str:
        return "chosen_input_ids"


class PreferenceChatProcessor(BatchProcessor[dict, ProcessedPreferenceChatDict]):
    """
    A batch processor that converts preference chat data into chosen/rejected chat template outputs.

    This processor wraps two ChatProcessor instances to handle the chosen and rejected
    transcripts independently, then combines the results.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        *,
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        chat_template: str | None = None,
        system_prompt_field: str | None = "system",
        chat_template_kwargs_field: str | None = "chat_template_kwargs",
        mask_user_turns: bool = True,
    ):
        self._chosen = ChatProcessor(
            tokenizer,
            chat_template=chat_template,
            messages_field=chosen_field,
            system_prompt_field=system_prompt_field,
            chat_template_kwargs_field=chat_template_kwargs_field,
            mask_user_turns=mask_user_turns,
        )
        self._rejected = ChatProcessor(
            tokenizer,
            chat_template=chat_template,
            messages_field=rejected_field,
            system_prompt_field=system_prompt_field,
            chat_template_kwargs_field=chat_template_kwargs_field,
            mask_user_turns=mask_user_turns,
        )
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field

    def __call__(self, batch: Sequence[dict]) -> Sequence[ProcessedPreferenceChatDict]:
        valid_batch: list[dict] = []
        skipped_indices: list[int] = []
        for idx, example in enumerate(batch):
            chosen = example.get(self.chosen_field)
            rejected = example.get(self.rejected_field)
            if not chosen or not rejected:
                skipped_indices.append(idx)
                continue
            valid_batch.append(example)

        if skipped_indices:
            logger.warning(
                "Preference batch had %d invalid rows with empty chosen/rejected; skipping. Indices: %s",
                len(skipped_indices),
                skipped_indices[:10],
            )

        if not valid_batch:
            return []

        chosen_rows = self._chosen(valid_batch)
        rejected_rows = self._rejected(valid_batch)

        out: list[ProcessedPreferenceChatDict] = []
        for chosen, rejected in zip(chosen_rows, rejected_rows, strict=True):
            out.append(
                {
                    "chosen_input_ids": chosen["input_ids"],
                    "chosen_assistant_masks": chosen["assistant_masks"],
                    "rejected_input_ids": rejected["input_ids"],
                    "rejected_assistant_masks": rejected["assistant_masks"],
                }
            )

        return out

    @property
    def output_exemplar(self):
        return {
            "chosen_input_ids": np.zeros((0,), dtype=np.int32),
            "chosen_assistant_masks": np.zeros((0,), dtype=np.int32),
            "rejected_input_ids": np.zeros((0,), dtype=np.int32),
            "rejected_assistant_masks": np.zeros((0,), dtype=np.int32),
        }

    @property
    def num_cpus(self) -> int:
        return self._chosen.num_cpus

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "chosen": self._chosen.metadata,
            "rejected": self._rejected.metadata,
            "chosen_field": self.chosen_field,
            "rejected_field": self.rejected_field,
        }


class DpoExample(eqx.Module):
    """A preference pair example for DPO/SimPO training."""

    chosen: LmExample
    rejected: LmExample


class PreferencePairDataset(
    MappedAsyncDataset[tuple[ProcessedPreferenceChatDict, ProcessedPreferenceChatDict], DpoExample]
):
    """
    A dataset that yields preference pairs as DpoExample objects.

    Args:
        cache: The cache of processed preference chat data.
        Pos: The position axis.
        max_segments_per_example: The maximum number of segments to pack into a single example. Set to 1 to disable packing.
        slice_strategy: The strategy to use when an example is too long.
        mask_user_turns: Whether to mask user tokens from the loss.
    """

    def __init__(
        self,
        cache: TreeCache[ProcessedPreferenceChatDict],
        Pos: Axis,
        *,
        max_segments_per_example: int = 1,
        slice_strategy: Literal["left", "right", "raise", "drop"] = "raise",
        mask_user_turns: bool = True,
    ):
        self.packed: GreedyPrepackedDataset[ProcessedPreferenceChatDict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos
        self.mask_user_turns = mask_user_turns

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit)
        def _create_dpo_example(
            e: tuple[ProcessedPreferenceChatDict, ProcessedPreferenceChatDict],
        ) -> DpoExample:
            example, seg_ids = e

            def build_one(prefix: Literal["chosen", "rejected"]) -> LmExample:
                if prefix == "chosen":
                    tokens = hax.named(example["chosen_input_ids"], self.Pos)
                    seg = hax.named(seg_ids["chosen_input_ids"], self.Pos)
                    mask = hax.named(example["chosen_assistant_masks"], self.Pos)
                else:
                    tokens = hax.named(example["rejected_input_ids"], self.Pos)
                    seg = hax.named(seg_ids["rejected_input_ids"], self.Pos)
                    mask = hax.named(example["rejected_assistant_masks"], self.Pos)

                if self.mask_user_turns:
                    loss_weight = hax.roll(mask, shift=-1, axis=self.Pos)
                else:
                    loss_weight = None

                return LmExample.causal(tokens=tokens, loss_weight=loss_weight, segment_ids=seg)

            out = DpoExample(chosen=build_one("chosen"), rejected=build_one("rejected"))
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_dpo_example)


def preprocessor_for_preference_format(
    format: PreferenceChatLmDatasetFormat, tokenizer: HfTokenizer
) -> PreferenceChatProcessor:
    """Create a preprocessor for preference chat format."""
    return PreferenceChatProcessor(
        tokenizer,
        chosen_field=format.chosen_field,
        rejected_field=format.rejected_field,
        chat_template=format.chat_template,
        system_prompt_field=format.system_prompt,
        chat_template_kwargs_field=format.chat_template_kwargs,
        mask_user_turns=format.mask_user_turns,
    )


def dataset_for_preference_format(
    format: PreferenceChatLmDatasetFormat,
    Pos: Axis,
    cache: TreeCache[ProcessedPreferenceChatDict],
) -> PreferencePairDataset:
    """Create a dataset for preference chat format."""
    return PreferencePairDataset(
        cache,
        Pos,
        max_segments_per_example=64 if format.pack else 1,
        mask_user_turns=format.mask_user_turns,
        slice_strategy=format.slice_strategy,
    )
