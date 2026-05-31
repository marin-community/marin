# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-tail formal-methods and hardware-description PPL slices backed by public datasets."""

from __future__ import annotations

import posixpath

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import EPIC_5005, FORMAL_HARDWARE_ISSUE, LongTailPplFamily

FAMILY = LongTailPplFamily.FORMAL_HARDWARE


def _registry_key(name: str) -> str:
    return posixpath.join("long_tail_ppl", FAMILY.value, name)


def _tags(split: str) -> tuple[str, ...]:
    return ("long_tail_ppl", f"epic:{EPIC_5005}", f"issue:{FORMAL_HARDWARE_ISSUE}", FAMILY.value, f"split:{split}")


def formal_hardware_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return {
        _registry_key("verilogeval_prompt"): raw_text_dataset(
            HfDatasetSpec(id="dakies/nvlabs-verilogeval"),
            text_key="prompt",
            split="test",
            tags=_tags("test"),
        ),
        _registry_key("verilogeval_canonical_solution"): raw_text_dataset(
            HfDatasetSpec(id="dakies/nvlabs-verilogeval"),
            text_key="canonical_solution",
            split="test",
            tags=_tags("test"),
        ),
    }
