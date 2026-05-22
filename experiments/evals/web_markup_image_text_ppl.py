# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-tail web markup and image-text PPL slices backed by public datasets."""

from __future__ import annotations

import posixpath

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import EPIC_5005, WEB_RAW_ISSUE, LongTailPplFamily

FAMILY = LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT


def _registry_key(name: str) -> str:
    return posixpath.join("long_tail_ppl", FAMILY.value, name)


def _tags(split: str) -> tuple[str, ...]:
    return ("long_tail_ppl", f"epic:{EPIC_5005}", f"issue:{WEB_RAW_ISSUE}", FAMILY.value, f"split:{split}")


def web_markup_image_text_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return {
        _registry_key("svg_stack_val"): raw_text_dataset(
            HfDatasetSpec(id="starvector/svg-stack"),
            text_key="Svg",
            split="val",
            tags=_tags("val"),
        ),
        _registry_key("svg_stack_test"): raw_text_dataset(
            HfDatasetSpec(id="starvector/svg-stack"),
            text_key="Svg",
            split="test",
            tags=_tags("test"),
        ),
    }
