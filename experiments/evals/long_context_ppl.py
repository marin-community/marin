# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context reading and retrieval PPL slices.

Issue #5825 (parent #5819) calls for long-context validation coverage that
moves before e19/e20 flops while staying cheap enough for periodic tracking,
and explicitly excludes AA-LCR documents/questions or other held-out
benchmark items from core tracking.

Two slice families are exposed here, both backed by public Hugging Face
mirrors that are not part of AA-LCR:

- **Raw long-doc PPL** (``raw_text_dataset``) over PG19 (pre-1919 books) and
  GovReport (long US government reports). These score full-document
  perplexity at the per-bundle length cap.
- **Target-only retrieval PPL** (``supervised_text_dataset``) over SCROLLS
  subsets — QASPER (academic-paper span QA), NarrativeQA (long-form QA over
  books/scripts), and QuALITY (long-doc MCQA). SCROLLS preprocesses each
  task into flat ``input``/``output`` fields, so the perplexity-gap loss
  mask falls on the answer span exactly.

The slice set is intentionally limited per #5825 — five slices, all reused
between the 32K default and the 64K opt-in tier in
``perplexity_gap_registry``. SEC EDGAR documents are deferred until the
TeraflopAI/SEC-EDGAR datakit source (#5305) lands so we share one
materialization.
"""

from __future__ import annotations

import posixpath

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

LONG_CONTEXT_EPIC = 5819
LONG_CONTEXT_ISSUE = 5825
FAMILY = "long_context"


def _registry_key(name: str) -> str:
    return posixpath.join(FAMILY, name)


def _tags(split: str, kind: str) -> tuple[str, ...]:
    return (
        FAMILY,
        f"epic:{LONG_CONTEXT_EPIC}",
        f"issue:{LONG_CONTEXT_ISSUE}",
        f"kind:{kind}",
        f"split:{split}",
    )


def long_context_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Raw long-document PPL slices.

    PG19 ships a ``test`` split with whole-book text. GovReport-summarization
    exposes the report body in the ``document`` column on its ``validation``
    split. Both are public, non-benchmark-derived, and disjoint from AA-LCR.
    """
    return {
        _registry_key("pg19_test"): raw_text_dataset(
            HfDatasetSpec(id="deepmind/pg19"),
            text_key="text",
            split="test",
            tags=_tags("test", "raw_long_doc"),
        ),
        _registry_key("govreport_validation"): raw_text_dataset(
            HfDatasetSpec(id="ccdv/govreport-summarization"),
            text_key="document",
            split="validation",
            tags=_tags("validation", "raw_long_doc"),
        ),
    }


def long_context_supervised_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Target-only retrieval PPL slices.

    Backed by SCROLLS subsets, which expose every task as flat
    ``input``/``output`` strings. The perplexity-gap scorer treats the
    ``input`` prefix as conditioning and computes per-byte loss only over
    the ``output`` span, giving a clean retrieval / answer-span signal.
    """
    return {
        _registry_key("scrolls_qasper"): supervised_text_dataset(
            HfDatasetSpec(id="tau/scrolls", name="qasper"),
            input_key="input",
            target_key="output",
            split="validation",
            tags=_tags("validation", "supervised_qa"),
        ),
        _registry_key("scrolls_narrative_qa"): supervised_text_dataset(
            HfDatasetSpec(id="tau/scrolls", name="narrative_qa"),
            input_key="input",
            target_key="output",
            split="validation",
            tags=_tags("validation", "supervised_qa"),
        ),
        _registry_key("scrolls_quality"): supervised_text_dataset(
            HfDatasetSpec(id="tau/scrolls", name="quality"),
            input_key="input",
            target_key="output",
            split="validation",
            tags=_tags("validation", "supervised_mcqa"),
        ),
    }


def long_context_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Union of raw and supervised long-context slices."""
    return {**long_context_raw_validation_sets(), **long_context_supervised_validation_sets()}
