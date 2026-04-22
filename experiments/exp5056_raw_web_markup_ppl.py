# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
#5056: Raw web, markup, and image-text PPL slices.

Parent: #5005.

Byte-level perplexity-gap eval slices that preserve surface syntax normally
stripped by cleaned web corpora (HTML, WARC/WAT metadata, web tables, SVG XML,
OCR strings, captions, alt-text, EXIF-like metadata, URL-heavy records). The
goal is to surface perplexity-gap buckets that cleaned-corpus slices
(Paloma / uncheatable-eval) hide.

Targets (tracked in #5056):
  - Common Crawl WARC/WAT: raw HTTP headers, raw HTML, WAT JSON.
  - Web Data Commons Web Tables: raw <table> HTML plus extracted JSON metadata.
  - SVG-Stack: SVG XML programs and captions.
  - TextOCR / OCR-VQA: OCR strings and scene-text annotations.
  - LAION metadata: URL / alt-text / EXIF-like fields — deferred pending
    explicit subset selection and safety-filter review.

Per #5056 design review, slices are oversplit by surface form (one entry per
surface form, e.g. `raw_web_markup/cc_warc_html`, `raw_web_markup/cc_wat_json`)
so the gap-finder bucket analysis in `marin/evaluation/perplexity_gap.py`
stays clean. Grouping happens post-hoc via tags.

This module is intentionally a registration point: downloaders land in
follow-up PRs and populate ``ACTIVE_RAW_WEB_MARKUP_DATASETS``. The aggregator
``raw_web_markup_raw_validation_sets()`` is wired into
``experiments/defaults.py::default_raw_validation_sets()`` so new slices flow
into ``exp_model_perplexity_gap_marin_vs_llama.py`` and its siblings without
touching any other file.
"""

import posixpath
from typing import Any

# Prefix applied to every slice name so the gap-finder report groups the new
# surface-preserving slices together. Top-level constant per CLAUDE.md.
RAW_WEB_MARKUP_PREFIX = "raw_web_markup"

# Populated by follow-up PRs. Keys are slice names relative to
# ``RAW_WEB_MARKUP_PREFIX`` (e.g. ``cc_warc_html``); values are
# ``RawTextEvaluationDataset`` instances produced via
# ``marin.evaluation.perplexity_gap.raw_text_dataset``.
#
# Convention: each source contributes one entry per surface form. Do not
# concatenate surfaces into a single ``text`` stream — the gap-finder truncates
# each doc to ``max_doc_bytes=32_768`` and reports per-slice bpb, so mixing
# surfaces inside a slice loses the signal we want.
ACTIVE_RAW_WEB_MARKUP_DATASETS: dict[str, Any] = {}


def raw_web_markup_raw_validation_sets() -> dict[str, Any]:
    """Return raw-text eval slices covering web markup and image-adjacent text.

    Slice names are prefixed with :data:`RAW_WEB_MARKUP_PREFIX`. Returns an
    empty mapping until downloaders land in the follow-up PRs tracked by
    #5056; callers should treat an empty result as "no raw-web-markup slices
    are registered yet", not as an error.
    """
    return {
        posixpath.join(RAW_WEB_MARKUP_PREFIX, slice_name): dataset
        for slice_name, dataset in ACTIVE_RAW_WEB_MARKUP_DATASETS.items()
    }
