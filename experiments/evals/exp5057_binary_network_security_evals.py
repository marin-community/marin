# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5057 binary/network/security raw-text eval slices.

This first pass keeps scope intentionally narrow: materialize a small Zeek-flow sample from the
listable UWF CSV export so perplexity-gap runs can measure structured network-log surface forms
without downloading a broad security corpus.

Issue: https://github.com/marin-community/marin/issues/5057 (parent #5005).
"""

from __future__ import annotations

import posixpath

from marin.datakit.download.uwf_zeek import UwfZeekSampleSource, uwf_zeek_sample_step
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.step_spec import StepSpec

ISSUE_5057 = 5057
EPIC_5005 = 5005
UWF_ZEEK_SLICE_KEY = "binary_network_security/uwf_zeek"

UWF_ZEEK_SOURCE = UwfZeekSampleSource(slice_key=UWF_ZEEK_SLICE_KEY)
UWF_ZEEK_RAW = uwf_zeek_sample_step(UWF_ZEEK_SOURCE)


def binary_network_security_raw_validation_sets(
    *,
    raw_root: str | None = None,
    binary_network_security_raw: StepSpec | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return the first-pass binary/network/security raw validation slices."""

    if raw_root is None and binary_network_security_raw is None:
        binary_network_security_raw = UWF_ZEEK_RAW

    if raw_root is not None:
        source = posixpath.join(raw_root, "binary/uwf/zeek.jsonl.gz")
    else:
        assert binary_network_security_raw is not None
        source = binary_network_security_raw.as_executor_step().cd("data.jsonl.gz")

    return {
        UWF_ZEEK_SLICE_KEY: raw_text_dataset(
            source,
            tags=("binary_network_security", f"epic:{EPIC_5005}", f"issue:{ISSUE_5057}", UWF_ZEEK_SLICE_KEY),
        )
    }
