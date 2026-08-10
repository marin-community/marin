# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec

# Extracted indexed HTML response byte ranges from all 4,573 CC-SUPPLEMENTAL-2026-22
# WARCs with XenonMolecule/jusText@20d27c00ebfbe927f86281933da687d3e636cba3,
# using its sklearn model and English stoplist. experiments/datakit/focus_crawl.py
# wrote the resulting text and WARC provenance as normalized Parquet.
_FOCUS_CRAWL_ARTIFACT = "s3://marin-us-east-02a/marin/data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9"


def common_crawl_focus_normalize_steps() -> tuple[StepSpec, ...]:
    """Return steps that read the existing Focus Crawl normalized artifact."""
    return (
        StepSpec(
            name="normalized/common-crawl-focus-2026-22",
            hash_attrs={"artifact": _FOCUS_CRAWL_ARTIFACT},
            fn=lambda _output_path: read_artifact(_FOCUS_CRAWL_ARTIFACT, NormalizedData),
        ),
    )
