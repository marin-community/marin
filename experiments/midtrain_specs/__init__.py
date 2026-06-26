# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi-specific midtraining cell specs.

The ``data_sections/`` subdirectory contains JSON copies of the ``data:``
block from canonical 1e21 K=0.20 midtraining runs (one per mix). They are
the source of truth for math-validation-set identity: midtraining for
smaller bases at K=0.20 must use these blocks verbatim so the held-out
math val partition is byte-identical to the 1e21/1e22 sweep.

Provenance per file:

| mix     | source run                                                |
|---------|-----------------------------------------------------------|
| p33m67  | gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.5-efbc63 |
| p50m50  | gs://marin-us-east5/checkpoints/delphi-1e21-p50m50-9p25b-lr0.5-973c46 |
| p67m33  | gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49 |

All three share:
- math cache: ``gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519``
- ``num_validation_sequences``: ``{"nemotron_cc_math_v1/4plus": 12500}``
- ``shuffle_before_trainval_split``: True
- ``shuffle``: ``{io_block_size: 256, window_blocks: 512, perm_type: feistel}``

Only ``train_weights`` differs across mixes.
"""

import json
from pathlib import Path
from typing import Any

_DATA_SECTIONS_DIR = Path(__file__).parent / "data_sections"

# Canonical mix names (must match a JSON file in data_sections/).
DELPHI_MIDTRAIN_MIXES: tuple[str, ...] = ("p33m67", "p50m50", "p67m33")

# Source-of-truth identity (used by tests/validators).
MATH_CACHE_DIR: str = "gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519"
MATH_VAL_SEQUENCES: int = 12_500


def load_legacy_data_section(mix: str) -> dict[str, Any]:
    """Load the canonical ``data:`` block for one mix.

    The returned dict is the verbatim Levanter data config from the 1e21
    K=0.20 reference run for ``mix``. Passing it as
    ``MidtrainSpec.data_section_override`` (with provenance) makes the
    rendered Levanter YAML use bit-identical val carve-out to the
    reference.
    """
    if mix not in DELPHI_MIDTRAIN_MIXES:
        raise ValueError(f"Unknown mix {mix!r}. Registered Delphi midtrain mixes: {DELPHI_MIDTRAIN_MIXES}")
    path = _DATA_SECTIONS_DIR / f"{mix}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# Provenance string passed to MidtrainSpec.data_section_provenance.
LEGACY_PROVENANCE: dict[str, str] = {
    "p33m67": "legacy:delphi-1e21-p33m67-9p25b-lr0.5-efbc63",
    "p50m50": "legacy:delphi-1e21-p50m50-9p25b-lr0.5-973c46",
    "p67m33": "legacy:delphi-1e21-p67m33-9p25b-lr0.5-114e49",
}
