# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source retirements preserve declarations and artifacts in the historical ledger."""

import json
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id

RETIRED_ADMISSION_SOURCE_IDS = frozenset(
    canonical_source_id(row["id"])
    for row in json.loads(Path(__file__).with_name("retired-inventory-only.json").read_text())["entries"]
    if row.get("disposition") == "retired_admission_check"
)
