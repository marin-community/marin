# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Look up sources retired from admission in the historical ledger."""

from functools import cache
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id

from experiments.post_training.task_curation.registry import read_registry_json

RETIREMENTS_PATH = Path(__file__).with_name("retired-inventory-only.json")


@cache
def retired_admission_source_ids() -> frozenset[str]:
    return frozenset(
        canonical_source_id(row["id"])
        for row in read_registry_json(RETIREMENTS_PATH)["entries"]
        if row.get("disposition") == "retired_admission_check"
    )
