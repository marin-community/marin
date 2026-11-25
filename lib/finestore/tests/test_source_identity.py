# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stored records written under historical IDs resolve where they are read."""

import json
from pathlib import Path

from finestore.rl.source_identity import SOURCE_ID_ALIASES, canonical_source_id

RENAME_JOIN = json.loads(Path(__file__).with_name("source_id_renames.json").read_text())


def test_a_stored_historical_id_resolves_to_its_generated_identity():
    assert canonical_source_id("MarinSkyRL:eurus2_code") == "PRIME-RL:eurus2_code"
    assert canonical_source_id("Task Trove:laion__codeforces-v3") == "TaskTrove:laion__codeforces-v3"


def test_every_renamed_published_id_resolves_to_its_generated_identity():
    """The table covers the complete join of main's published declarations onto
    their generated identities: every renamed ID resolves to its exact target."""
    unresolved = {old: canonical_source_id(old) for old, new in RENAME_JOIN.items() if canonical_source_id(old) != new}
    assert unresolved == {}


def test_an_alias_target_resolves_to_itself():
    """Aliases are permanent and single-step: no target is itself a historical ID."""
    chained = {target for target in SOURCE_ID_ALIASES.values() if canonical_source_id(target) != target}
    assert chained == set()


def test_an_unlisted_id_stays_itself():
    assert canonical_source_id("Marin:new-source") == "Marin:new-source"
