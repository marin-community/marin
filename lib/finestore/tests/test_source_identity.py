# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stored records written under historical IDs resolve where they are read."""

from finestore.rl.source_identity import canonical_source_id


def test_a_stored_historical_id_resolves_to_its_generated_identity():
    assert canonical_source_id("MarinSkyRL:eurus2_code") == "PRIME-RL:eurus2_code"
    assert canonical_source_id("Task Trove:laion__codeforces-v3") == "TaskTrove:laion__codeforces-v3"


def test_an_unlisted_id_stays_itself():
    assert canonical_source_id("Marin:new-source") == "Marin:new-source"
