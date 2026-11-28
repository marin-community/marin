# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The v4 carry-forward planner resolves the earlier population through the census record.

Regression: the planner read `actor.input_artifact` — a field the landed ActorCensusReceipt
does not carry. The earlier population comes from the census record at the receipt input's
evidence_url. This test runs against the real sealed v4 rst receipt.
"""

import json
from pathlib import Path

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.sample_carry_forward_v4 import earlier_population_path

FIXTURE = Path(__file__).with_name("fixtures") / "actor_census_receipt_v4_rst.json"


def test_earlier_population_resolves_through_the_census_record():
    receipt = ActorCensusReceipt.model_validate(json.loads(FIXTURE.read_text()))
    assert receipt.input.dataset_revision == "269ce8ab72c2002fa24c19ef70110651f157da10"
    population = earlier_population_path(receipt)
    assert population.endswith("rst-repaired-admission-v4/rst-population")
