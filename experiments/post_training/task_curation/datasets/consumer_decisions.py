# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit per-source user decisions; consumer checks never write this registry."""

import json
from functools import cache
from pathlib import Path

from finestore.rl.curation import ConsumerDecision


@cache
def consumer_decisions() -> dict[str, ConsumerDecision]:
    document = json.loads(Path(__file__).with_suffix(".json").read_text())
    decisions = {source_id: ConsumerDecision.model_validate(row) for source_id, row in document.items()}
    if any(source_id != decision.source_id for source_id, decision in decisions.items()):
        raise ValueError("Consumer decision registry contains a mismatched source")
    return decisions
