# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proposal documents for tests, with one shared set of stable header defaults."""

import json
from collections.abc import Mapping
from typing import Any


def proposal_document(*, proposal_id: str, source: Mapping[str, str], body: str, **header: Any) -> str:
    """Render a proposal fixture; callers supply the fields that matter to their scenario."""
    return json.dumps(
        {
            "header": {
                "id": proposal_id,
                "source": dict(source),
                "grounding": "unverified",
                "null_reason": None,
                "environment": "reasoning",
                "solver_machine": "shellsim",
                "verification": "code",
                "composite": [],
                "research_need": "not_needed",
                "research": [],
                "build": [],
                "resources": [],
                **header,
            },
            "body": body,
        }
    )
