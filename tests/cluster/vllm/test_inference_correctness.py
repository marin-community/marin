# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the shared checker against an existing dedicated vLLM endpoint."""

import asyncio

import pytest
from marin.testing.inference.endpoint_checks import check_endpoint


@pytest.mark.manual
@pytest.mark.timeout(10800)
def test_candidate_matches_shared_correctness_policy(correctness_endpoint):
    endpoint = correctness_endpoint
    report = asyncio.run(
        check_endpoint(
            endpoint.url,
            endpoint.model,
            endpoint.fixture,
            endpoint.runtime_source,
            endpoint.output,
        )
    )
    assert report["status"] == "passed", {
        "report": str(endpoint.output),
        "checks": {name: result["status"] for name, result in report["checks"].items()},
    }
