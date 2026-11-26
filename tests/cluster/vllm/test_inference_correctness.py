# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the shared checker against an existing dedicated vLLM endpoint."""

import pytest
from marin.testing.inference.results import CheckStatus


@pytest.mark.manual
@pytest.mark.asyncio
@pytest.mark.timeout(10800)
async def test_candidate_matches_shared_correctness_policy(correctness_endpoint, correctness_spec, request):
    async with correctness_endpoint as endpoint:
        report = await correctness_spec.run(endpoint)
    output = request.config.getoption("--correctness-output")
    if output:
        report.save(output)
    assert report.status == CheckStatus.PASSED, {
        name: {"status": getattr(report, name).status, "details": getattr(report, name).details}
        for name in ("reference", "batch", "cache", "decode")
    }
