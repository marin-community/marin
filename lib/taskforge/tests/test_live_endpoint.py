# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live check that the configured GLM endpoint accepts the high-pool token and serves glm-5.3."""

import httpx
import pytest

from taskforge.llm.test_support import GLM_MODEL


@pytest.mark.live_glm
def test_endpoint_serves_glm_model(glm_source):
    endpoint = glm_source.current()
    response = httpx.get(
        f"{endpoint.base_url}/models", headers={"Authorization": f"Bearer {endpoint.token}"}, timeout=30
    )
    response.raise_for_status()
    assert GLM_MODEL in {m["id"] for m in response.json()["data"]}
