# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live check that the configured GLM endpoint accepts the interactive token and serves glm-5.3."""

import httpx
import pytest


@pytest.mark.live_glm
def test_endpoint_serves_glm_model(glm_settings):
    response = httpx.get(
        f"{glm_settings.base_url}/models",
        headers={"Authorization": f"Bearer {glm_settings.token}"},
        timeout=30,
    )
    response.raise_for_status()
    assert glm_settings.model in {m["id"] for m in response.json()["data"]}
