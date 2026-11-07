# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskforge_adapters.campaign import OpenAIModelPort, endpoint_source, provider_config
from taskforge_adapters.glm_client import FixedEndpoint, Pool, RelayEndpoint


def test_fixed_model_port_builds_fixed_endpoint():
    port = OpenAIModelPort("model", "TOKEN", base_url="https://example.test/v1", pool=Pool.HIGH)

    source = endpoint_source(port, "secret")

    assert isinstance(source, FixedEndpoint)
    assert source.current().base_url == "https://example.test/v1"
    assert source.current().pool is Pool.HIGH
    assert provider_config(port).model == "model"


def test_relay_model_port_preserves_service_name():
    port = OpenAIModelPort("model", "TOKEN", relay_job="/owner/relay", relay_service="served-model")

    source = endpoint_source(port, "secret")

    assert isinstance(source, RelayEndpoint)
    assert source.relay_job == "/owner/relay"
    assert source.endpoint_name == "served-model"


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"base_url": "https://example.test/v1", "relay_job": "/owner/relay", "relay_service": "service"},
        {"base_url": "https://example.test"},
        {"relay_job": "/owner/relay"},
    ),
)
def test_model_port_rejects_ambiguous_or_incomplete_routes(kwargs):
    with pytest.raises(ValueError):
        OpenAIModelPort("model", "TOKEN", **kwargs)
