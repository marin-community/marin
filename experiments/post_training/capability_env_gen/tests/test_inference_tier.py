from capability_pipeline.inference import GLMClient


def test_bulk_tier_lowers_every_request_to_bulk():
    client = GLMClient(base_url="http://relay", token="t", tier="bulk")
    headers = client.headers()
    assert headers["x-priority"] == "bulk"
    assert headers["Authorization"] == "Bearer t"


def test_interactive_tier_sends_no_priority_override():
    client = GLMClient(base_url="http://relay", token="t", tier="interactive")
    assert "x-priority" not in client.headers()
