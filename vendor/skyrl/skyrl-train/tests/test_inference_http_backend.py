import pytest

from skyrl_train.inference_engines.inference_http_backend import OpenAIRequest
from skyrl_train.inference_engines.request_timing import RequestTimings


@pytest.mark.parametrize("header_name", ["x-request-id", "X-Request-Id", "X-REQUEST-ID"])
def test_http_request_identity_joins_engine_decode_evidence(header_name):
    request = OpenAIRequest({header_name: "skyrl-group-fixture-123", "x-data-parallel-rank": "3"})
    records = []
    timings = RequestTimings(records.append)
    timings.start(request.headers["x-request-id"], "prompt_0")
    engine_id = "chatcmpl-" + request.headers["X-Request-Id"]
    timings.observe(engine_id, [("QUEUED", 1), ("SCHEDULED", 2)], 1, 3, prompt_tokens=32768)
    timings.observe(engine_id, [], 60, 13, prompt_tokens=32768)
    timings.finish(request.headers["x-request-id"])
    assert request.headers["X-data-parallel-rank"] == "3"
    assert records[-1]["request_id"] == "skyrl-group-fixture-123"
    assert records[-1]["generated_tokens"] == 61
    assert records[-1]["context_tokens"] == 32829


def test_missing_http_request_ids_are_stable_within_a_request_and_unique_across_requests():
    first, second = OpenAIRequest({}), OpenAIRequest({})
    assert first.headers["x-request-id"] == first.headers["X-Request-Id"]
    assert first.headers["x-request-id"] != second.headers["x-request-id"]
