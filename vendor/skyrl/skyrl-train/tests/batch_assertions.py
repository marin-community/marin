import pytest


def assert_metric_parity(actual: dict, expected: dict, *, required: set[str]):
    execution_prefixes = ("timing/", "perf/", "async/performance/", "system/")
    actual_keys = {key for key in actual if not key.startswith(execution_prefixes)}
    expected_keys = {key for key in expected if not key.startswith(execution_prefixes)}
    assert actual_keys == expected_keys
    for key in required:
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    for key in actual_keys - required:
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7, nan_ok=True), key
