import pytest


WORKER_OMITTED_METRICS = {
    f"policy/mismatch/{scope}/{key}"
    for scope in ("pooled", "staleness0")
    for key in (
        "log_ratio_abs_p95",
        "log_ratio_abs_p99",
        "log_ratio_abs_p999",
        "frac_below_1e_5",
        "ess_fraction",
        "kl_k1",
        "kl_k3",
        "chi2",
    )
} | {"policy/rollout_train_prob_diff_std"}


def assert_metric_parity(actual: dict, expected: dict, *, required: set[str], absent: set[str]):
    execution_prefixes = ("timing/", "perf/", "async/performance/", "system/")
    actual_keys = {key for key in actual if not key.startswith(execution_prefixes)}
    expected_keys = {key for key in expected if not key.startswith(execution_prefixes)}
    assert actual_keys == expected_keys - absent
    for key in required:
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    for key in actual_keys - required:
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7, nan_ok=True), key
