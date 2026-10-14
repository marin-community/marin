import pandas as pd
import pytest

from experiments.domain_phase_mix.release import keys, tables

BUCKETS = ["a", "b", "c"]


def test_mixture_id_ignores_sub_1e9_noise_and_column_order():
    w1 = {"a": 0.5, "b": 0.3, "c": 0.2}
    w2 = {"c": 0.2 + 1e-12, "b": 0.3, "a": 0.5 - 1e-12}
    assert keys.mixture_id(w1) == keys.mixture_id(w2)
    assert keys.mixture_id(w1) != keys.mixture_id({"a": 0.5, "b": 0.2, "c": 0.3})


def test_exposure_id_separates_pool_fractions():
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    full = {"a": 1.0, "b": 1.0, "c": 1.0}
    half = {"a": 1.0, "b": 0.5, "c": 1.0}
    assert keys.exposure_id(w, full, "v1") != keys.exposure_id(w, half, "v1")
    assert keys.exposure_id(w, full, "v1") == keys.exposure_id(dict(reversed(w.items())), full, "v1")


def test_schedule_id_is_order_sensitive():
    a, b = {"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6}
    assert keys.schedule_id(a, b, 2400, 3007) != keys.schedule_id(b, a, 2400, 3007)


def test_observation_registry_is_stable_and_opaque(tmp_path):
    reg = keys.ObservationRegistry(salt="unit-test-salt", map_path=tmp_path / "map.csv")
    first = reg.id_for("lwspu_t9_snc_cap08_3e18-6c15f1")
    assert first == reg.id_for("lwspu_t9_snc_cap08_3e18-6c15f1")
    assert first.startswith("obs_") and len(first) == 16 and "6c15f1" not in first
    reg.save()
    reloaded = keys.ObservationRegistry(salt="unit-test-salt", map_path=tmp_path / "map.csv")
    assert reloaded.id_for("lwspu_t9_snc_cap08_3e18-6c15f1") == first


def test_add_aggregates_uses_given_weights():
    df = pd.DataFrame({"bpb__u1": [1.0, 2.0], "bpb__u2": [3.0, 4.0], "bpb__t1": [0.5, 0.5]})
    out = tables.add_aggregates(df, {"uncheatable_fixed": {"u1": 0.25, "u2": 0.75}, "obe_easy": {"t1": 1.0}})
    assert out["bpb__uncheatable_fixed"].tolist() == pytest.approx([2.5, 3.5])
    assert out["bpb__obe_easy"].tolist() == [0.5, 0.5]
