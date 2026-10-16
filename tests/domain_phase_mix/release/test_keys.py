# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import keys, tables

WEIGHTS = {"uncheatable": {"u1": 0.25, "u2": 0.75}}


def test_mixture_id_ignores_sub_1e9_noise_and_column_order():
    w1 = {"a": 0.5, "b": 0.3, "c": 0.2}
    w2 = {"c": 0.2 + 1e-12, "b": 0.3, "a": 0.5 - 1e-12}
    assert keys.mixture_id(w1) == keys.mixture_id(w2)
    assert keys.mixture_id(w1) != keys.mixture_id({"a": 0.5, "b": 0.2, "c": 0.3})


def test_mixture_id_treats_negative_zero_as_zero():
    assert keys.mixture_id({"a": 1.0, "b": 0.0}) == keys.mixture_id({"a": 1.0, "b": -1e-17})
    w = {"a": 1.0, "b": 0.0}
    assert keys.exposure_id(w, {"a": 1.0, "b": 0.0}, "v1") == keys.exposure_id(w, {"a": 1.0, "b": -0.0}, "v1")


def test_exposure_id_separates_pool_fractions():
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    full = {"a": 1.0, "b": 1.0, "c": 1.0}
    half = {"a": 1.0, "b": 0.5, "c": 1.0}
    assert keys.exposure_id(w, full, "v1") != keys.exposure_id(w, half, "v1")
    assert keys.exposure_id(w, full, "v1") == keys.exposure_id(dict(reversed(w.items())), full, "v1")


def test_schedule_id_is_order_sensitive():
    a, b = {"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6}
    assert keys.schedule_id(a, b, 2400, 3007) != keys.schedule_id(b, a, 2400, 3007)


def test_add_aggregates_uses_given_weights():
    df = pd.DataFrame({"bpb__u1": [1.0, 2.0], "bpb__u2": [3.0, 4.0], "bpb__t1": [0.5, 0.5]})
    out = tables.add_aggregates(df, {"uncheatable": {"u1": 0.25, "u2": 0.75}, "obe_easy": {"t1": 1.0}})
    assert out["bpb__uncheatable"].tolist() == pytest.approx([2.5, 3.5])
    assert out["bpb__obe_easy"].tolist() == [0.5, 0.5]


def test_add_aggregates_fills_missing_components_with_fixed_rule_records_only():
    # Rows: complete components; complete components and a record under other weights (legacy, pooled schema 2);
    # a missing component and a fixed-rule record; a missing component and no record.
    df = pd.DataFrame({"bpb__u1": [1.0, 1.0, np.nan, np.nan], "bpb__u2": [3.0, 3.0, 4.0, 4.0]})
    recorded = pd.DataFrame({"uncheatable": [2.5 + 1e-7, 2.51, 3.9, np.nan]})
    fixed_rule = pd.DataFrame({"uncheatable": [True, False, True, True]})
    out = tables.add_aggregates(df, WEIGHTS, recorded, fixed_rule)
    assert out.bpb__uncheatable.tolist()[:3] == [2.5, 2.5, 3.9] and np.isnan(out.bpb__uncheatable[3])


@pytest.mark.parametrize(
    "u1, record, fixed_rule",
    [(np.nan, 3.9, False), (1.0, 2.6, True)],
    ids=["needs_a_record_under_other_weights", "fixed_rule_record_disagrees_with_components"],
)
def test_add_aggregates_never_mixes_weightings(u1, record, fixed_rule):
    df = pd.DataFrame({"bpb__u1": [u1], "bpb__u2": [3.0]})
    with pytest.raises(ValueError):
        tables.add_aggregates(
            df, WEIGHTS, pd.DataFrame({"uncheatable": [record]}), pd.DataFrame({"uncheatable": [fixed_rule]})
        )
