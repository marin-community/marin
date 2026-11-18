# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Route-keyed sampling orders: weighted mixtures and ordered stage sequences."""

from collections import Counter

import datasets
import pytest

from skyrl_train.domain_sampling import DomainWeightedOrder, OrderedStages, weighted_quotas


def _route_dataset(routes: list[str]) -> datasets.Dataset:
    return datasets.Dataset.from_dict({"teacher_route": routes})


def _route_of(dataset: datasets.Dataset, row: int) -> str:
    return str(dataset[row]["teacher_route"])


WEIGHTS = {"math": 0.7, "swe": 0.3}
ROUTES = ["math"] * 8 + ["swe"] * 4
STAGES = [{"source": "math", "quota": 3}, {"source": "swe", "quota": 2}]


def _weighted_order(dataset: datasets.Dataset) -> DomainWeightedOrder:
    return DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=6)


def _ordered_order(dataset: datasets.Dataset) -> OrderedStages:
    return OrderedStages(dataset, STAGES, seed=11, window_size=4)


@pytest.mark.parametrize("build", [_weighted_order, _ordered_order], ids=["domain-weighted", "ordered"])
def test_order_resumes_from_another_instance_state(build):
    """A fresh sampler restored from a checkpoint's state continues an uninterrupted stream exactly."""
    dataset = _route_dataset(ROUTES)
    whole = build(dataset)
    expected = [whole.next_index() for _ in range(18)]

    checkpoint = build(dataset)
    for _ in range(9):
        checkpoint.next_index()
    resumed = build(dataset)
    resumed.load_state_dict(checkpoint.state_dict())
    assert [resumed.next_index() for _ in range(9)] == expected[9:]


def test_domain_weighted_windows_compose_the_exact_weighted_quotas():
    """Each ten-prompt window contains seven math and three SWE prompts."""
    dataset = _route_dataset(ROUTES)
    order = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=10)
    for _ in range(4):
        drawn = [_route_of(dataset, order.next_index()) for _ in range(10)]
        assert Counter(drawn) == Counter({"math": 7, "swe": 3})


def test_ordered_stages_yield_the_declared_sequence():
    """Draws follow the declared stage order, quota per stage visit, deterministically."""
    dataset = _route_dataset(ROUTES)
    order = OrderedStages(dataset, STAGES, seed=7, window_size=4)
    labels = [_route_of(dataset, order.next_index()) for _ in range(20)]
    assert labels == ["math", "math", "math", "swe", "swe"] * 4
    replay = OrderedStages(dataset, STAGES, seed=7, window_size=4)
    assert [_route_of(dataset, replay.next_index()) for _ in range(20)] == labels


def test_ordered_stages_without_quota_draw_the_pool_row_count():
    dataset = _route_dataset(["a"] * 2 + ["b"] * 1)
    order = OrderedStages(dataset, [{"source": "a"}, {"source": "b"}], seed=3, window_size=2)
    labels = [_route_of(dataset, order.next_index()) for _ in range(6)]
    assert labels == ["a", "a", "b", "a", "a", "b"]


def test_ordered_stages_keep_window_draws_unique():
    dataset = _route_dataset(ROUTES)
    order = OrderedStages(dataset, STAGES, seed=11, window_size=4)
    for _ in range(5):
        window = [order.next_index() for _ in range(4)]
        assert len(set(window)) == 4


def test_ordered_stages_reject_unknown_route():
    dataset = _route_dataset(ROUTES)
    with pytest.raises(ValueError, match="routes the dataset does not have"):
        OrderedStages(dataset, [{"source": "web", "quota": 2}], seed=1, window_size=4)


def test_ordered_stages_reject_quota_larger_than_a_window_unique_visit():
    dataset = _route_dataset(["math"] * 2 + ["swe"] * 4)
    with pytest.raises(ValueError, match="cannot supply a window-unique stage visit"):
        OrderedStages(dataset, [{"source": "math", "quota": 3}, {"source": "swe", "quota": 2}], seed=1, window_size=4)


def test_weighted_quotas_rotates_the_remainder_across_windows():
    """The remainder lands on the highest-fraction route, ties rotating with the batch index."""
    assert weighted_quotas(10, {"a": 1.0, "b": 2.0}, 0) == {"a": 3, "b": 7}
    for index in range(3):
        quotas = weighted_quotas(2, {"a": 1.0, "b": 1.0, "c": 1.0}, index)
        assert sum(quotas.values()) == 2
        assert sorted(name for name, quota in quotas.items() if quota) != sorted(
            name for name, quota in weighted_quotas(2, {"a": 1.0, "b": 1.0, "c": 1.0}, index + 1).items() if quota
        )
