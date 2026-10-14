# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.belay import registry as R


def _experiment(**overrides) -> R.Experiment:
    fields = dict(
        id="X0001",
        slug="sqrt-softplus-router",
        name="Sqrt-softplus router",
        category="MoE and routing",
        summary="Combine routed experts with sqrt(softplus(logit)).",
        source=R.Source.AGENT,
        status=R.Status.DONE,
        trials=(
            R.Trial(
                variant="raw",
                rung="d512",
                protocol="equal_steps",
                batch="327",
                date="2026-10-13",
                comment_url="https://example.com/c",
                delta=(-0.001, 0.002),
                settings={"router_combine": "sqrt_softplus_raw"},
            ),
        ),
    )
    fields.update(overrides)
    return R.Experiment(**fields)


def test_checked_in_registry_validates():
    experiments = R.load_registry()
    assert experiments
    assert [e.id for e in experiments] == sorted(e.id for e in experiments)


def test_write_then_load_round_trips(tmp_path):
    experiment = _experiment()
    path = R.write_experiment(experiment, tmp_path)
    assert R.load_registry(tmp_path) == [experiment]
    assert path.name == "X0001-sqrt-softplus-router.yaml"


def test_next_id_follows_the_highest():
    assert R.next_id([]) == "X0001"
    assert R.next_id([_experiment(id="X0009"), _experiment(id="X0041", slug="other")]) == "X0042"


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "E0001"},
        {"slug": "Not A Slug"},
        {"category": "Vibes"},
        {"source": R.Source.INTAKE},
    ],
)
def test_validate_rejects_malformed_entries(overrides):
    with pytest.raises(ValueError):
        R.validate([_experiment(**overrides)])


def test_validate_rejects_duplicate_slugs():
    with pytest.raises(ValueError):
        R.validate([_experiment(), _experiment(id="X0002")])


def test_page_data_uses_the_page_field_names():
    idea = R.page_data([_experiment()], "test")["ideas"][0]
    assert idea["id"] == "sqrt-softplus-router" and idea["xid"] == "X0001"
    assert idea["trials"][0]["delta"] == [-0.001, 0.002]
