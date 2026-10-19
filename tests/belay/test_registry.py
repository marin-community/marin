# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from experiments.belay import registry as R
from scripts.ci.belay_intake import IntakeComment, registry_entry


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
                id="t001",
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


@pytest.mark.parametrize(
    "trial_ids", [("t1",), ("t01",), ("T001",), ("001",), ("t1000",), ("t001", "t001"), ("t002", "t001")]
)
def test_validate_rejects_malformed_repeated_or_out_of_order_trial_ids(trial_ids):
    (trial,) = _experiment().trials
    trials = tuple(dataclasses.replace(trial, id=i) for i in trial_ids)
    with pytest.raises(ValueError):
        R.validate([_experiment(trials=trials)])


def test_validate_accepts_gaps_in_trial_ids():
    (trial,) = _experiment().trials
    R.validate([_experiment(trials=(trial, dataclasses.replace(trial, id="t003")))])


def test_load_names_the_file_and_the_missing_key(tmp_path):
    text = R.write_experiment(_experiment(), tmp_path).read_text().replace("- id: t001\n  variant", "- variant")
    (tmp_path / "X0001-sqrt-softplus-router.yaml").write_text(text)
    with pytest.raises(ValueError, match=r"X0001-sqrt-softplus-router.yaml: Trial: missing keys \['id'\]"):
        R.load_registry(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"commit": "e8cd1b9"},
        {"commit": "E8CD1B97EC"},
        {"commit": "e8cd1b97ec133ea54cd42ac520bf069ee659d2d3a"},
    ],
)
def test_validate_rejects_malformed_commits(overrides):
    (trial,) = _experiment().trials
    with pytest.raises(ValueError):
        R.validate([_experiment(trials=(dataclasses.replace(trial, **overrides),))])


@pytest.mark.parametrize("commit", ["e8cd1b97ec", "e8cd1b97ec133ea54cd42ac520bf069ee659d2d3"])
def test_validate_accepts_short_and_full_commits(commit):
    (trial,) = _experiment().trials
    R.validate([_experiment(trials=(dataclasses.replace(trial, commit=commit),))])


def test_validate_rejects_duplicate_slugs():
    with pytest.raises(ValueError):
        R.validate([_experiment(), _experiment(id="X0002")])


def test_page_data_uses_the_page_field_names():
    idea = R.page_data([_experiment()], "test")["ideas"][0]
    assert idea["id"] == "sqrt-softplus-router" and idea["xid"] == "X0001"
    assert idea["trials"][0]["delta"] == [-0.001, 0.002]


def test_intake_bot_entries_are_valid_registry_files(tmp_path):
    comment = IntakeComment(
        comment_id=5,
        author="someone",
        url="https://github.com/o/r/issues/1#issuecomment-5",
        updated_at="2026-10-14T00:00:00Z",
        reactions=(),
        body="**Title:** Per-head router temperature\n**Category:** Vibes\n**Idea:** learn it\nmore text",
    )
    filename, text = registry_entry("X0042", comment, "ClassicLarry", frozenset(R.CATEGORIES))
    (tmp_path / filename).write_text(text)
    (experiment,) = R.load_registry(tmp_path)
    assert experiment.id == "X0042" and experiment.slug == "per-head-router-temperature"
    assert experiment.category == "Unsorted"  # not one of the categories
    assert experiment.status is R.Status.APPROVED and experiment.source is R.Source.INTAKE
    assert experiment.intake == R.Intake(url=comment.url, author="someone", approved_by="ClassicLarry")
    assert experiment.summary == "learn it"


def test_placeholder_bpb_scales_losses_and_signed_deltas_only():
    trial = R.Trial(
        id="t001",
        variant="beta (0.95, 0.999), sigma 0.005",
        rung="d512",
        protocol="equal_steps",
        batch="1",
        date="2026-10-14",
        comment_url="u",
        baseline="df1-ctrl 2.9544",
        delta=(-0.0088,),
        loss=(3.0,),
        note="mean -0.0088, k 1.036x",
    )
    (idea,) = R.page_data([_experiment(trials=(trial,))], "test", placeholder_bpb_ratio=0.5)["ideas"]
    (t,) = idea["trials"]
    assert t["baseline"] == "df1-ctrl 1.4772" and t["delta"] == [-0.0044] and t["loss"] == [1.5]
    assert t["note"] == "mean -0.0044, k 1.036x"
    assert t["variant"] == "beta (0.95, 0.999), sigma 0.005"


def test_week_summary_lists_trials_and_decisions_in_range():
    decided = R.Event(type=R.EventType.ADOPTED, batch="330", date="2026-10-15", comment_url="u", note="in Ckpt 2")
    tried = _experiment()
    adopted = _experiment(id="X0002", slug="other", name="Other", trials=(), events=(decided,))
    text = R.week_summary([tried, adopted], "2026-10-12", "2026-10-19")
    assert "**X0001** Sqrt-softplus router — 1 trial (d512)" in text
    assert "**X0002** Other — adopted: in Ckpt 2" in text
    assert R.week_summary([tried], "2026-10-19", "2026-10-26").startswith("No experiments")
