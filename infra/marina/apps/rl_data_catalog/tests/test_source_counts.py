# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.quality import quality_display, review_completeness
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence
from infra.marina.apps.rl_data_catalog.tests.review_fixtures import review_declarations


@pytest.mark.parametrize(
    "name,blocked,remaining,admitted,state",
    [
        ("clean", 1, 2, 2, "static_retained"),
        ("eval", 1, 2, 2, "static_retained"),
        ("clean", 3, 0, 0, "static_blocked"),
    ],
)
def test_no_oracle_counts_subtract_unique_blockers_and_separate_eval(
    name: str, blocked: int, remaining: int, admitted: int, state: str
) -> None:
    source = next(source for _, source in contamination_declarations() if source.name == name)
    source = with_declaration_admission_census(source)
    observation = count_evidence(source, oracle_tasks=0, blocked=blocked)
    result = filtration(source.model_dump(mode="json"), observation)
    # Both reasons affect the same task. Subtract the measured union only once.
    assert result["example_count"] == remaining
    assert result["admitted_count"] == admitted
    assert result["admitted_count"] + result["excluded_count"] == result["population_count"] == 3
    assert result["state"] == state
    assert result["count_evidence"]["policy"]["sha256"] == observation.policy.sha256


@pytest.mark.parametrize("changed", ["dataset_revision", "intended_use", "task_specs_sha256"])
def test_stale_count_evidence_never_carries_a_count_to_another_input(changed: str) -> None:
    source = with_declaration_admission_census(contamination_declarations()[0][1])
    observation = count_evidence(source, oracle_tasks=0, blocked=1)
    declaration = source.model_dump(mode="json")
    if changed == "task_specs_sha256":
        declaration["actor_census_input"][changed] = "e" * 64
    else:
        declaration[changed] = "eval" if changed == "intended_use" else "another-revision"
    result = filtration(declaration, observation)
    assert result["example_count"] is None
    assert result["admitted_count"] is None


def test_oracle_inventory_and_missing_static_audit_do_not_establish_counts() -> None:
    source = with_declaration_admission_census(contamination_declarations()[0][1])
    oracle = filtration(source.model_dump(mode="json"), count_evidence(source, oracle_tasks=1, blocked=None))
    static = filtration(source.model_dump(mode="json"), count_evidence(source, oracle_tasks=0, blocked=None))
    assert oracle["state"] == "pending_oracle" and oracle["example_count"] is None
    assert static["state"] == "pending_static" and static["example_count"] is None


@pytest.mark.parametrize("provisional,declaration", review_declarations())
def test_sample_tolerance_display_preserves_counts_and_full_finality(provisional, declaration):
    completeness = review_completeness(declaration, [], [])
    assert completeness is not None
    assert completeness["provisional"] is provisional
    assert completeness["reviewed"] == declaration.quality_result.reviewed
    assert completeness["required"] == declaration.quality_result.sample_count
    assert completeness["coverage_complete"] is (declaration.quality_result.unavailable_count == 0)
    if declaration.name.startswith("sample-missing-"):
        assert quality_display(declaration.model_dump(mode="json"), None, None).quality.value == "SAMPLE"
        assert not completeness["coverage_complete"]
    if declaration.name == "full-waiver-over-cap":
        # Above the 1% unreviewed bound the waiver is not whole-source finality.
        displayed = quality_display(declaration.model_dump(mode="json"), None, None)
        assert displayed.quality.value != "FULL"
        assert displayed.quality.value == "SAMPLE"


def test_sample_coverage_display_uses_the_recorded_oversight_coverage() -> None:
    declaration = {
        "id": "Marin:recovered",
        "oversight": "sample",
        "oversight_evidence": {"level": "sample", "reviewed": 212767, "verification": "carried evidence"},
        "within_source_dedup": {"unique_count": 438810},
        "quality_result": {"reviewed": 96, "sample_count": 100},
    }
    displayed = quality_display(declaration, None, None)
    assert displayed.quality.value == "SAMPLE"
    assert displayed.review_status == "SAMPLE, 212,767 of 438,810 reviewed"


def test_sample_coverage_display_stays_bare_without_recorded_coverage() -> None:
    declaration = {
        "id": "Marin:plain",
        "oversight": "sample",
        "oversight_evidence": {"level": "sample", "reviewed": None, "verification": "panel only"},
        "within_source_dedup": {"unique_count": 438810},
        "quality_result": {"reviewed": 96, "sample_count": 100},
    }
    displayed = quality_display(declaration, None, None)
    assert displayed.quality.value == "SAMPLE"
    assert displayed.review_status is None


@pytest.mark.parametrize(
    "status,admitted,state",
    [("Excluded", 0, "excluded"), ("Deprecated", 0, "excluded"), ("Pending", None, "pending")],
)
def test_headline_counts_follow_the_source_status(status: str, admitted: int | None, state: str) -> None:
    """Only Available sources admit; excluded/deprecated count zero, pending stays unmeasured."""
    source = with_declaration_admission_census(contamination_declarations()[0][1])
    observation = count_evidence(source, oracle_tasks=0, blocked=1)
    declaration = source.model_dump(mode="json")
    declaration["status"] = status
    result = filtration(declaration, observation)
    assert result["admitted_count"] == admitted
    assert result["state"] == state
    assert result["source_status"] == status
    # the retained projection stays visible as evidence for every status
    assert "retains 2 tasks" in result["reason"]
    assert result["population_count"] == 3


def test_admitted_headline_sums_available_sources_only() -> None:
    """The served admitted total is the sum over Available sources only."""
    sources = {
        status: with_declaration_admission_census(contamination_declarations()[0][1])
        for status in ("Available", "Excluded", "Pending", "Deprecated")
    }
    counts = {status: count_evidence(source, oracle_tasks=0, blocked=1) for status, source in sources.items()}
    admitted = []
    for status, source in sources.items():
        declaration = source.model_dump(mode="json")
        declaration["status"] = status
        admitted.append(filtration(declaration, counts[status])["admitted_count"])
    # Available admits 2; Excluded/Deprecated contribute 0; Pending contributes nothing
    total = sum(value or 0 for value in admitted)
    assert total == 2
    assert admitted == [2, 0, None, 0]
