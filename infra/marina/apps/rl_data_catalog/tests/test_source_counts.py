# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence


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
