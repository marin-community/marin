# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.github import inventory
from iac.github.audit import audit_required_ci_checks
from iac.github.inventory import RequiredStatusCheck

from scripts.ci.dependency_update_policy import GITHUB_ACTIONS_APP_ID, REQUIRED_CHECKS, REQUIRED_CI_RULESET_NAME


def test_required_ci_audit_reports_stale_agentic_lint_check() -> None:
    live = tuple(RequiredStatusCheck(context, GITHUB_ACTIONS_APP_ID) for context in (*REQUIRED_CHECKS, "agentic-lint"))

    findings = audit_required_ci_checks(live)

    assert len(findings) == 1
    assert findings[0].code == "required-ci-drift"
    assert "agentic-lint:15368" in findings[0].detail


def test_ruleset_inventory_reads_required_check_identities(monkeypatch) -> None:
    responses = iter(
        [
            [{"name": "require main CI", "id": 20721903}],
            {
                "rules": [
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "required_status_checks": [
                                {"context": "marin-lint", "integration_id": GITHUB_ACTIONS_APP_ID}
                            ]
                        },
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(inventory, "_gh_json", lambda *_: next(responses))

    assert inventory.github_required_status_checks("marin-community/marin", REQUIRED_CI_RULESET_NAME) == (
        RequiredStatusCheck("marin-lint", GITHUB_ACTIONS_APP_ID),
    )
