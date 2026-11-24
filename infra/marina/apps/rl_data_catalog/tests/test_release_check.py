# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import requests
from finestore.rl.status_taxonomy import taxonomy_tables
from marina.client import MarinaRequestError
from shellbox.requirements import Requirement

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog import release_check
from infra.marina.apps.rl_data_catalog.release_browser import verify_rendered_partition
from infra.marina.apps.rl_data_catalog.release_check import main
from infra.marina.apps.rl_data_catalog.release_evidence import catalog_state, projected_sources, task_totals
from infra.marina.apps.rl_data_catalog.server.source_counts import SourceCounts
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence, filtered_declaration


@pytest.mark.parametrize(
    "status,path,pipeline,allowed",
    [
        (404, "api/assessment?source_id=example", None, True),
        (404, "api/assessment?source_id=example", "required-assessment", False),
        (404, "api/assessment?source_id=unknown", None, False),
        (404, "api/source?source_id=example", None, False),
        (500, "api/assessment?source_id=example", None, False),
    ],
)
def test_browser_preserves_empty_assessment_response_but_requires_expected_data(
    monkeypatch: pytest.MonkeyPatch, status: int, path: str, pipeline: str | None, allowed: bool
) -> None:
    reader = release_check.CatalogReader("http://localhost/", requests.Session())

    def unavailable(path: str) -> None:
        raise MarinaRequestError(status, "No assessment published")

    monkeypatch.setattr(reader, "get", unavailable)
    sources = [{"id": "example", "assessment": {"pipeline_key": pipeline}}]
    if allowed:
        actual_status, body = release_check.browser_api_response(reader, path, sources)
        assert actual_status == 404
        assert "No assessment published" in body["detail"]
    else:
        with pytest.raises(MarinaRequestError):
            release_check.browser_api_response(reader, path, sources)


def test_release_counts_include_eval_pending_and_unique_static_and_final_exclusions() -> None:
    declarations = {
        source.id.rpartition(":")[2]: with_declaration_admission_census(source)
        for _, source in contamination_declarations()
    }
    counts = SourceCounts(
        sources=(
            count_evidence(declarations["clean"], oracle_tasks=0, blocked=1),
            count_evidence(declarations["eval"], oracle_tasks=0, blocked=0),
            count_evidence(declarations["required"], oracle_tasks=1, blocked=None),
        )
    )
    rows = [declarations[name].model_dump(mode="json") for name in ("clean", "eval", "required")]
    rows.append(filtered_declaration(0).model_dump(mode="json"))
    result = projected_sources({"sources": rows}, counts)
    assert asdict(task_totals(result["sources"])) == {
        "population": 12,
        "admitted": 5,
        "train": 2,
        "evaluation": 3,
        "pending": 3,
        "excluded": 4,
        "unmeasured": 0,
    }


def test_release_counts_do_not_rebind_old_observations_to_new_task_bytes() -> None:
    source = with_declaration_admission_census(contamination_declarations()[1][1])
    counts = SourceCounts(sources=(count_evidence(source, oracle_tasks=0, blocked=0),))
    row = source.model_dump(mode="json")
    row["actor_census_input"]["task_specs_sha256"] = "e" * 64
    projected = projected_sources({"sources": [row]}, counts)
    totals = task_totals(projected["sources"])
    assert totals.admitted == 0 and totals.unmeasured == totals.population == 3
    assert projected["sources"][0]["filtration"]["example_count"] is None


@pytest.mark.parametrize("shown,admitted,omitted", [(59, 30, [20, 10, 30, 20]), (60, 29, [20, 10, 30, 20])])
def test_release_checker_rejects_population_or_admission_reconciliation_regressions(
    shown: int, admitted: int, omitted: list[int]
) -> None:
    verify_rendered_partition(110, 60, 60, 30, [20, 10, 30, 20])
    with pytest.raises(ValueError, match="do not equal"):
        verify_rendered_partition(110, 60, shown, admitted, omitted)


@pytest.mark.parametrize("failure", ["command", "production", "workspace"])
def test_release_cli_persists_failed_gate_receipt_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "version": 2,
                "source_revision": "a" * 40,
                "preview_base_version": 1,
                "package": {},
                "evidence_before": {},
                "expected_sources": {},
                "browser": {},
                "api_invariants": {},
                "passed": True,
                "phase": "preflight",
                "production_version": 76,
                "inputs": {
                    "source_counts_json": '{"sources":[]}',
                    "source_counts_sha256": hashlib.sha256(b'{"sources":[]}').hexdigest(),
                },
            }
        )
    )
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_check",
            "--revision",
            "2",
            "--baseline",
            str(baseline),
            "--production-version",
            "76",
            "--service-url",
            "http://127.0.0.1:1234",
            "--receipt",
            str(receipt),
        ],
    )

    def git_output(command: list[str], **kwargs: Any) -> str:
        if command[1] == "status":
            return ""
        if command[1] == "branch":
            return "release-test"
        if command[1] == "ls-files":
            return "infra/marina/apps/rl_data_catalog/release_check.py\n"
        return "a" * 40

    real_run = subprocess.run

    def process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "docker":
            return real_run(command, **kwargs)
        return subprocess.CompletedProcess(command, 7 if failure == "command" and command[0] != "git" else 0)

    def response(session: Any, method: str, url: str, **kwargs: Any) -> requests.Response:
        result = requests.Response()
        result.status_code = 200
        result._content = b'{"current_version":77}'
        return result

    monkeypatch.setattr(subprocess, "check_output", git_output)
    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr("requests.Session.request", response)
    if failure == "workspace":
        monkeypatch.setattr(release_check, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as exit_code:
        main()
    result = json.loads(receipt.read_text())
    assert exit_code.value.code == 1 and result["passed"] is False
    if failure == "command":
        assert result["commands"][-1]["exit_code"] == 7
        assert Path(result["commands"][-1]["log"]).exists()
    elif failure == "production":
        assert result["error"]["exception_type"] == "ValueError"
        assert result["error"]["gate"] == "production-version"
    else:
        assert result["error"]["gate"] == "workspace-environment"


@pytest.mark.parametrize("full,failed", [(False, False), (True, False), (True, True)])
def test_cli_records_nonblocking_audit_separately_from_hosted_success(tmp_path, monkeypatch, full, failed):
    receipt_path = tmp_path / "receipt.json"
    args = [
        "release_check",
        "--revision",
        "20",
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--production-version",
        "76",
        "--receipt",
        str(receipt_path),
    ]
    if full:
        args.append("--full-browser-sweep")
    monkeypatch.setattr(sys, "argv", args)

    def checked(args, artifacts, receipt):
        if failed:
            receipt.gate = "hosted-browser-sweep"
            raise ValueError("full audit finding")

    monkeypatch.setattr(release_check, "run_checks", checked)
    monkeypatch.setattr(
        release_check, "launch_postpublish", lambda *args: {"blocking": False, "launch_error": "unavailable"}
    )
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == (1 if failed else 0)
    recorded = json.loads(receipt_path.read_text())
    assert recorded["passed"] is (not failed)
    assert recorded["blocking"] is (not full)
    assert recorded["phase"] == ("postpublish" if full else "hosted")
    assert recorded["postpublish_job"] == (None if full else {"blocking": False, "launch_error": "unavailable"})


def test_release_cli_fails_closed_with_history_timeout_sql_and_bindings(tmp_path, monkeypatch):
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_check",
            "--revision",
            "2",
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--production-version",
            "76",
            "--receipt",
            str(receipt_path),
        ],
    )

    def query(sql, parameters):
        if sql.startswith("SELECT c.load_id"):
            return {"rows": [{"load_id": "fixture", "snapshot_id": "frozen"}]}
        if sql.startswith("SELECT DISTINCT pipeline_key FROM atlas_packets"):
            return {"rows": [{"pipeline_key": "retained-source"}]}
        if "md5(packet)" in sql and parameters:
            raise MarinaRequestError(504, "query exceeded its statement timeout")
        return {"rows": []}

    def checked(args, artifacts, receipt):
        receipt.gate = "catalog-history-readback"
        catalog_state(query)

    monkeypatch.setattr(release_check, "run_checks", checked)
    with pytest.raises(SystemExit) as exit_code:
        main()
    recorded = json.loads(receipt_path.read_text())
    assert exit_code.value.code == 1 and not recorded["passed"] and recorded["blocking"]
    assert recorded["error"]["gate"] == "catalog-history-readback"
    message = recorded["error"]["message"]
    assert "504" in message and "query exceeded its statement timeout" in message
    assert "pipeline_0" in message and "retained-source" in message and "md5(packet)" in message


class _NotFoundReader:
    """A reader whose served revision predates the checked release's endpoints."""

    def get(self, path: str) -> dict[str, Any]:
        raise MarinaRequestError(404, "Not Found")


def test_browser_api_response_serves_the_candidate_taxonomy_when_the_served_revision_lacks_it():
    status, tables = release_check.browser_api_response(_NotFoundReader(), "api/taxonomy", [])
    assert status == 200
    assert tables == taxonomy_tables(requirement_values=[requirement.value for requirement in Requirement])
    assert set(tables) == {"status", "stage", "reason", "use", "method", "backend", "backend-reason"}


class _Flaky504Reader:
    """A reader whose cold plan exceeds the statement timeout before warming."""

    def __init__(self, failures: int) -> None:
        self.failures = failures

    def get(self, path: str) -> dict[str, Any]:
        if self.failures > 0:
            self.failures -= 1
            raise MarinaRequestError(504, "query exceeded its statement timeout")
        return {"tasks": []}


def test_browser_api_response_retries_transient_read_timeouts(monkeypatch):
    monkeypatch.setattr(release_check.time, "sleep", lambda _seconds: None)
    reader = _Flaky504Reader(failures=2)
    status, data = release_check.browser_api_response(reader, "api/curriculum/tasks?sample_pool=true", [])
    assert status == 200 and data == {"tasks": []}
    exhausted = _Flaky504Reader(failures=3)
    with pytest.raises(MarinaRequestError):
        release_check.browser_api_response(exhausted, "api/curriculum/tasks?sample_pool=true", [])
    assert exhausted.failures == 0
