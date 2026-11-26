# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests
from finestore.rl.status_taxonomy import taxonomy_tables
from marina.client import MarinaRequestError
from shellbox.requirements import Requirement

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog import release_browser, release_check
from infra.marina.apps.rl_data_catalog.release_browser import verify_rendered_partition
from infra.marina.apps.rl_data_catalog.release_check import main
from infra.marina.apps.rl_data_catalog.release_evidence import catalog_state, projected_sources, task_totals
from infra.marina.apps.rl_data_catalog.server.json_digest import json_sha256
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


@pytest.mark.parametrize("failure", ["command", "production", "workspace", "covered"])
def test_release_cli_persists_failed_gate_receipt_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "version": 2,
                # A hosted check at the preflight's own checkout reuses its code gates.
                "source_revision": "a" * 40 if failure == "covered" else "b" * 40,
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
    elif failure == "covered":
        assert result["commands"] == [] and result["code_gates_from"] == "a" * 40
        assert result["error"]["gate"] == "production-version"
    else:
        assert result["error"]["gate"] == "workspace-environment"


@pytest.mark.parametrize("postpublish,failed", [(False, False), (True, False), (True, True)])
def test_cli_records_the_hosted_and_postpublish_phases_without_a_detached_audit(
    tmp_path, monkeypatch, postpublish, failed
):
    """R40 #60: the hosted check no longer detaches a second full sweep beside the
    graph's postpublish phase; postpublish is its own blocking phase."""
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
    if postpublish:
        args += ["--postpublish", "--hosted-receipt", str(tmp_path / "hosted.json")]
    monkeypatch.setattr(sys, "argv", args)

    def checked(args, artifacts, receipt):
        if failed:
            receipt.gate = "hosted-browser-sweep"
            raise ValueError("postpublish finding")

    monkeypatch.setattr(release_check, "run_checks", checked)
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == (1 if failed else 0)
    recorded = json.loads(receipt_path.read_text())
    assert recorded["passed"] is (not failed)
    assert recorded["blocking"] is True
    assert recorded["phase"] == ("postpublish" if postpublish else "hosted")


def _hosted_receipt(tmp_path: Path, **overrides: Any) -> Path:
    document = {
        "phase": "hosted",
        "passed": True,
        "revision": 41,
        "browser": {"scope": "full", "api_response_sha256": {"api/sources": "a" * 64}},
        **overrides,
    }
    path = tmp_path / "hosted.json"
    path.write_text(json.dumps(document))
    return path


def test_postpublish_compares_against_the_revisions_full_hosted_sweep(tmp_path):
    assert release_check.hosted_sweep_baseline(_hosted_receipt(tmp_path), 41)["scope"] == "full"
    with pytest.raises(ValueError, match="not a passing hosted receipt"):
        release_check.hosted_sweep_baseline(_hosted_receipt(tmp_path, revision=40), 41)
    with pytest.raises(ValueError, match="no full browser sweep"):
        release_check.hosted_sweep_baseline(
            _hosted_receipt(tmp_path, browser={"scope": "representative", "api_response_sha256": {"k": "v"}}), 41
        )


class _ServedReader:
    def __init__(self, documents: dict[str, Any]) -> None:
        self.documents = documents
        self.cache: dict[str, Any] = {"api/sources": "stale cache entry"}

    def get(self, path: str) -> Any:
        return self.documents[path]


def test_postpublish_diff_rereads_every_recorded_response_and_maps_changes_to_families():
    served = {
        "api/sources": {"sources": []},
        "api/curriculum?release=r": {"hierarchy": [1]},
        "api/source?source_id=x": {},
    }
    recorded = {key: json_sha256(value) for key, value in served.items()}
    reader = _ServedReader({**served, "api/curriculum?release=r": {"hierarchy": [2]}})
    assert release_check.changed_api_keys(reader, recorded) == ["api/curriculum?release=r"]
    assert reader.cache == {}
    families = {release_check.sweep_family(key) for key in served}
    assert families == {
        release_check.SweepFamily.SOURCES,
        release_check.SweepFamily.CURRICULUM,
        release_check.SweepFamily.EVIDENCE,
    }
    assert release_check.sweep_family("measurements/oracle-inventory.json") is release_check.SweepFamily.READABILITY


class _HierarchyReader:
    def get(self, path: str) -> Any:
        assert path == "api/curriculum?release=current"
        return {"hierarchy": [{"kind": "subject", "subject_id": f"D{index:02d}"} for index in range(1, 21)]}


def test_sweep_reads_the_pending_manifest_without_a_browser(tmp_path, monkeypatch):
    """R40 #62: the pending-manifest prefetch ran on a never-navigated page, so its package
    fetch was cross-origin and every sweep died on CORS ('Failed to fetch'). The manifest
    ships with the served package: the reader reads it, and no browser starts for it."""

    class _ManifestReader:
        def __init__(self) -> None:
            self.paths: list[str] = []
            self.cache: dict[str, Any] = {}

        def get(self, path: str) -> Any:
            self.paths.append(path)
            if path == "api/curriculum/releases":
                return {"releases": [{"release_id": "current", "selected": True}]}
            if path == "measurements/oracle-inventory.json":
                return {}
            if path == "measurements/pending-filtered-releases.json":
                return {"sources": [{"source_id": "Marin:pending-a"}, {"source_id": "Marin:pending-b"}]}
            raise AssertionError(f"unexpected read: {path}")

    def _no_browser(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("browser_checks launched a browser for the pending manifest")

    monkeypatch.setattr(release_check, "sync_playwright", _no_browser)
    monkeypatch.setattr(release_check, "sweep_units", lambda *args, **kwargs: [])
    monkeypatch.setattr(release_check, "run_sweep_units", lambda *args, **kwargs: [])
    reader = _ManifestReader()
    release_check.browser_checks(reader, "https://x/", {"sources": []}, tmp_path, local=True, stored_groups=[])
    assert reader.paths == [
        "api/curriculum/releases",
        "measurements/oracle-inventory.json",
        "measurements/pending-filtered-releases.json",
    ]
    assert release_check.pending_release_sources(reader) == {"Marin:pending-a", "Marin:pending-b"}


def test_representative_evidence_smoke_takes_precomputed_cases(monkeypatch):
    """R40 #62: the smoke unit's default path fetched the pending manifest on a page that
    never navigated, so the parallel pool's four smoke units died on CORS. The unit hands
    the sweep its one precomputed case instead, and no manifest is read."""
    captured: dict[str, Any] = {}

    def _sweep(page: Any, root: str, sources: list[dict[str, Any]], *, layouts: Any, cases: Any = None) -> list:
        captured["cases"] = cases
        return [{"source_id": sources[0]["id"]}]

    monkeypatch.setattr(release_browser, "evidence_sweep", _sweep)
    sources = [{"id": "Marin:a", "assessment": {"pipeline_key": "k"}}]
    release_browser.evidence_smoke(None, "https://x/", sources, cases=[{"id": "Marin:a"}])
    assert captured["cases"] == sources


def test_full_sweep_splits_into_bounded_parallel_units(tmp_path, monkeypatch):
    """R40 #60: the full sweep was one serial page walk (40+ min); it is split per layout
    and per domain/case chunk so a bounded page pool runs it."""
    sources = [{"id": f"Marin:s{index}"} for index in range(3)]
    releases = [{"release_id": "current", "selected": True}]
    cases = [{"id": f"Marin:s{index}"} for index in range(10)]
    monkeypatch.setattr(release_check, "evidence_cases", lambda rows, pending: cases)
    full = set(release_check.SweepFamily)
    units = release_check.sweep_units(
        _HierarchyReader(), {"sources": sources}, releases, tmp_path, full, set(), root="https://x/"
    )
    representative = release_check.sweep_units(
        _HierarchyReader(), {"sources": sources}, releases, tmp_path, set(), set(), root="https://x/"
    )
    by_family = Counter(unit.family for unit in units)
    layouts = len(release_check.LAYOUTS)
    # 20 domains in chunks of 8 (3 chunks) plus the release walk; 10 cases in chunks of 8 (2 chunks).
    assert by_family[release_check.SweepFamily.CURRICULUM] == layouts * 4
    assert by_family[release_check.SweepFamily.EVIDENCE] == layouts * 2
    assert by_family[release_check.SweepFamily.SOURCES] == layouts
    assert by_family[release_check.SweepFamily.READABILITY] == layouts
    assert len({unit.label for unit in units}) == len(units)
    assert len(representative) == layouts * 4


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


def test_landed_cut_takes_the_attested_revision_without_git(monkeypatch):
    """A release bundle is not a checkout: the attested full sha replaces the git checks."""
    monkeypatch.setattr(release_check, "git", lambda *args: pytest.fail("bundle mode must not call git"))
    assert release_check.landed_cut("a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="not a full commit sha"):
        release_check.landed_cut("HEAD")


class _TimedCurriculumReader:
    """A candidate reader whose sample-pool pages take a configured time on a fake clock."""

    def __init__(self, seconds: dict[str, float | None], clock: list[float]) -> None:
        self.seconds = seconds
        self.clock = clock
        self.cache: dict[str, Any] = {}
        self.requested: list[str] = []

    cache_key = staticmethod(release_check.CatalogReader.cache_key)

    def get(self, path: str) -> Any:
        self.requested.append(path)
        if path.startswith("api/curriculum?"):
            return {"hierarchy": [{"kind": "subject", "subject_id": subject} for subject in self.seconds]}
        subject = dict(parse_qsl(urlsplit(path).query))["subject_id"]
        seconds = self.seconds[subject]
        if seconds is None:
            raise MarinaRequestError(504, "query exceeded its statement timeout")
        self.clock[0] += seconds
        return {"tasks": []}


def test_curriculum_latency_refuses_a_slow_subject_page():
    """Regression (R40 #54): whole-subject sample pools exceeded the gateway timeout on the
    real catalog, found only after the publish by the postpublish sweep."""
    releases = [{"release_id": "old", "selected": False}, {"release_id": "current", "selected": True}]
    clock = [0.0]
    fast = _TimedCurriculumReader({"D01": 3.0, "D02": 4.0}, clock)
    measured = release_check.curriculum_latency(fast, releases, clock=lambda: clock[0])
    assert measured["release_id"] == "current" and measured["subjects"] == {"D01": 3.0, "D02": 4.0}
    assert all("release=current" in path for path in fast.requested)
    slow = _TimedCurriculumReader({"D01": 3.0, "D02": 61.0, "D03": None}, clock)
    with pytest.raises(ValueError, match=re.escape("'D02': 61.0, 'D03': None")):
        release_check.curriculum_latency(slow, releases, clock=lambda: clock[0])
