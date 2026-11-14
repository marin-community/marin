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

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog.release_browser import verify_rendered_partition
from infra.marina.apps.rl_data_catalog.release_check import main
from infra.marina.apps.rl_data_catalog.release_evidence import projected_sources, task_totals
from infra.marina.apps.rl_data_catalog.server.source_counts import SourceCounts
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence, filtered_declaration


def test_release_counts_include_eval_pending_and_unique_static_and_final_exclusions() -> None:
    declarations = {source.name: with_declaration_admission_census(source) for _, source in contamination_declarations()}
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


@pytest.mark.parametrize("failure", ["command", "production"])
def test_release_cli_persists_failed_gate_receipt_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
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

    def process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7 if failure == "command" and command[0] != "git" else 0)

    def response(session: Any, method: str, url: str, **kwargs: Any) -> requests.Response:
        result = requests.Response()
        result.status_code = 200
        result._content = b'{"current_version":77}'
        return result

    monkeypatch.setattr(subprocess, "check_output", git_output)
    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr("requests.Session.request", response)
    with pytest.raises(SystemExit) as exit_code:
        main()
    result = json.loads(receipt.read_text())
    assert exit_code.value.code == 1 and result["passed"] is False
    if failure == "command":
        assert result["commands"][-1]["exit_code"] == 7
        assert Path(result["commands"][-1]["log"]).exists()
    else:
        assert result["error"]["type"] == "ValueError"
        assert result["error"]["gate"] == "production-version"
