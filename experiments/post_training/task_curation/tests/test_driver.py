# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, run

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.driver import main, source_input_artifacts
from experiments.post_training.task_curation.sources import rl_data_pipelines


def test_staged_input_adopts_exact_pin_and_calendar_version(tmp_path):
    source = rl_data_pipelines()["Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2"].source
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "source.jsonl").write_text('{"instruction":"fixture"}\n')
    declaration = {"source": asdict(source), "path": str(staged), "version": "2026.10.06.13"}
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": declaration}))
    artifact = source_input_artifacts(manifest)["Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2"]
    assert artifact.version == "2026.10.06.13"
    adopted = run(artifact)[0]
    assert adopted.path == str(staged)
    assert (staged / "source.jsonl").read_text() == '{"instruction":"fixture"}\n'
    declaration["source"]["revision"] = "0" * 40
    manifest.write_text(json.dumps({"Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": declaration}))
    with pytest.raises(ValueError, match="does not match the pinned recipe"):
        source_input_artifacts(manifest)


@pytest.mark.parametrize(
    "first_status,unsupported,keep,infra,expected_mode",
    [
        ("gated", 1, 0, 0, "full"),
        ("sampled", 1, 0, 0, "normalize_only"),
        ("sampled", 0, 0, 0, "full"),  # Missing positive witness alone.
        ("sampled", 1, 1, 0, "full"),  # Mixed readiness retains review.
        ("sampled", 1, 0, 1, "full"),
    ],
)
def test_full_driver_retains_sample_provenance_and_adopts_only_admitted_evidence(
    tmp_path, monkeypatch, first_status, unsupported, keep, infra, expected_mode
):
    captured = {}
    if first_status == "sampled":

        def capture_campaign(steps, **kwargs):
            captured.update(steps=steps, **kwargs)

        monkeypatch.setattr("experiments.post_training.task_curation.driver.run_campaign", capture_campaign)
    definitions = rl_data_pipelines()
    selected = {name: definitions[name] for name in ("MarinSkyRL:aime24", "MarinSkyRL:deepscaler")}
    monkeypatch.setattr("experiments.post_training.task_curation.driver.rl_data_pipelines", lambda: selected)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    arguments = [
        "--runtime-manifest",
        str(runtime),
        "--review-transport",
        "direct-chat",
        "--model-revision",
        "fixture-revision",
        "--review-cache",
        str(tmp_path / "cache"),
        "--max-workers",
        "1",
        "--coordinator-memory",
        "16g",
        "--normalized-shards",
        "1",
        "--worker-image",
        "fixture-image",
        "--report-path",
        str(tmp_path / "full.json"),
    ]
    runner = CliRunner()
    planned = runner.invoke(main, arguments)
    assert planned.exit_code == 0, planned.output
    sources = json.loads(planned.output)["sources"]
    identity = hashlib.sha256(
        canonical_json(
            {
                "sources": sorted((s["name"], s["version"], s["fingerprint"]) for s in sources),
                "worker_image": "fixture-image",
            }
        ).encode()
    ).hexdigest()
    outcomes = [
        {
            "name": source["name"],
            "path": str(tmp_path / "artifacts" / source["name"] / source["version"]),
            "status": first_status if index == 0 else "failed",
            "error": None if index == 0 else "fixture review failed",
        }
        for index, source in enumerate(sources)
    ]
    sample_report = tmp_path / "sample.json"
    sample_report.write_text(
        json.dumps(
            {
                "mode": "sample",
                "status": "failed",
                "sample_identity": identity,
                "counts": dict(Counter(s["status"] for s in outcomes)),
                "sources": outcomes,
            }
        )
    )
    if first_status == "sampled":
        source_path = Path(outcomes[0]["path"])
        (source_path / "work/verified").mkdir(parents=True)
        (source_path / "report.json").write_text(
            json.dumps(
                {
                    "mode": "sample",
                    "status": "sampled",
                    "incomplete_reviews": 0,
                    "processed_rows": 100,
                    "verification": {
                        "status": "inconclusive" if unsupported or infra else "skipped",
                        "counts": {"unsupported": unsupported, "infra_error": infra, "skipped": 100},
                    },
                }
            )
        )
        (source_path / "work/verified/manifest.json").write_text(
            json.dumps(
                {
                    "input_rows": 100,
                    "dispositions": {"keep": keep, "defer": 100 - keep},
                }
            )
        )
    result = runner.invoke(
        main,
        [
            *arguments,
            "--mode",
            "full",
            "--sample-report",
            str(sample_report),
            "--base-url",
            "https://fixture.invalid",
            "--run",
        ],
    )
    assert result.exit_code == 0, result.output
    if first_status == "sampled":
        admitted, failed = captured["steps"]
        evidence = [dep for dep in admitted.deps if "/verification-input/" in dep.name]
        assert len(evidence) == 1
        assert evidence[0].adopt_source == outcomes[0]["path"]
        assert evidence[0].adopt_config == {
            "campaign_report": str(sample_report),
            "sample_identity": identity,
            "sample_source": sources[0]["name"],
            "sample_fingerprint": sources[0]["fingerprint"],
            "sample_path": outcomes[0]["path"],
        }
        binding = admitted.build_config(
            StepContext.for_run(str(tmp_path / "full-source"), str(tmp_path / "artifacts"), deps=admitted.deps)
        )
        assert binding.identity["mode"] == expected_mode
        assert binding.verification_report_path == outcomes[0]["path"] + "/verification/report.json"
        assert not any("/verification-input/" in dep.name for dep in failed.deps)
        assert captured["sample_outcomes"][admitted.name].status == "sampled"
        assert captured["sample_outcomes"][failed.name].status == "failed"
        assert captured["sample_identity"] == identity
        return
    full = json.loads((tmp_path / "full.json").read_text())
    assert full["sample_identity"] == identity
    assert full["status"] == "completed"
    assert full["counts"] == {"not_admitted": len(outcomes)}
    full_names = [source["name"] for source in full["sources"]]
    assert full_names != [source["name"] for source in sources]
    assert [full["sample_outcomes"][name] for name in full_names] == outcomes


def test_recorded_review_cli_changes_only_selected_source_plan(tmp_path, monkeypatch):
    definitions = rl_data_pipelines()
    selected = {name: definitions[name] for name in ("MarinSkyRL:aime24", "MarinSkyRL:deepscaler")}
    monkeypatch.setattr("experiments.post_training.task_curation.driver.rl_data_pipelines", lambda *args: selected)
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    bundle = tmp_path / "manual.json"
    bundle.write_text('{"schema_version":"recorded-review-v1"}')
    manifest = tmp_path / "manual-manifest.json"
    manifest.write_text(
        json.dumps(
            {"MarinSkyRL:aime24": {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}}
        )
    )
    options = {
        "--runtime-manifest": str(runtime),
        "--review-transport": "direct-chat",
        "--model-revision": "fixture-revision",
        "--review-cache": str(tmp_path / "cache"),
        "--max-workers": "1",
        "--coordinator-memory": "16g",
        "--normalized-shards": "1",
        "--worker-image": "fixture-image",
        "--report-path": str(tmp_path / "report.json"),
    }
    arguments = [item for pair in options.items() for item in pair]
    runner = CliRunner()
    original = runner.invoke(main, arguments)
    revised = runner.invoke(main, [*arguments, "--recorded-review-bundles", str(manifest)])
    assert original.exit_code == 0, original.output
    assert revised.exit_code == 0, revised.output
    previous, current = json.loads(original.output)["sources"], json.loads(revised.output)["sources"]
    assert len(current) == len(previous) == 2
    assert previous[0]["fingerprint"] != current[0]["fingerprint"]
    assert previous[1] == current[1]
    manifest.write_text(
        json.dumps(
            {"misspelled-source": {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}}
        )
    )
    invalid = runner.invoke(main, [*arguments, "--recorded-review-bundles", str(manifest)])
    assert invalid.exit_code != 0
    assert isinstance(invalid.exception, ValueError)


def test_source_cli_selects_catalog_order_without_changing_source_identity(tmp_path, monkeypatch):
    definitions = rl_data_pipelines()
    names = ("MarinSkyRL:aime24", "MarinSkyRL:apps", "MarinSkyRL:deepscaler")
    catalog = {name: definitions[name] for name in names}
    monkeypatch.setattr("experiments.post_training.task_curation.driver.rl_data_pipelines", lambda: catalog)
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    arguments = [
        "--runtime-manifest",
        str(runtime),
        "--review-transport",
        "direct-chat",
        "--model-revision",
        "fixture-revision",
        "--review-cache",
        str(tmp_path / "cache"),
        "--max-workers",
        "1",
        "--coordinator-memory",
        "16g",
        "--normalized-shards",
        "1",
        "--worker-image",
        "fixture-image",
        "--report-path",
        str(tmp_path / "report.json"),
    ]
    runner = CliRunner()
    full = runner.invoke(main, arguments)
    subset = runner.invoke(main, [*arguments, "--source", names[2], "--source", names[0]])
    assert full.exit_code == 0, full.output
    assert subset.exit_code == 0, subset.output
    full_sources = json.loads(full.output)["sources"]
    assert json.loads(subset.output)["sources"] == [full_sources[0], full_sources[2]]

    unknown = runner.invoke(main, [*arguments, "--source", "unknown", "--run"])
    assert unknown.exit_code == 2
    assert "Unknown source: unknown" in unknown.output
