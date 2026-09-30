import argparse
import json
import sys
import types
from pathlib import Path

import pytest

from capability_pipeline import generate
from capability_pipeline.catalog import build_pilot, ingest_catalog
from capability_pipeline.inference import atomic_json, digest


def test_adoption_requires_all_inputs_before_creating_output(tmp_path, pilot):
    invocation = args(tmp_path, pilot, adopt_proposal_checkpoint=str(tmp_path))
    with pytest.raises(generate.GenerateError, match="requires checkpoint"):
        generate.generate(invocation)
    assert not Path(invocation.out).exists()


def test_adopted_checkpoint_skips_proposer_and_reuses_bound_receipt(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    snapshot = tmp_path / "checkpoint"
    snapshot.mkdir()
    for name in ("snapshot-capture.json", "pull-manifest.json"):
        atomic_json(snapshot / name, {"fixture": name})
    archive, launch = tmp_path / "source.tar", tmp_path / "launch.json"
    archive.write_bytes(b"mocked source archive")
    atomic_json(launch, {"fixture": True})
    calls = []

    def adopt(**kwargs):
        calls.append("adopt")
        root = kwargs["destination"]
        assert kwargs["target_identity"] == json.loads(
            (root / "generate-run.json").read_text()
        )
        proposal = root / "proposal"
        proposal.mkdir()
        proposal_outputs(proposal, [{"proposal_hash": "fixture"}])
        atomic_json(
            root / "proposal-receipt.json",
            {
                "schema_version": generate.SCHEMA,
                "stage": "proposal",
                "identity_sha256": kwargs["target_identity"]["identity_sha256"],
                "exit_code": 2,
                "accepted_count": 1,
                "files": generate._files(proposal, generate._PROPOSAL_FILES),
                "tree_sha256": generate.tree_sha256(proposal),
                "adoption": {
                    "original_source_archive_sha256": generate.sha256(archive),
                    "original_launch_receipt_sha256": generate.sha256(launch),
                },
            },
        )

    monkeypatch.setitem(
        sys.modules,
        "capability_pipeline.proposal_adoption",
        types.SimpleNamespace(validate_and_adopt=adopt),
    )
    monkeypatch.setattr(
        generate.cli,
        "propose",
        lambda _: pytest.fail("adoption must not regenerate proposals"),
    )

    def construct(stage_args):
        calls.append("construct")
        construction_outputs(Path(stage_args.out))
        return 0

    monkeypatch.setattr(generate.synthesis, "synthesize", construct)
    monkeypatch.setattr(generate, "_coverage", lambda *_: {"state": "complete"})
    invocation = args(
        tmp_path,
        pilot,
        adopt_proposal_checkpoint=str(snapshot),
        adoption_source_archive=str(archive),
        adoption_launch_receipt=str(launch),
    )
    assert generate.generate(invocation) == 0
    assert generate.generate(invocation) == 0
    assert calls == ["adopt", "construct"]
    plain_resume = args(tmp_path, pilot)
    assert generate.generate(plain_resume) == 0
    assert calls == ["adopt", "construct"]
    archive.write_bytes(b"different source")
    with pytest.raises(generate.GenerateError, match="identity or settings changed"):
        generate.generate(invocation)


def args(tmp_path, pilot, **changes):
    values = {
        "pilot": str(pilot),
        "out": str(tmp_path / "run"),
        "concurrency": 2,
        "tier": "interactive",
        "proposal_repair_rounds": 1,
        "proposal_hold_seconds": 10,
        "structured_output": "json_schema",
        "omp": "omp",
        "model": "glm-orion/glm-5.3",
        "session_time": 60,
        "max_continuations": 1,
        "validation_timeout": 60,
        "taskcompendium_source": None,
        "runtime_runner": None,
        "daytona_tools": None,
        "research_overlay": None,
        "coverage_manifest": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def proposal_outputs(root: Path, accepted):
    for name, value in {
        "accepted.json": accepted,
        "rejected.json": [],
        "null.json": [],
        "plans.json": {},
        "proposals.json": [],
        "report.json": {"state": "needs_iteration"},
        "run.json": {"stage": "propose"},
    }.items():
        atomic_json(root / name, value)


def construction_outputs(root: Path):
    for name, value in {
        "report.json": {"state": "complete"},
        "run.json": {"stage": "synthesize"},
        "tasks.json": [],
    }.items():
        atomic_json(root / name, value)


@pytest.fixture
def pilot(tmp_path):
    path = tmp_path / "pilot.json"
    atomic_json(path, {"capabilities": [{"capability_id": "cap.one"}]})
    return path


@pytest.fixture
def fake_accepted(monkeypatch):
    monkeypatch.setattr(
        generate.synthesis,
        "load_accepted",
        lambda path: json.loads(Path(path).read_text()),
    )


def test_exit_two_with_accepted_sibling_constructs_and_requires_coverage(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    calls = []

    def propose(stage_args):
        calls.append(("propose", stage_args.pilot, stage_args.limit))
        proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}])
        return 2

    def synthesize(stage_args):
        calls.append(("synthesize", stage_args.accepted, stage_args.limit))
        construction_outputs(Path(stage_args.out))
        return 0

    monkeypatch.setattr(generate.cli, "propose", propose)
    monkeypatch.setattr(generate.synthesis, "synthesize", synthesize)
    monkeypatch.setattr(
        generate,
        "_coverage",
        lambda *_: {"state": "pending", "reason": "independent accounting absent"},
    )

    assert generate.generate(args(tmp_path, pilot)) == 2
    assert [entry[0] for entry in calls] == ["propose", "synthesize"]
    assert calls[0][2] is None and calls[1][2] is None
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["generation_state"] == "complete"
    assert report["state"] == "needs_continuation"
    assert report["training_ready_quality"]["state"] == "unassessed"


def test_completed_receipts_resume_without_replaying_stages(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    counts = {"propose": 0, "synthesize": 0}

    def propose(stage_args):
        counts["propose"] += 1
        proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}])
        return 0

    def synthesize(stage_args):
        counts["synthesize"] += 1
        construction_outputs(Path(stage_args.out))
        return 0

    monkeypatch.setattr(generate.cli, "propose", propose)
    monkeypatch.setattr(generate.synthesis, "synthesize", synthesize)
    monkeypatch.setattr(generate, "_coverage", lambda *_: {"state": "complete"})
    invocation = args(tmp_path, pilot)
    assert generate.generate(invocation) == 0
    assert generate.generate(invocation) == 0
    assert counts == {"propose": 1, "synthesize": 1}


def test_tampered_frozen_proposal_fails_before_construction(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    monkeypatch.setattr(
        generate.cli,
        "propose",
        lambda stage_args: (
            proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}]) or 0
        ),
    )
    monkeypatch.setattr(
        generate.synthesis,
        "synthesize",
        lambda stage_args: construction_outputs(Path(stage_args.out)) or 0,
    )
    monkeypatch.setattr(generate, "_coverage", lambda *_: {"state": "complete"})
    invocation = args(tmp_path, pilot)
    assert generate.generate(invocation) == 0
    atomic_json(
        tmp_path / "run" / "proposal" / "accepted.json", [{"proposal_hash": "b" * 64}]
    )
    with pytest.raises(generate.GenerateError, match="proposal receipt"):
        generate.generate(invocation)


def test_interrupted_construction_reuses_same_directory_without_proposal_regeneration(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    counts = {"propose": 0, "synthesize": 0}

    def propose(stage_args):
        counts["propose"] += 1
        proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}])
        return 0

    def synthesize(stage_args):
        counts["synthesize"] += 1
        construction_outputs(Path(stage_args.out))
        return 2 if counts["synthesize"] == 1 else 0

    monkeypatch.setattr(generate.cli, "propose", propose)
    monkeypatch.setattr(generate.synthesis, "synthesize", synthesize)
    monkeypatch.setattr(
        generate,
        "_coverage",
        lambda *_: {"state": "incomplete" if counts["synthesize"] == 1 else "complete"},
    )
    invocation = args(tmp_path, pilot)
    assert generate.generate(invocation) == 2
    assert not (tmp_path / "run" / "construction-receipt.json").exists()
    assert generate.generate(invocation) == 0
    assert counts == {"propose": 1, "synthesize": 2}


def test_empty_accepted_inventory_skips_toolchain_and_model_construction(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    monkeypatch.setattr(
        generate.cli,
        "propose",
        lambda stage_args: proposal_outputs(Path(stage_args.out), []) or 2,
    )
    monkeypatch.setattr(
        generate.synthesis,
        "synthesize",
        lambda *_: pytest.fail("empty accepted inventory must not construct"),
    )
    monkeypatch.setattr(generate, "_coverage", lambda *_: {"state": "incomplete"})
    assert generate.generate(args(tmp_path, pilot)) == 2
    progress = json.loads((tmp_path / "run" / "construction-progress.json").read_text())
    assert progress["state"] == "skipped_no_accepted_proposals"


def test_all_null_inventory_uses_real_coverage_without_a_synthesis_root(
    tmp_path, fake_accepted, monkeypatch
):
    capability = {
        "id": "d01.one",
        "kind": "capability",
        "parent_id": None,
        "name": "One",
        "outcome": "Solve one.",
        "includes": ["one"],
        "excludes": ["other"],
        "prerequisites": [],
        "sample_tasks": [],
    }
    catalog = {
        "catalog_version": "test",
        "curricula": [
            {
                "routing_facet": "subject_domain",
                "curriculum": {
                    "subject_id": "D01",
                    "subject_name": "One",
                    "version": "test",
                    "sections": [capability],
                },
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    atomic_json(catalog_path, catalog)
    manifest = build_pilot(
        ingest_catalog(catalog_path),
        [("d01.one", "fixture")],
        source_path="catalog.json",
    )
    pilot_path = tmp_path / "pilot.json"
    atomic_json(pilot_path, manifest)

    def propose(stage_args):
        output = Path(stage_args.out)
        frozen_manifest = json.loads(Path(stage_args.pilot).read_text())
        nulls = [
            {"capability_id": "d01.one", "slot": slot, "status": "null"}
            for slot in range(1, 11)
        ]
        for name, value in {
            "input_pilot.json": frozen_manifest,
            "accepted.json": [],
            "rejected.json": [],
            "null.json": nulls,
            "plans.json": {},
            "proposals.json": nulls,
            "run.json": {
                "stage": "propose",
                "pilot_hash": digest(frozen_manifest),
                "capability_ids": ["d01.one"],
            },
            "report.json": {
                "stage": "proposal_review",
                "state": "needs_iteration",
                "capabilities": 1,
                "expected_slots": 10,
                "accepted": 0,
                "rejected_or_needs_repair": 0,
                "null": 10,
                "missing_slots": [],
            },
        }.items():
            atomic_json(output / name, value)
        return 2

    monkeypatch.setattr(generate.cli, "propose", propose)
    monkeypatch.setattr(
        generate.synthesis,
        "synthesize",
        lambda *_: pytest.fail("all-null accounting must not synthesize"),
    )
    assert generate.generate(args(tmp_path, pilot_path)) == 0
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["coverage"]["state"] == "complete"
    assert report["coverage"]["report"]["accounting_complete"]
    assert report["generation_state"] == "complete_no_accepted_proposals"
    assert report["training_ready_quality"]["state"] == "unassessed"


def test_parser_has_no_limit_argument(tmp_path, pilot):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate.add_parser(sub)
    parsed = parser.parse_args(
        ["generate", "--pilot", str(pilot), "--out", str(tmp_path / "out")]
    )
    assert not hasattr(parsed, "limit")
    assert parsed.concurrency == 256
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "--pilot",
                str(pilot),
                "--out",
                str(tmp_path / "out"),
                "--limit",
                "1",
            ]
        )


def test_resume_rejects_changed_effective_repair_budget(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    monkeypatch.setattr(
        generate.cli,
        "propose",
        lambda stage_args: (
            proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}]) or 0
        ),
    )
    monkeypatch.setattr(
        generate.synthesis,
        "synthesize",
        lambda stage_args: construction_outputs(Path(stage_args.out)) or 0,
    )
    monkeypatch.setattr(generate, "_coverage", lambda *_: {"state": "complete"})
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    invocation = args(tmp_path, pilot)
    assert generate.generate(invocation) == 0
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "2")
    with pytest.raises(generate.GenerateError, match="identity"):
        generate.generate(invocation)


def test_accounted_terminal_rejections_freeze_without_replaying_construction(
    tmp_path, pilot, fake_accepted, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        generate.cli,
        "propose",
        lambda stage_args: (
            proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}]) or 0
        ),
    )

    def rejected(stage_args):
        calls.append(stage_args.out)
        construction_outputs(Path(stage_args.out))
        return 2

    monkeypatch.setattr(generate.synthesis, "synthesize", rejected)
    # Coverage's independent tests verify that only typed, exhausted semantic
    # failures qualify. This checks the orchestration boundary's response.
    monkeypatch.setattr(generate, "_coverage", lambda *_: {"state": "complete"})
    invocation = args(tmp_path, pilot)
    assert generate.generate(invocation) == 0
    assert generate.generate(invocation) == 0
    assert len(calls) == 1
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["construction"]["exit_code"] == 2
    assert report["generation_state"] == "complete_with_rejections"
    assert report["training_ready_quality"]["state"] == "unassessed"


@pytest.mark.parametrize("covered", [False, True])
def test_synthesis_exit_zero_is_not_all_accepted(tmp_path, pilot, fake_accepted, monkeypatch, covered):
    """synthesize exits 0 once every item is terminal (a failed item is an outcome).  Only
    report.json state == "complete" freezes the construction receipt as complete."""
    calls = []
    monkeypatch.setattr(
        generate.cli,
        "propose",
        lambda stage_args: (
            proposal_outputs(Path(stage_args.out), [{"proposal_hash": "a" * 64}]) or 0
        ),
    )

    def all_terminal_not_accepted(stage_args):
        calls.append(stage_args.out)
        construction_outputs(Path(stage_args.out))
        atomic_json(Path(stage_args.out) / "report.json",
                    {"state": "needs_continuation", "all_terminal": True, "exit_code": 0})
        return 0

    monkeypatch.setattr(generate.synthesis, "synthesize", all_terminal_not_accepted)
    monkeypatch.setattr(
        generate, "_coverage", lambda *_: {"state": "complete" if covered else "pending"}
    )
    invocation = args(tmp_path, pilot)
    receipt_path = tmp_path / "run" / "construction-receipt.json"
    if covered:
        # Honest terminal accounting still freezes -- as rejections, never as "complete".
        assert generate.generate(invocation) == 0
        assert receipt_path.exists()
        report = json.loads((tmp_path / "run" / "report.json").read_text())
        assert report["generation_state"] == "complete_with_rejections"
        assert generate.generate(invocation) == 0
        assert len(calls) == 1
    else:
        assert generate.generate(invocation) == 2
        assert not receipt_path.exists()
        report = json.loads((tmp_path / "run" / "report.json").read_text())
        assert report["generation_state"] == "needs_continuation"
        assert report["construction"]["synthesis_state"] == "needs_continuation"
        # Not frozen: a resume re-enters construction in the same directory.
        assert generate.generate(invocation) == 2
        assert len(calls) == 2
