# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The three arms as data, and the dry run that proves they differ only where their keys say."""

import asyncio
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("taskforge")

from taskforge.stage10_proposal.model import TaskProposal

from experiments.post_training.capability_driven_envs.acceptance_measure import driver
from experiments.post_training.capability_driven_envs.acceptance_measure.dryrun import check_arms, dry_run
from experiments.post_training.capability_driven_envs.acceptance_measure.proposals import (
    ProposalMismatch,
    read_proposal,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import arm_runtime
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import (
    ARMS_DIR,
    changed_keys,
    load_arms,
    load_treatment,
)

FIXTURES = Path(__file__).parent / "data" / "acceptance_measure"


def fixtures() -> list[tuple[str, TaskProposal]]:
    out = []
    for path in sorted(FIXTURES.glob("*.proposal.json")):
        raw = path.read_bytes()
        out.append((path.name.removesuffix(".proposal.json"), read_proposal(raw, hashlib.sha256(raw).hexdigest())))
    return out


def write_arm(directory: Path, name: str, extends: str | None, values: dict) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"name": name, "description": "test", "extends": extends, "set": values}))
    return path


def test_each_arm_extends_the_one_before_by_its_levers_only():
    arms = load_arms()

    assert set(changed_keys(arms["B"], arms["C"])) == {"review.saturated_accept"}
    assert set(changed_keys(arms["A"], arms["B"])) == {
        "author.guidance",
        "author.reference_files",
        "author.tools",
        "author.probe_battery.enforce_at_submit",
        "author.probe_battery.witness_replay",
        "review.feedback",
        "review.class_feedback",
    }
    assert len({t.digest for t in arms.values()}) == 3


def test_an_arm_file_with_an_unknown_key_or_an_incomplete_base_is_refused(tmp_path):
    """Arm files name their text files relative to themselves, so the test arms sit beside a copy of the texts."""
    for directory in ("text", "shadow"):
        shutil.copytree(ARMS_DIR.parent / directory, tmp_path / directory)
    tmp_path = tmp_path / "arms"
    tmp_path.mkdir()
    base = json.loads((ARMS_DIR / "A.json").read_text())["set"]
    with pytest.raises(ValueError, match=r"no such treatment key: author\.temperature"):
        load_treatment(write_arm(tmp_path, "X", None, base | {"author.temperature": 0.2}))
    partial = {k: v for k, v in base.items() if k != "solver.k"}
    with pytest.raises(ValueError, match=r"leaves solver\.k unset"):
        load_treatment(write_arm(tmp_path, "Y", None, partial))


def test_a_proposal_whose_bytes_changed_since_the_sample_is_refused():
    raw = next(FIXTURES.glob("*.proposal.json")).read_bytes()

    with pytest.raises(ProposalMismatch, match="is not the sampled"):
        read_proposal(raw, "0" * 64)


@pytest.fixture(scope="module")
def dry(tmp_path_factory) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("dry")
    return asyncio.run(dry_run(load_arms(), fixtures(), out)), out


def test_the_dry_run_renders_three_sampled_proposals_and_the_arms_differ_only_where_intended(dry):
    summary, out = dry

    assert len(summary["proposals"]) == 3
    assert all(not pair["unintended"] and not pair["required_but_unchanged"] for pair in summary["pairs"].values())
    assert summary["pairs"]["B-C"]["changed_surfaces"] == ["policy:review.saturated_accept", "review.saturated_decision"]
    a_b = set(summary["pairs"]["A-B"]["changed_surfaces"])
    assert {"author.system", "author.tools", "author.workspace", "review.modify_feedback"} <= a_b
    assert not {"author.user", "author.sampling", "adversary.brief.false_accept", "validation.plan"} & a_b
    assert set(summary["tools"]["B"]) - set(summary["tools"]["A"]) == {"embed_file", "run_probe_battery"}
    assert set(summary["workspace"]["B"]) - set(summary["workspace"]["A"]) == {
        "reference/helpers/grade_kit.py",
        "reference/patterns/probe-battery.md",
        "reference/patterns/strict-grading.md",
    }
    decisions = {arm: d["decision"]["type"] for arm, d in summary["saturated_decision"].items()}
    assert decisions == {"A": "Modify", "B": "Modify", "C": "Accept"}
    assert summary["saturated_decision"]["C"]["band_label"] == "saturated"
    assert summary["fa_defect_findings"] == {"A": 1, "B": 2, "C": 2}
    assert (out / "A" / summary["proposals"][0] / "system.md").exists()


def test_the_control_arm_system_turn_is_the_stack_prompt_and_b_appends_the_kit_text(dry):
    _, out = dry
    capability = fixtures()[0][0]
    a = (out / "A" / capability / "system.md").read_text()
    b = (out / "B" / capability / "system.md").read_text()

    assert b.startswith(a)
    addition = b[len(a) :]
    assert "run_probe_battery" in addition and "answer contract" in addition
    assert "run_probe_battery" not in a


def test_check_arms_refuses_a_difference_outside_the_changed_keys_surfaces(dry):
    _, out = dry
    arms = load_arms()
    rendered = {
        name: [json.loads((out / name / c / "surfaces.json").read_text()) for c, _ in fixtures()] for name in arms
    }
    check_arms(arms, rendered)
    leaked = copy.deepcopy(rendered)
    leaked["C"][1]["adversary.brief.false_reject"]["brief"] += " (leaked)"

    with pytest.raises(AssertionError, match="differ outside their keys' surfaces"):
        check_arms(arms, leaked)


def test_a_policy_overlay_reaches_every_arm_and_the_manifest_records_it():
    arms = load_arms()
    overlay = {"solver.k": 4}

    runtime = arm_runtime(arms["B"], (overlay,))

    assert (runtime.solver_k, runtime.review.solver_k, runtime.band.high) == (4, 4, 0.875)
    assert runtime.adversaries[0].spec.brief == runtime.policy.prompt("adversary.false_accept").template
    plain, overlaid = driver.manifest(b"{}", "0" * 40), driver.manifest(b"{}", "0" * 40, (overlay,))
    assert overlaid["policy_overlays"] == [overlay] and plain["policy_overlays"] == []
    assert all(
        plain["arms"][name]["resolved_policy_digest"] != overlaid["arms"][name]["resolved_policy_digest"]
        for name in arms
    )
    with pytest.raises(ValueError, match=r"no entry solver\.kk"):
        arms["A"].policy({"solver.kk": 4})
