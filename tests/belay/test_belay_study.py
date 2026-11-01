# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Studies on the custom track: the approved budget bounds the runs, checkpoints stay under the study's prefix,
the analysis script is the only code a study adds, and its thread reads as a study with no verdict."""

from pathlib import Path

import pytest

from experiments.belay.code_change import refuse_outside_study
from experiments.belay.config import TRACKS, Track
from experiments.belay.study import (
    Budget,
    StudyStatus,
    check_study,
    checkpoint_root,
    parse_study,
    refuse_input,
    script_path,
    study_command,
    study_post,
)
from experiments.belay.study_checkpoint import selftest

STUDY_SPEC = """\
experiment: X0300
kind: study
track: custom
question: Do routed experts specialize by token type at d512?
runs:
  base:
    track: speedrun
    seed: 0
    variant: speedrun recipe, checkpoints kept
    hours: 0.5
  wide:
    track: moe-d512
    seed: 0
    variant: Basic MoE d512, checkpoints kept
    flags: [--model-set, num_experts_per_token=8]
    hours: 1
analysis:
  - script: experiments/belay/studies/X0300.py
    input: base
"""
STUDIES = Path(__file__).parent / "studies"


def spec_with(**run: object) -> str:
    """The study spec with one run, ``extra``, added."""
    fields = "\n".join(f"    {key}: {value}" for key, value in run.items())
    return STUDY_SPEC.replace("analysis:", f"  extra:\n{fields}\nanalysis:")


def test_a_study_within_its_budget_passes() -> None:
    check_study(parse_study(STUDY_SPEC), Budget(nodes=2, hours=1))


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (None, "no study budget"),
        (Budget(nodes=1, hours=4), "2 nodes at once"),  # two 1-node runs launched together
        (Budget(nodes=4, hours=0.75), "run wide is 1.0 h"),
        (Budget(nodes=2, hours=0.6), "run wide is 1.0 h"),
    ],
)
def test_a_study_past_its_budget_is_refused(budget: Budget | None, reason: str) -> None:
    with pytest.raises(SystemExit, match=reason) as refused:
        check_study(parse_study(STUDY_SPEC), budget)
    assert "escalate" in str(refused.value)


@pytest.mark.parametrize(
    ("run", "reason"),
    [
        ({"track": "hero-mfu", "seed": 0, "variant": "v", "hours": 1}, "production priority or the hero rack"),
        ({"track": "custom", "seed": 0, "variant": "v", "hours": 1}, "names track custom"),
        ({"track": "speedrun", "seed": 0, "variant": "v", "hours": 1, "flags": "[--output-path, s3://x]"}, "Belay sets"),
        (
            {"track": "speedrun", "seed": 0, "variant": "v", "hours": 1, "flags": "[--model-set, brand_new=1]"},
            "new model",
        ),
    ],
)
def test_runs_a_session_may_not_launch_are_refused(run: dict, reason: str) -> None:
    with pytest.raises(SystemExit, match=reason):
        check_study(parse_study(spec_with(**run)), Budget(nodes=64, hours=100))


def test_study_checkpoints_stay_under_the_study_prefix() -> None:
    study = parse_study(STUDY_SPEC)
    command = study_command(study, study.runs["base"])
    output = command[command.index("--output-path") + 1]
    assert output == "tmp/ttl=30d/belay/studies/X0300/belay-X0300-base"
    assert "--save-checkpoints" in command
    assert command[command.index("--priority") + 1] == TRACKS[Track.SPEEDRUN].band
    root = checkpoint_root(study, study.runs["base"])
    assert root == "s3://marin-us-east-02a/tmp/ttl=30d/belay/studies/X0300/belay-X0300-base/checkpoints"
    refuse_input("X0300", root)
    refuse_input("X0300", f"{root}/step-3030")
    for outside in (
        "s3://marin-us-east-02a/tmp/ttl=30d/belay/runs/belay-X0300-base/checkpoints",
        "s3://marin-us-east-02a/tmp/ttl=30d/belay/studies/X0301/belay-X0301-base/checkpoints",
        "s3://marin-us-east-02a/tmp/ttl=30d/belay/studies/X0300/../X0301/checkpoints",
        "gs://marin-us-central2/tmp/ttl=30d/belay/studies/X0300/run",
        "/tmp/ttl=30d/belay/studies/X0300/run",
    ):
        with pytest.raises(SystemExit, match="is not under"):
            refuse_input("X0300", outside)


def test_a_resume_from_outside_the_prefix_is_refused() -> None:
    elsewhere = "[--resume-from, s3://marin-us-east-02a/checkpoints/hero/step-1]"
    run = {"track": "speedrun", "seed": 0, "variant": "v", "hours": 1, "flags": elsewhere}
    with pytest.raises(SystemExit, match="is not under"):
        check_study(parse_study(spec_with(**run)), Budget(nodes=64, hours=100))


def test_the_analysis_script_is_the_only_code_a_study_adds() -> None:
    script = script_path("X0300")
    assert script == "experiments/belay/studies/X0300.py"
    refuse_outside_study([script], script)
    with pytest.raises(SystemExit, match="no model code"):
        refuse_outside_study([script, "experiments/grug/fast_track/model.py"], script)


def test_a_study_post_reads_as_a_study_with_no_verdict() -> None:
    study = parse_study(STUDY_SPEC)
    url = "https://github.com/marin-community/marin/blob/marin_belay_v1_registry/experiments/belay/analysis/X0300/report.md"
    heading, body = study_post(study, "Expert token types", StudyStatus.REPORT, Budget(2, 1), "abc1234", url)
    assert heading == "X0300 · Expert token types · custom · report"
    lines = body.splitlines()
    assert lines[0] == "\U0001f916 **X0300 · Expert token types · custom · report**"
    assert "**Question.** Do routed experts specialize by token type at d512?" in lines
    assert "| Run | Track | Hardware | Nodes | Est. hours | Checkpoints |" in lines
    assert any("belay-X0300-wide" in line and "| moe-d512 |" in line for line in lines)
    assert any(line.startswith("Budget: 2 nodes, 1 h; planned 1.5 of 2 node-hours.") for line in lines)
    assert any(line.startswith("Analysis: `experiments/belay/studies/X0300.py` on belay-X0300-base") for line in lines)
    assert lines[-1] == f"Report: [X0300 report]({url}). A study has no verdict."
    plan_heading, plan_body = study_post(study, "Expert token types", StudyStatus.PLAN, Budget(2, 1), "abc1234")
    assert plan_heading.endswith("· study plan") and "Report:" not in plan_body


def test_an_analysis_script_runs_on_a_tiny_synthetic_checkpoint(tmp_path: Path) -> None:
    assert selftest(STUDIES / "norms_study.py") == [Path("report.md"), Path("rms.png")]
    silent = tmp_path / "silent.py"
    silent.write_text("def analyze(params, out_dir):\n    assert params\n")
    with pytest.raises(SystemExit, match=r"wrote no report\.md"):
        selftest(silent)
