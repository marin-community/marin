# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin Belay guardrails: launches only from a clean, pushed commit of this week's branch, never reusing a run
name or passing the node cap; every Track resolves to a real launch or is refused; code changes land only behind
a default-off flag that leaves the default model unchanged; results follow each Track's fixed rule."""

import dataclasses
import datetime
import io
import json
import shlex
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml
from rigging.auth import IapCredentialsUnavailable

from experiments.belay import __main__ as belay_main
from experiments.belay import page
from experiments.belay.__main__ import (
    carried_ideas,
    ladder_rungs_left,
    launch,
    prior_trials_hold,
    reproduce,
    score_value,
    write_spec,
)
from experiments.belay.analysis import Header, series_from_rows, write_report
from experiments.belay.branch import (
    OURS_MARKER,
    SPLIT_MARKER,
    THEIRS_MARKER,
    Rollover,
    abort_rollover,
    carry,
    cut_fresh_week,
    enter_week,
    finish_port,
    git,
    load_rollover,
    require_clean_pushed_head,
    require_week_branch,
)
from experiments.belay.code_change import (
    Probe,
    named_sources,
    new_flags,
    refuse_flag_off_change,
    refuse_outside_allowed,
)
from experiments.belay.config import CONFIG, TRACKS, WEEKLY_BRANCH, Track
from experiments.belay.page import (
    Idea,
    PageState,
    mark_reviewed,
    publish_thread,
    read_thread,
    require_unedited,
    thread_url,
)
from experiments.belay.posts import (
    Comment,
    Row,
    control_label,
    hardware,
    render,
    stage_section,
    table,
    title,
    updated,
)
from experiments.belay.registry_entry import potential_of, with_step_time
from experiments.belay.results import Verdict, after_equal_data, change, overall_verdict, verdict, wall_step_time
from experiments.belay.submit import (
    LADDER_RUNGS,
    HeroMfuResult,
    IrisJob,
    check_launch,
    experiment_id,
    hero_mfu_result,
    hero_mfu_steps,
    iris_jobs,
    launch_band,
    launch_command,
    new_spec_text,
    parse_spec,
    previous_tracks,
    registry_of,
    run_id,
    run_suffix,
    submit_run,
    tracks_of_comment,
    with_launched,
)
from experiments.belay.sync import is_belay_file, run_total_steps
from experiments.grug.fast_track import belay_s7, belay_tracks

MONDAY = datetime.date(2026, 11, 2)
NODE_CAP = 8
SPEC = """experiment: X0001
track: competitive-d512
runs:
  seed0: {seed: 0, variant: v, flags: [--opt-set, muonh_decay_power=0.5]}
  control: {seed: 0, variant: c, control: true}
"""


def commit_file(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "--quiet", "-m", name)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clone whose origin has the base branch and last week's branch: base plus a winning idea's flag commit and a
    losing idea's flag commit."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", str(origin), str(clone)], check=True)
    git(clone, "config", "user.email", "test@example.com")
    git(clone, "config", "user.name", "test")
    git(clone, "checkout", "--quiet", "-b", CONFIG.base_branch)
    commit_file(clone, "recipe.txt", "recipe\n")
    git(clone, "push", "--quiet", "origin", CONFIG.base_branch)
    last_week = CONFIG.week_branch(MONDAY - datetime.timedelta(days=7))
    git(clone, "checkout", "--quiet", "-b", last_week)
    commit_file(clone, "winner.txt", "X0001: winner (default off)\n")
    commit_file(clone, "loser.txt", "X0002: loser (default off)\n")
    git(clone, "push", "--quiet", "origin", last_week)
    return clone


def commit_of(repo: Path, path: str) -> str:
    return git(repo, "log", "-1", "--format=%H", "--", path)


def test_a_fresh_week_is_the_recipe_plus_only_the_carried_commits(repo: Path) -> None:
    winner = commit_of(repo, "winner.txt")
    git(repo, "checkout", "--quiet", CONFIG.base_branch)
    commit_file(repo, "recipe_fix.txt", "fix\n")
    git(repo, "push", "--quiet", "origin", CONFIG.base_branch)
    week = CONFIG.week_branch(MONDAY)

    assert cut_fresh_week(repo, Rollover(week, [winner], {winner: "X0001"}), dry_run=False) is None

    assert git(repo, "rev-parse", "HEAD~1") == git(repo, "rev-parse", f"origin/{CONFIG.base_branch}")
    assert (repo / "winner.txt").exists() and (repo / "recipe_fix.txt").exists()
    assert not (repo / "loser.txt").exists()


def test_start_enters_a_cut_week_and_brings_in_the_recipe(repo: Path) -> None:
    week = CONFIG.week_branch(MONDAY)
    assert not enter_week(repo, week)  # not cut yet: start sends the session to weekly-rollover
    git(repo, "checkout", "--quiet", "-b", week, f"origin/{CONFIG.base_branch}")
    git(repo, "push", "--quiet", "origin", week)
    git(repo, "checkout", "--quiet", CONFIG.base_branch)
    commit_file(repo, "tooling.txt", "new Belay tooling\n")
    git(repo, "push", "--quiet", "origin", CONFIG.base_branch)

    assert enter_week(repo, week)

    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == week
    assert (repo / "tooling.txt").exists()
    assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", f"origin/{week}")  # the merge is pushed


def conflicting_week(repo: Path) -> tuple[str, Rollover]:
    """The recipe changed the winner's file, so carrying the winner conflicts; the cut waits for a port."""
    winner = commit_of(repo, "winner.txt")
    git(repo, "checkout", "--quiet", CONFIG.base_branch)
    commit_file(repo, "winner.txt", "the recipe changed this file too\n")
    git(repo, "push", "--quiet", "origin", CONFIG.base_branch)
    rollover = Rollover(CONFIG.week_branch(MONDAY), [winner], {winner: "X0001"})
    assert cut_fresh_week(repo, rollover, dry_run=False) == winner
    return winner, rollover


def test_a_conflicting_carry_waits_for_a_port_and_pushes_nothing(repo: Path) -> None:
    winner, _ = conflicting_week(repo)
    assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", f"origin/{CONFIG.base_branch}")
    assert not git(repo, "ls-remote", "--heads", "origin", CONFIG.week_branch(MONDAY))
    assert load_rollover(repo).commits == [winner]


def test_a_port_commits_once_resolved_and_records_the_new_commit(repo: Path) -> None:
    winner, rollover = conflicting_week(repo)
    with pytest.raises(SystemExit, match="still conflicted"):
        finish_port(repo, rollover)
    (repo / "winner.txt").write_text(f"{OURS_MARKER}HEAD\nrecipe\n{SPLIT_MARKER}\nwinner\n{THEIRS_MARKER}x\n")
    git(repo, "add", "winner.txt")
    with pytest.raises(SystemExit, match="conflict markers remain"):
        finish_port(repo, rollover)
    (repo / "winner.txt").write_text("the recipe changed this file too\nX0001: winner (default off)\n")
    git(repo, "add", "winner.txt")
    ported = finish_port(repo, rollover)
    assert rollover.ports == {winner: ported}
    assert git(repo, "log", "-1", "--format=%s", ported) == git(repo, "log", "-1", "--format=%s", winner)
    assert carry(repo, rollover) is None
    assert git(repo, "rev-parse", "HEAD~1") == git(repo, "rev-parse", f"origin/{CONFIG.base_branch}")


def test_an_aborted_port_leaves_the_recipe_and_nothing_pending(repo: Path) -> None:
    conflicting_week(repo)
    abort_rollover(repo)
    assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", f"origin/{CONFIG.base_branch}")
    assert not git(repo, "status", "--porcelain", "--untracked-files=no")
    with pytest.raises(SystemExit, match="no weekly rollover"):
        load_rollover(repo)


def test_a_ladder_is_every_competitive_rung_and_the_approvers_track_wins() -> None:
    rungs = (Track.COMPETITIVE_D512, Track.COMPETITIVE_D768, Track.COMPETITIVE_D1024, Track.COMPETITIVE_D1280)
    assert tracks_of_comment("Track: ladder\nTake X0037 up the ladder.") == (rungs, False)
    instructions = "Maintainer ClassicLarry: run it with Track: ladder, settings from t004."
    assert tracks_of_comment("Track: speedrun\nRun X0037 again.", instructions) == (rungs, False)
    assert tracks_of_comment("Run X0037 again.", "Maintainer ClassicLarry: seed 0 only.") == ((Track.SPEEDRUN,), True)


def test_each_rung_has_its_own_experiment_id_on_the_same_entry() -> None:
    ids = [experiment_id("X0037", track, ladder=True) for track in (Track.COMPETITIVE_D512, Track.COMPETITIVE_D1280)]
    assert ids == ["X0037d512", "X0037d1280"]
    assert experiment_id("X0037", Track.SPEEDRUN, ladder=False) == "X0037"
    assert {registry_of(i) for i in [*ids, "X0037"]} == {"X0037"}
    # The app reads a run's experiment up to its first '-'; a rung's runs must keep that whole id.
    assert "belay-X0037d768-seed0".removeprefix("belay-").partition("-")[0] == "X0037d768"


def test_a_ladder_idea_waits_for_its_last_rung() -> None:
    d512 = {"id": "t005", "rung": "competitive-d512", "loss": [3.1]}
    d768 = {"id": "t006", "rung": "competitive-d768"}
    speedrun = {"id": "t004", "rung": "speedrun"}
    assert ladder_rungs_left([speedrun, d512, d768], d512) == [d768]
    assert ladder_rungs_left([speedrun, d512, {**d768, "loss": [2.9]}], d512) == []


def test_a_ladder_is_held_only_by_trials_on_its_own_rungs() -> None:
    rungs = (Track.COMPETITIVE_D512, Track.COMPETITIVE_D768)
    speedrun_win = {"trials": [{"id": "t004", "rung": "speedrun", "delta": [-0.0074]}]}
    assert prior_trials_hold(speedrun_win, None, rungs) is None
    assert prior_trials_hold(speedrun_win, None, (Track.SPEEDRUN,)) is not None
    on_rung = {"trials": [{"id": "t005", "rung": "competitive-d768", "delta": [-0.002]}]}
    assert prior_trials_hold(on_rung, None, rungs) is not None


def test_only_ideas_in_play_are_carried() -> None:
    def entry(xid: str, delta: float, step: str = "0%", **extra) -> dict:
        trial = {"rung": "speedrun", "protocol": "equal_steps", "delta": [delta], "step_time_change": step}
        return {"id": xid, "code_commits": ["c" + xid], "trials": [trial], "events": [], **extra}

    adopted = {"type": "adopted", "checkpoint": CONFIG.recipe_tag}
    entries = [
        entry("X0001", -0.012),  # won at equal time, not adopted: carried
        entry("X0002", +0.020),  # lost: left behind
        entry("X0003", -0.012, events=[adopted]),  # already in the recipe: nothing to carry
        entry("X0004", -0.002),  # noise: left behind
        entry("X0005", -0.012, "+8%"),  # stage 2 pending, and running on the page: carried
        {"id": "X0006", "trials": [], "events": []},  # settings only: nothing to carry
    ]
    running = Idea(
        5, "intake", "a", None, "b", None, "running", "X0005", None, None, None, None, [], None, None, None, None
    )
    state = PageState([running], [], False, 8, [], [])
    assert carried_ideas(entries, state) == ["X0001", "X0005"]


def test_launch_refused_off_this_weeks_branch(repo: Path) -> None:
    with pytest.raises(SystemExit, match="not this week's branch"):
        require_week_branch(repo, MONDAY)


def test_launch_refused_when_week_branch_lacks_the_base(repo: Path) -> None:
    git(repo, "checkout", "--quiet", "-b", CONFIG.week_branch(MONDAY))
    git(repo, "checkout", "--quiet", CONFIG.base_branch)
    commit_file(repo, "recipe_fix.txt", "fix\n")
    git(repo, "push", "--quiet", "origin", CONFIG.base_branch)
    git(repo, "checkout", "--quiet", CONFIG.week_branch(MONDAY))
    with pytest.raises(SystemExit, match="does not contain"):
        require_week_branch(repo, MONDAY)


def test_launch_refused_from_a_dirty_or_unpushed_tree(repo: Path) -> None:
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert require_clean_pushed_head(repo, branch) == git(repo, "rev-parse", "HEAD")

    (repo / "winner.txt").write_text("edited\n")
    with pytest.raises(SystemExit, match="differ from HEAD"):
        require_clean_pushed_head(repo, branch)

    git(repo, "commit", "--quiet", "-am", "unpushed")
    with pytest.raises(SystemExit, match="not pushed"):
        require_clean_pushed_head(repo, branch)


def test_launch_refused_for_a_reused_run_name() -> None:
    jobs = [IrisJob("/u/belay-X0001-seed0-coord", "succeeded", "2026-11-01T00:00")]
    with pytest.raises(SystemExit, match="already ran"):
        check_launch(parse_spec(SPEC), jobs, NODE_CAP)


def test_launch_held_at_the_node_cap() -> None:
    jobs = [IrisJob(f"/u/belay-X9{i:03d}-seed0-coord", "running", "") for i in range(NODE_CAP - 1)]
    with pytest.raises(SystemExit, match="held"):
        check_launch(parse_spec(SPEC), jobs, NODE_CAP)
    check_launch(parse_spec(SPEC), jobs[:-1], NODE_CAP)  # two runs fit once one more node is free


def test_a_node_grant_lets_an_idea_past_the_cap_up_to_its_grant() -> None:
    full = [IrisJob(f"/u/belay-X9{i:03d}-seed0-coord", "running", "") for i in range(NODE_CAP)]
    with pytest.raises(SystemExit, match="soft cap"):
        check_launch(parse_spec(SPEC), full, NODE_CAP)
    check_launch(parse_spec(SPEC), full, NODE_CAP, grant=2)  # its two runs fit its grant
    with pytest.raises(SystemExit, match="node grant 1"):
        check_launch(parse_spec(SPEC), full, NODE_CAP, grant=1)


def test_a_shared_control_is_not_launched_again_and_takes_no_node() -> None:
    text = new_spec_text(
        "X0002", Track.COMPETITIVE_D512, "v", [("--opt-set", "muonh_decay_power=0.5")], "", "belay-X0001-control"
    )
    spec = parse_spec(text)
    control = spec.runs["control"]
    assert run_id(spec, control) == "belay-X0001-control"
    assert run_id(spec, spec.runs["seed0"]) == "belay-X0002-seed0"
    # The owner's control is running and the cap has room for exactly one more node: the variant alone fits.
    jobs = [IrisJob("/u/belay-X0001-control-coord", "running", "")]
    jobs += [IrisJob(f"/u/belay-X9{i:03d}-seed0-coord", "running", "") for i in range(NODE_CAP - 2)]
    check_launch(spec, jobs, NODE_CAP)


def test_score_na_marks_an_entry_with_no_potential_score() -> None:
    assert score_value("na") is None
    assert score_value("7") == 7


def test_comment_track_defaults_to_speedrun_and_unknown_is_refused() -> None:
    assert tracks_of_comment("Try sqrt-softplus routing.") == ((Track.SPEEDRUN,), True)
    assert tracks_of_comment("**Track:** moe-d768\nTry sqrt-softplus routing.") == ((Track.MOE_D768,), False)
    with pytest.raises(SystemExit, match="not one of"):
        tracks_of_comment("Track: competitive-d2048\nTry sqrt-softplus routing.")
    with pytest.raises(SystemExit, match="not one of"):
        parse_spec(SPEC.replace("competitive-d512", "competitive-d2048"))


# hero-mfu's launcher lives on its own branch (config.HERO_MFU_BRANCH), which tests it there.
@pytest.mark.parametrize("track", [t for t in Track if TRACKS[t].code == WEEKLY_BRANCH])
def test_each_track_resolves_to_a_real_launch_or_is_refused(track: Track) -> None:
    spec = parse_spec(SPEC.replace("competitive-d512", track.value))
    launch = TRACKS[track].launch
    if launch is None:
        with pytest.raises(SystemExit, match="no launch recipe"):
            launch_command(spec, spec.runs["seed0"])
        return
    command = launch_command(spec, spec.runs["seed0"])
    args = command[command.index("-m") + 2 :]
    size = args[args.index("--size") + 1]
    # The launcher itself must accept the track's arguments.
    if launch[0] == belay_s7.__name__:
        assert belay_s7.recipe_args(belay_s7.Ladder(args[args.index("--ladder") + 1]), size)
    else:
        assert launch[0] == belay_tracks.__name__
        assert belay_tracks.track_args(belay_tracks.Track(args[args.index("--track") + 1]), size)


def test_code_change_refused_outside_the_allowed_paths() -> None:
    refuse_outside_allowed(["experiments/grug/fast_track/model.py", "tests/grug/test_fast_track_toy.py"])
    with pytest.raises(SystemExit, match="only"):
        refuse_outside_allowed(["experiments/grug/fast_track/model.py", "lib/levanter/src/levanter/main.py"])
    with pytest.raises(SystemExit, match="add a test"):
        refuse_outside_allowed(["experiments/grug/fast_track/model.py"])


def test_code_change_flag_must_be_new_and_default_off() -> None:
    before = {"GrugModelConfig": {"mla": "True"}, "GrugMoeMuonHConfig": {"lr": "0.02"}}
    on = {"GrugModelConfig": {"mla": "True", "toy": "True"}, "GrugMoeMuonHConfig": {"lr": "0.02"}}
    off = {"GrugModelConfig": {"mla": "True", "toy": "False"}, "GrugMoeMuonHConfig": {"lr": "0.02"}}
    changed = {"GrugModelConfig": {"mla": "False", "toy": "False"}, "GrugMoeMuonHConfig": {"lr": "0.02"}}
    assert new_flags(before, off) == ["GrugModelConfig.toy"]
    with pytest.raises(SystemExit, match="must be off"):
        new_flags(before, on)
    with pytest.raises(SystemExit, match="existing fields"):
        new_flags(before, changed)
    with pytest.raises(SystemExit, match="no new config flag"):
        new_flags(before, before)


def test_code_change_refused_when_flag_off_model_differs() -> None:
    before = Probe({}, "a" * 64, 0.0)
    refuse_flag_off_change(before, Probe({}, "a" * 64, 0.0))
    with pytest.raises(SystemExit, match="differ"):
        refuse_flag_off_change(before, Probe({}, "b" * 64, 0.0))


@pytest.mark.parametrize(
    ("track", "variant", "control", "expected"),
    [
        (Track.COMPETITIVE_D512, 2.866, 2.872, Verdict.WIN),
        (Track.COMPETITIVE_D512, 2.8665, 2.8717, Verdict.WIN),  # -0.0052, past the 0.005 band
        (Track.COMPETITIVE_D512, 2.8760, 2.8717, Verdict.NOISE),  # +0.0043
        (Track.COMPETITIVE_D512, 2.8775, 2.8717, Verdict.LOSS),  # +0.0058
        (Track.VANILLA_D768, 1.420, 1.414, Verdict.LOSS),
        (Track.HERO_MFU, 11.40, 11.465, Verdict.WIN),  # 0.57% faster
        (Track.HERO_MFU, 11.44, 11.465, Verdict.NOISE),  # 0.2% faster: rack noise
        (Track.HERO_MFU, 11.53, 11.465, Verdict.LOSS),
    ],
)
def test_verdict_rule(track: Track, variant: float, control: float, expected: Verdict) -> None:
    baseline = TRACKS[track]
    assert verdict(round(change(variant, control, baseline), 4), baseline) is expected


def test_overall_verdict() -> None:
    assert overall_verdict([Verdict.NOISE, Verdict.WIN]) is Verdict.WIN
    assert overall_verdict([Verdict.LOSS, Verdict.LOSS]) is Verdict.LOSS
    assert overall_verdict([Verdict.LOSS, Verdict.NOISE]) is Verdict.NOISE


def test_comment_edited_after_approval_refused() -> None:
    approved = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.UTC)
    require_unedited(approved - datetime.timedelta(minutes=5), approved)
    with pytest.raises(SystemExit, match="edited after approval"):
        require_unedited(approved + datetime.timedelta(minutes=5), approved)


def test_first_spec_on_a_fresh_weekly_branch(repo: Path) -> None:
    """A weekly branch cut fresh from the recipe has no runs/ directory yet; the first spec creates it."""
    branch = CONFIG.week_branch(MONDAY)
    assert cut_fresh_week(repo, Rollover(branch, [], {}), dry_run=False) is None
    git(repo, "push", "--quiet", "-u", "origin", branch)
    spec = write_spec(repo, branch, SPEC, dry_run=False)
    assert (repo / "experiments/belay/runs/X0001.yaml").read_text() == SPEC
    assert spec.track is Track.COMPETITIVE_D512
    assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", f"origin/{branch}")


def test_idea_with_prior_trials_is_held_unless_retest_requested() -> None:
    entry = {"trials": [{"id": "t002", "variant": "0.85->0.98 over 10%", "rung": "d512", "delta": [0.0216, 0.0162]}]}
    held = prior_trials_hold(entry, None, (Track.SPEEDRUN,))
    assert held is not None and "t002" in held and "0.0216" in held
    assert prior_trials_hold(entry, "Maintainer ClassicLarry: retest on current recipe", (Track.SPEEDRUN,)) is None
    assert prior_trials_hold({}, None, (Track.SPEEDRUN,)) is None


WB = "https://wandb.ai/marin-community/marin_moe/runs"
RECIPE = "[Belay Ckpt Oct 20 2026](https://github.com/marin-community/marin/tree/belay-ckpt-2026-10-20)"


@pytest.mark.parametrize("track", [t for t in Track if t is not Track.CUSTOM])
def test_every_table_has_hardware_steps_and_time(track: Track) -> None:
    plan = table([Row("belay-X0001-control", control_label(track, "abc1234567"), 0, hardware(track), 100)])
    result = table([Row("belay-X0001-seed0", "v", 0, hardware(track), 100, 0.5, 0.1, 3.0, -0.01, 1.2375)])
    for text in (plan, result):
        header = text.splitlines()[0]
        assert header == (
            "| Run | Variant | Seed | Hardware | Steps | s/step | Time | Paloma macro loss | Δ vs control "
            "| Paloma macro BPB |"
        )
        assert f"| {hardware(track)} | 100 |" in text
    assert "| 50 s (+10%) |" in result and result.endswith("| 1.2375 |")
    assert plan.splitlines()[2].split(" | ")[1].startswith("control: ")


def test_control_rows_name_and_link_the_control() -> None:
    assert control_label(Track.SPEEDRUN, "abc") == f"control: current recipe, {RECIPE} (speedrun d512)"
    assert (
        control_label(Track.COMPETITIVE_D1024, "abc")
        == f"control: current recipe, {RECIPE} (competitive d1024, its finished run `s7n-d1024`)"
    )
    tracks = "https://github.com/marin-community/marin/blob/abc/experiments/grug/fast_track/belay_tracks.py"
    assert control_label(Track.MOE_D768, "abc") == f"control: [Basic MoE d768]({tracks})"
    assert control_label(Track.VANILLA_D512, "abc") == f"control: [Vanilla Track d512]({tracks})"
    hero = "https://github.com/marin-community/marin/tree/abc"
    assert control_label(Track.HERO_MFU, "abc") == f"control: [hero MFU stack mfl-mp-qall]({hero})"
    assert [hardware(t) for t in (Track.SPEEDRUN, Track.COMPETITIVE_D1280, Track.HERO_MFU)] == [
        "8\u00d7H100",
        "32\u00d7H100",
        "64\u00d7GB200",
    ]


def no_github(args: list[str]) -> str:
    raise AssertionError(f"unexpected GitHub call {args}")


@pytest.fixture
def app_threads(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """The page's thread store, standing in for the Belay API."""
    threads: dict[str, dict] = {}

    def call(path: str, body: dict | None = None) -> dict:
        assert path.startswith("agent/threads/")
        registry_id = path.removeprefix("agent/threads/")
        if body is None:
            if registry_id not in threads:
                raise page.PageRequestError(404, "no thread")
            return threads[registry_id]
        threads[registry_id] = body
        return {"registry_id": registry_id}

    monkeypatch.setattr(page, "call", call)
    return threads


def test_one_thread_per_idea_plan_then_result(app_threads: dict[str, dict]) -> None:
    stage_1 = stage_section(
        1, "equal data, 3,030 steps per arm", [Row("belay-X0001-seed0", "v", 0, "8\u00d7H100", 3030)]
    )
    plan_title = title("X0001", "Toy", Track.SPEEDRUN, "plan")
    plan = render(Comment(plan_title, {1: stage_1}, ["Defaults chosen: none"]))
    assert read_thread("X0001") is None
    assert (
        publish_thread("X0001", plan_title, plan, 3, "plan", dry_run=False) == f"{CONFIG.public_page_url}#status/X0001"
    )
    measured = Row("belay-X0001-seed0", "v", 0, "8\u00d7H100", 3030, 0.2, 0.31, 2.98, -0.0578)
    win = title("X0001", "Toy", Track.SPEEDRUN, "win at equal data")
    result = updated(
        read_thread("X0001"),
        win,
        1,
        stage_section(1, "equal data, 3,030 steps per arm", [measured]),
        ("X0001 t001", "https://example.com/t001.md"),
    )
    publish_thread("X0001", win, result, 3, "result", dry_run=False)
    assert list(app_threads) == ["X0001"]
    thread = app_threads["X0001"]
    assert (thread["title"], thread["kind"], thread["idea_id"]) == (win, "result", 3)
    assert thread["body"].startswith("\U0001f916 **X0001 · Toy · speedrun · win at equal data**")
    assert "\u22120.0578" in thread["body"] and thread["body"].count("**Stage 1.") == 1
    assert thread["body"].endswith("Analysis: [X0001 t001](https://example.com/t001.md)")


def fake_iap_token() -> str:
    return "iap-id-token"


def no_iap_token() -> str:
    raise AssertionError("a local page must not ask for an IAP token")


@dataclass
class Sent:
    """What one HTTP call carried."""

    method: str
    url: str
    headers: dict[str, str]
    body: dict | None


def test_a_marina_page_is_called_with_the_iap_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Sent] = []

    class Response:
        status_code = 200

        def json(self) -> dict:
            return {"ok": True}

    def request(method: str, url: str, *, json: dict | None, headers: dict, timeout: int) -> Response:
        sent.append(Sent(method, url, headers, json))
        return Response()

    applet = "https://applets.marina.oa.dev/a/33801d46-e280-46f9-b994-2c78603cdf15/"
    monkeypatch.setattr(page, "CONFIG", dataclasses.replace(CONFIG, page_url=applet))
    monkeypatch.setattr(page, "marina_token", fake_iap_token)
    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setenv(page.TOKEN_ENV, "a-local-token")  # never sent to Marina
    session = page.SessionCredentials("loom-1101-120000", "s3cret")
    assert page.call("agent/ideas/3/report", {"note": "x"}, session=session) == {"ok": True}
    assert page.call("public/state") == {"ok": True}
    assert [(s.method, s.url) for s in sent] == [
        ("POST", f"{applet}api/agent/ideas/3/report"),
        ("GET", f"{applet}api/public/state"),
    ]
    assert sent[0].headers["Authorization"] == "Bearer iap-id-token"
    assert (sent[0].headers["X-Belay-Session"], sent[0].headers["X-Belay-Session-Secret"]) == (session.id, "s3cret")
    assert all(page.TOKEN_HEADER not in s.headers for s in sent)
    with pytest.raises(ValueError, match="only for a local preview"):
        page.call("agent/state", token="a-local-token")


def test_a_local_page_takes_the_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[urllib.request.Request] = []

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def urlopen(request: urllib.request.Request, timeout: int) -> Response:
        sent.append(request)
        return Response(b'{"ok": true}')

    monkeypatch.setattr(page, "CONFIG", dataclasses.replace(CONFIG, page_url="http://127.0.0.1:8091/belay/"))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(page, "marina_token", no_iap_token)
    monkeypatch.setenv(page.TOKEN_ENV, "local-token")
    monkeypatch.delenv(page.SESSION_ENV, raising=False)
    monkeypatch.setattr(page, "SESSION_FILE", Path("/nonexistent/belay-session.json"))
    page.call("agent/state")
    page.call("public/state")
    headers = [{k.lower(): v for k, v in r.header_items()} for r in sent]
    assert headers[0]["x-belay-token"] == "local-token" and "authorization" not in headers[0]
    assert "x-belay-token" not in headers[1] and "x-belay-session" not in headers[0]


def test_marina_calls_use_the_machine_service_account_before_a_cached_login(monkeypatch) -> None:
    class Accounts:
        def __init__(self, token: str | None) -> None:
            self.token = token

        def get_token(self) -> str:
            if self.token is None:
                raise IapCredentialsUnavailable("no service account")
            return self.token

    monkeypatch.setattr(page, "bearer_token", lambda: "cached-login-token")
    monkeypatch.setattr(page, "_SERVICE_ACCOUNT_TOKENS", Accounts("loom-vm-token"))
    assert page.marina_token() == "loom-vm-token"
    monkeypatch.setattr(page, "_SERVICE_ACCOUNT_TOKENS", Accounts(None))
    assert page.marina_token() == "cached-login-token"


def intake_idea(url: str | None) -> Idea:
    origin = "intake" if url else "proposal"
    return Idea(
        3,
        origin,
        "someone",
        None,
        "b",
        url,
        "approved",
        "X0001",
        None,
        None,
        None,
        None,
        [],
        "2026-11-05T00:00:00+00:00",
        None,
        None,
        None,
    )


def test_a_reviewed_suggestion_gets_one_reaction_and_no_reply() -> None:
    calls: list[list[str]] = []
    idea = intake_idea("https://github.com/marin-community/marin/issues/9514#issuecomment-41")
    mark_reviewed(idea, github_posting=True, gh=calls.append)
    assert calls == [["repos/marin-community/marin/issues/comments/41/reactions", "-X", "POST", "-f", "content=eyes"]]
    mark_reviewed(idea, github_posting=False, gh=calls.append)
    assert len(calls) == 1


def test_status_links_use_the_public_page() -> None:
    assert thread_url("X0001").startswith(CONFIG.public_page_url)


def test_code_change_may_read_only_the_sources_the_approved_text_names() -> None:
    text = (
        "Port #10512 (see https://github.com/marin-community/marin/pull/10512), building on #10498.\n"
        "Paper: [ref](https://arxiv.org/abs/2501.00001). Not a PR: issue/#12, step #3."
    )
    assert named_sources(text) == [
        "#10512",
        "#10498",
        "https://github.com/marin-community/marin/pull/10512",
        "https://arxiv.org/abs/2501.00001",
    ]
    assert named_sources("Try sqrt-softplus routing.") == []


def test_an_internal_proposal_gets_no_intake_reaction() -> None:
    mark_reviewed(intake_idea(None), github_posting=True, gh=no_github)


def test_a_later_stage_keeps_earlier_sections_and_renames_train_time() -> None:
    body = (
        "\U0001f916 **X0221 · CPLM Plus copy head · speedrun · win at equal data · equal-time run pending**\n\n"
        "**Stage 1. Comparison: equal data.**\n\n| Run | Train time |\n|---|---|\n| a | 481 s |\n\n"
        "**Stage 2. Comparison: equal time, 480 s of training per arm.** Pending.\n\n"
        "Defaults chosen: x\nRule: y\nIdea: z\nAnalysis: [X0221 t001](u1)"
    )
    stage_2 = stage_section(2, "equal time, 480 s of training per arm", [])
    new = updated(body, "X0221 · CPLM Plus copy head · speedrun · win at equal time", 2, stage_2, ("X0221 t002", "u2"))
    assert "| Run | Time |" in new and "Pending" not in new
    assert new.index("**Stage 1.") < new.index("**Stage 2.") < new.index("Defaults chosen: x")
    assert new.endswith("Analysis: [X0221 t001](u1) · [X0221 t002](u2)")


def test_analysis_report_from_recorded_history(tmp_path: Path) -> None:
    recorded = json.loads((Path(__file__).parent / "data/x0217_history.json").read_text())
    control = series_from_rows("belay-X0217-control", recorded["control"])
    variant = series_from_rows("belay-X0217-seed0", recorded["variant"])
    header = Header(
        "X0217", "t005", "MuonH momentum warmup", "speedrun", "loss", "6d1ef53450", "ClassicLarry", None, "Rule"
    )
    final = {"belay-X0217-control": (3.0455, None, 1.26), "belay-X0217-seed0": (3.0678, 0.0223, 1.27)}
    report = write_report(tmp_path, header, control, variant, final)
    text = report.read_text()
    assert text.startswith("# X0217 · MuonH momentum warmup · speedrun · loss")
    for section in ("## Final eval", "## Train loss", "## Δ train loss", "## Speed", "## Gradient norm"):
        assert section in text
    assert "| belay-X0217-seed0 | 3.0678 | +0.0223 | 1.2700 |" in text
    assert (tmp_path / "t005_loss.png").stat().st_size > 0 and (tmp_path / "t005_delta.png").stat().st_size > 0


def test_run_names_never_repeat_an_earlier_run() -> None:
    assert run_suffix("X0001", []) == ""
    used = [
        IrisJob("/u/belay-X0001-control-coord", "succeeded", ""),
        IrisJob("/u/belay-X0001-control-b-coord", "failed", ""),
    ]
    suffix = run_suffix("X0001", used)
    assert suffix == "-c"
    spec = parse_spec(new_spec_text("X0001", Track.SPEEDRUN, "v", [("--opt-set", "muonh_decay_power=0.5")], suffix))
    check_launch(spec, used, NODE_CAP)  # no reused name


@pytest.mark.parametrize("track", [t for t in Track if TRACKS[t].code == WEEKLY_BRANCH and TRACKS[t].launch])
def test_launch_print_shows_the_full_command(track: Track, tmp_path: Path, monkeypatch, capsys) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    runs = tmp_path / "experiments/belay/runs"
    runs.mkdir(parents=True)
    (runs / "X0001.yaml").write_text(SPEC.replace("competitive-d512", track.value))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(belay_main, "page_state", lambda: PageState([], [], False, 8, [], []))  # no grant
    launch("X0001", "seed0", print_only=True, dry_run=False)
    printed = shlex.split(capsys.readouterr().out.strip())
    spec = parse_spec(SPEC.replace("competitive-d512", track.value))
    assert printed == launch_command(spec, spec.runs["seed0"])
    assert printed[printed.index("--run-id") + 1] == "belay-X0001-seed0"
    assert printed[printed.index("--target-cluster") + 1] == TRACKS[track].cluster
    assert printed[printed.index("--priority") + 1] == TRACKS[track].band


def test_recorded_launch_is_the_submitted_command() -> None:
    spec = parse_spec(SPEC)
    submitted = {key: launch_command(spec, run) for key, run in spec.runs.items()}
    recorded = yaml.safe_load(with_launched(SPEC, submitted))["launched"]
    assert {key: shlex.split(line) for key, line in recorded.items()} == submitted
    assert parse_spec(with_launched(SPEC, submitted)).runs == spec.runs  # the launcher ignores the record


def test_reproduce_line() -> None:
    line = reproduce("X0221", "9db6be9125abcdef", "seed0-b")
    assert line == (
        "Reproduce: [spec](https://github.com/marin-community/marin/blob/9db6be9125abcdef/experiments/belay/runs/"
        "X0221.yaml) · `git checkout 9db6be9125 && uv run python -m experiments.belay launch X0221 --run seed0-b`"
    )
    assert stage_section(1, "equal data", [], line).endswith(line)


def test_each_run_launches_at_its_own_seed() -> None:
    """Regression: the seed-1 confirm ran at seed 0 because the launch used the default seed."""
    spec = parse_spec(SPEC.replace("seed0: {seed: 0", "seed1-t: {seed: 1"))
    command = launch_command(spec, spec.runs["seed1-t"])
    assert command[command.index("--seed") + 1] == "1"


@pytest.mark.parametrize(
    ("time_change", "stage_1", "runs", "status", "line"),
    [
        (0.004, Verdict.LOSS, False, "loss at equal time (step time within 1%)", "within 1%"),
        (0.31, Verdict.WIN, True, None, "Variant 31% slower and won at equal data → equal-time run."),
        (0.31, Verdict.NOISE, False, "noise at equal data; slower, equal time not run", "did not win"),
        (-0.12, Verdict.LOSS, True, None, "Variant 12% faster → equal-time run"),
    ],
)
def test_after_equal_data_rule(time_change, stage_1, runs, status, line) -> None:
    rule = after_equal_data(time_change, stage_1, TRACKS[Track.SPEEDRUN].step_time_tolerance)
    assert rule.run_equal_time is runs and rule.final_status == status and line in rule.line


def test_potential_is_tied_to_the_results_it_saw() -> None:
    entry = {"status": "done", "trials": [{"id": "t001", "delta": [-0.0148]}], "events": []}
    first = potential_of(entry, 8, "equal-time win", "claude-opus-5-5")
    assert first["score"] == 8 and first["model"] == "claude-opus-5-5"
    assert potential_of(entry, 7, "other", "m")["inputs_digest"] == first["inputs_digest"]
    entry["trials"][0]["delta"] = [0.01]
    assert potential_of(entry, 8, "equal-time win", "m")["inputs_digest"] != first["inputs_digest"]
    text = yaml.safe_dump({"id": "X1", "trials": [{"id": "t001"}]})
    assert yaml.safe_load(with_step_time(text, "t001", 0.31))["trials"][0]["step_time_change"] == "+31%"


def test_a_priority_grant_raises_the_launch_band() -> None:
    assert launch_band(Track.SPEEDRUN, None, "X0001") == "batch"
    assert launch_band(Track.SPEEDRUN, "interactive", "X0001") == "interactive"
    assert launch_band(Track.MOE_D768, "production", "X0001") == "production"
    assert launch_band(Track.HERO_MFU, "production", "X0001") == "production"


def test_a_priority_grant_below_the_track_band_is_refused() -> None:
    with pytest.raises(SystemExit, match="below track hero-mfu's band production") as refused:
        launch_band(Track.HERO_MFU, "interactive", "X0001")
    assert "escalate --registry-id X0001" in str(refused.value)
    with pytest.raises(SystemExit, match="not one of"):
        launch_band(Track.SPEEDRUN, "system", "X0001")


def test_the_launch_command_carries_the_granted_band() -> None:
    spec = parse_spec(SPEC)
    run = next(r for r in spec.runs.values() if not r.control)
    granted = launch_command(spec, run, launch_band(spec.track, "interactive", spec.experiment))
    assert granted[granted.index("--priority") + 1] == "interactive"
    default = launch_command(spec, run)
    assert default[default.index("--priority") + 1] == TRACKS[spec.track].band
    recorded = yaml.safe_load(with_launched(SPEC, {run.name: granted}))["launched"][run.name]
    assert "--priority interactive" in recorded


def test_iris_refusing_the_band_escalates_without_a_retry(tmp_path: Path) -> None:
    refusal = (
        "ConnectError: Budget identity loom cannot submit PRODUCTION jobs (max band: INTERACTIVE). "
        "Resubmit with `--priority interactive`"
    )
    command = [sys.executable, "-c", f"import sys; print({refusal!r}, file=sys.stderr); sys.exit(1)"]
    with pytest.raises(SystemExit, match="priority production not allowed for user loom") as refused:
        submit_run(command, tmp_path, "X0001")
    assert "Grant the band in Iris user budgets, or lower the request" in str(refused.value)
    with pytest.raises(subprocess.CalledProcessError):
        submit_run([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, "X0001")


def test_a_post_shows_who_granted_the_priority() -> None:
    granted = dataclasses.replace(intake_idea(None), priority_grant="interactive", priority_by="ClassicLarry")
    assert granted.priority_line == "Priority: interactive (granted by @ClassicLarry)"
    assert intake_idea(None).priority_line is None


def test_storage_leaves_out_data_belay_runs() -> None:
    assert is_belay_file("runs/belay-X0221-seed0/checkpoints/step-100/metadata.json")
    assert not is_belay_file("runs/dbelay-d1024-open-web-math-p20-3802f0-s2/checkpoints/step-100/metadata.json")


def test_iris_jobs_lists_belays_iris_user_whoever_runs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sessions submit as Loom's IRIS_USER (loom); the watcher may run as someone else and must still see those runs.
    prefixes = []

    def fake_run(command, **kwargs):
        prefixes.append(command[command.index("--prefix") + 1])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("IRIS_USER", "maintainer")
    assert iris_jobs() == []
    assert prefixes == [f"/{CONFIG.iris_user}/belay-", "/maintainer/belay-"]
    prefixes.clear()
    monkeypatch.setenv("IRIS_USER", CONFIG.iris_user)
    iris_jobs()
    assert prefixes == [f"/{CONFIG.iris_user}/belay-"]


def test_run_total_steps_prefers_the_stop_step_over_the_schedule() -> None:
    schedule = {"trainer": {"trainer": {"num_train_steps": 30389}}}
    assert run_total_steps(schedule) == 30389
    assert run_total_steps({**schedule, "stop_after_steps": 12000}) == 12000
    assert run_total_steps({}) is None


def test_an_official_ladder_rung_trains_the_baseline_steps_over_the_hero_step_time_change() -> None:
    slower = HeroMfuResult("t005", +0.05)
    faster = HeroMfuResult("t006", -0.10)
    assert hero_mfu_steps(Track.COMPETITIVE_D768, slower) == round(30389 / 1.05)
    assert hero_mfu_steps(Track.COMPETITIVE_D768, faster) == round(30389 / 0.90)


def test_the_ideas_latest_hero_mfu_result_sets_the_ladder() -> None:
    def trial(tid: str, rung: str, delta: list[float] | None = None) -> dict:
        return {"id": tid, "rung": rung, **({"delta": delta} if delta is not None else {})}

    entry = {
        "trials": [
            trial("t001", "hero-mfu", [0.04]),
            trial("t002", "speedrun", [-0.01]),
            trial("t003", "hero-mfu", [0.02]),
            trial("t004", "hero-mfu"),  # launched, no result yet
        ]
    }
    assert hero_mfu_result(entry) == HeroMfuResult("t003", 0.02)
    assert hero_mfu_result({"trials": [trial("t001", "speedrun", [-0.01])]}) is None
    with pytest.raises(SystemExit, match="no step time result"):
        hero_mfu_result({"trials": [trial("t001", "hero-mfu", [float("nan")])]})


def test_an_official_ladder_spec_sets_the_variants_steps() -> None:
    setting = ("--opt-set", "muonh_decay_power=0.5")
    text = new_spec_text("X0037d768", Track.COMPETITIVE_D768, "v", [setting], num_steps=28942)
    runs = yaml.safe_load(text)["runs"]
    assert runs["seed0"]["flags"] == [*setting, "--num-steps", "28942"]
    assert "flags" not in runs["control"]


def test_hero_step_time_is_wall_clock_over_its_timed_window() -> None:
    # Steps 0-5 compile and warm up; the timed window is steps 6-23.
    timestamps = {step: 100.0 + (60.0 if step >= 6 else 0.0) + 11.5 * step for step in range(30)}
    assert wall_step_time(timestamps) == pytest.approx(11.5)
    with pytest.raises(SystemExit, match="fewer than two"):
        wall_step_time({3: 1.0, 30: 2.0})


def test_a_rerun_that_names_no_track_repeats_its_last_attempts_tracks() -> None:
    def trial(rung: str, date: str) -> dict:
        return {"rung": rung, "date": date}

    ladder = [
        trial("speedrun", "2026-11-01"),
        *(trial(f"competitive-d{n}", "2026-11-05") for n in (512, 768, 1024, 1280)),
    ]
    assert previous_tracks({"trials": ladder}) == tuple(LADDER_RUNGS)
    one_rung = [trial("speedrun", "2026-11-01"), trial("competitive-d768", "2026-11-05")]
    assert previous_tracks({"trials": one_rung}) == (Track.COMPETITIVE_D768,)
    assert previous_tracks({"trials": [trial("d512", "2026-09-26"), trial("speedrun", "2026-11-01")]}) == (
        Track.SPEEDRUN,
    )
    assert previous_tracks({"track": "custom", "trials": []}) == (Track.CUSTOM,)  # a study records no trials
    assert previous_tracks({"trials": [trial("d512", "2026-09-26")]}) is None  # only legacy rungs
