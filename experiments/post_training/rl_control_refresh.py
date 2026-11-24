# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan and qualify a new control cohort at a reviewed source revision."""

import argparse
import ast
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from marin.execution.lazy import run
from marin.rl.skyrl_records import RecordedLaunch, RecordedSource, read_stored_skyrl_attempt
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.provenance import Provenance

from experiments.post_training import rl, rl_compose, rl_gate, rl_publication
from experiments.post_training.rl_measurements import RunInputs, read_recorded_launch, read_task_set, recorded_sources
from experiments.post_training.rl_presets import ControlPin, ModelPresets, Preset
from experiments.post_training.snowball_rl import SNOWBALL


@dataclass(frozen=True)
class ControlRefresh:
    preset: Preset
    previous: ControlPin
    current: ControlPin


def plan_refresh(table: ModelPresets) -> list[ControlRefresh]:
    """Select control cohorts with stale revisions or recorded task sets."""
    provenance = Provenance.capture(rl_compose.REPO_ROOT)
    if provenance.dirty or not provenance.base_commit:
        raise ValueError("control refresh needs a clean source commit")
    digests: dict[tuple[tuple[str, str], ...], str] = {}

    def digest(sources: Sequence[RecordedSource]) -> str:
        paths = tuple(sorted((source.uri, source.relative_path) for source in sources))
        if paths not in digests:
            digests[paths] = read_task_set(sources).task_set_sha256
        return digests[paths]

    def signature(launch: RecordedLaunch, preset: Preset) -> tuple[str, ...]:
        return (
            digest(launch.train_data),
            *(digest(recorded_sources(launch, evaluation.name)) for evaluation in preset.evaluations),
        )

    planned = []
    for previous in table.control_pins.values():
        preset = table.by_name[previous.preset]
        if preset.recipe.layout.gpu_variant != "H100":
            raise ValueError(f"control refresh requires H100 placement for {preset.name}")
        stale = previous.revision != preset.revision
        if not stale:
            (config,) = rl_gate._compose_plans(preset, (RunInputs(preset_name=preset.name),))
            expected = RecordedLaunch.from_document(config, attempt=config["run"]["attempt_id"])
            current_tasks = signature(expected, preset)
            try:
                for index in range(previous.replicates):
                    output = rl_gate.control_path(preset, previous, index, table=table)
                    receipt = read_stored_skyrl_attempt(output)
                    if receipt is None or receipt.native.result.state != "succeeded":
                        stale = True
                        break
                    if signature(read_recorded_launch(receipt), preset) != current_tasks:
                        stale = True
                        break
            except FileNotFoundError:
                stale = True
        if stale:
            current = ControlPin(preset.name, provenance.base_commit, preset.revision, previous.replicates)
            planned.append(ControlRefresh(preset, previous, current))
    return planned


def qualify_refresh(
    planned: Sequence[ControlRefresh], table: ModelPresets, *, entry_module: str
) -> tuple[ModelPresets, tuple[tuple[ControlRefresh, str], ...]]:
    """Keep each native run and qualify every replicate before proposing pins."""
    updated = replace(table, control_pins={**table.control_pins, **{item.preset.name: item.current for item in planned}})
    reports = []
    for item in planned:
        preset = item.preset
        code = rl.main(
            table,
            argv=(
                "--preset",
                preset.name,
                "--version",
                preset.revision,
                "--control",
                "--replicates",
                str(item.current.replicates),
                "--run",
                "--wait",
            ),
            entry_module=entry_module,
        )
        if code:
            raise ValueError(f"control collection returned {code} for {preset.name}")
        for replicate in range(item.current.replicates):
            code = rl.main(updated, argv=("compare", "--preset", preset.name, "--leave-out", f"r{replicate}"))
            if code:
                raise ValueError(f"leave-one-out returned {code} for {preset.name} r{replicate}")
        link = rl_gate.control_link_step(preset, item.current, item.previous, table=updated)
        run(link)
        reports.append((item, link.path(marin_prefix())))
    return updated, tuple(reports)


def publish_refresh(reports: Sequence[tuple[ControlRefresh, str]], table: ModelPresets) -> list[str]:
    errors = []
    for item, report in reports:
        if item.preset.publishes:
            outputs = [
                rl_gate.control_path(item.preset, item.current, index, table=table)
                for index in range(item.current.replicates)
            ]
            result = rl_publication.publish_runs(outputs, item.preset, table=table, gate_path=report)
            errors.extend(result.errors)
    return errors


def _write_pins(path: Path, table: ModelPresets) -> None:
    source = path.read_bytes()
    model = next(
        row.value
        for row in ast.parse(source).body
        if isinstance(row, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SNOWBALL" for target in row.targets)
    )
    if not isinstance(model, ast.Call):
        raise ValueError("expected the checked-in Snowball preset declaration")
    binding = next(argument.value for argument in model.keywords if argument.arg == "control_pins")
    lines = source.splitlines(keepends=True)
    assert binding.end_lineno is not None and binding.end_col_offset is not None
    begin = sum(map(len, lines[: binding.lineno - 1])) + binding.col_offset
    end = sum(map(len, lines[: binding.end_lineno - 1])) + binding.end_col_offset
    entries = "\n".join(f"        {name!r}: {pin!r}," for name, pin in table.control_pins.items())
    path.write_bytes(source[:begin] + f"{{\n{entries}\n    }}".encode() + source[end:])


def _submit_pin_pr(
    table: ModelPresets,
    reports: Sequence[tuple[ControlRefresh, str]],
    source: str,
    branch: str,
    pull: int,
    published: str | None,
) -> None:
    title = "Refresh Snowball RL controls"
    body = [f"Pin the refreshed Snowball control cohorts collected at `{source}` after #{pull}."]
    for item, report_path in reports:
        report = rl_gate.read_gate_report(report_path)
        if report is None:
            raise ValueError(f"missing stored control link {report_path}")
        result = f"{report.verdict} ({report.effect})" if report.effect else f"{report.verdict}: {report.reason}"
        body.append(f"- `{item.preset.name}` at `{item.current.revision}` has a {result} link at `{report_path}`.")
    path = Path("experiments/post_training/snowball_rl.py")
    _write_pins(path, table)
    subprocess.run(["uvx", "black@25.9.0", str(path)], check=True)
    if published is None:
        subprocess.run(["git", "switch", "-c", branch], check=True)
        subprocess.run(["git", "add", "--", str(path)], check=True)
        subprocess.run(["git", "commit", "-m", title], check=True)
        subprocess.run(["gh", "auth", "setup-git"], check=True)
        subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], check=True)
    else:
        subprocess.run(["git", "fetch", "origin", published], check=True)
        saved = subprocess.check_output(["git", "show", f"{published}:{path}"])
        if saved != path.read_bytes():
            raise ValueError(f"control pin branch {branch} has different content")
    with TemporaryDirectory(prefix="rl-control-pin-pr-") as directory:
        body_path = Path(directory) / "body.md"
        body_path.write_text(body[0] + "\n\n" + "\n".join(body[1:]) + "\n")
        subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--draft",
                "--label",
                "agent-generated",
                "--title",
                title,
                "--body-file",
                str(body_path),
            ],
            check=True,
        )


def _configured_bot() -> str:
    bot = os.environ.get("RL_CONTROLS_BOT_LOGIN")
    if not bot or not os.environ.get("RL_CONTROLS_ALLOWED_LOGINS") or not os.environ.get("WANDB_API_KEY"):
        raise ValueError("control refresh requires configured bot, requester and W&B credentials")
    login = subprocess.check_output(["gh", "api", "user", "--jq", ".login"], text=True).strip()
    if login != bot:
        raise ValueError("control refresh token does not match the configured bot")
    return bot


def _refresh_controls(planned: Sequence[ControlRefresh], *, pull: int, bot: str) -> None:
    if not planned:
        subprocess.run(
            [
                "gh",
                "pr",
                "comment",
                str(pull),
                "--body",
                "🤖 All RL controls match this merge. No refresh is needed.",
            ],
            check=True,
        )
        return
    if not os.environ.get("TOGETHER_API_KEY"):
        raise ValueError("control collection requires Together grader credentials")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = f"rl-controls/{source}"
    existing = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
        check=False,
    )
    published = None
    if existing.returncode == 0:
        published = existing.stdout.split()[0]
        count = subprocess.check_output(
            ["gh", "pr", "list", "--head", branch, "--state", "all", "--json", "number", "--jq", "length"], text=True
        )
        if int(count):
            print(f"Control pin branch {branch} already has a pull request.")
            return
    elif existing.returncode != 2:
        existing.check_returncode()
    subprocess.run(["git", "config", "user.name", bot], check=True)
    subprocess.run(["git", "config", "user.email", f"{bot}@users.noreply.github.com"], check=True)
    table, reports = qualify_refresh(planned, SNOWBALL, entry_module="experiments.post_training.snowball_rl")
    for error in publish_refresh(reports, table):
        print(f"Control link publication: {error}", file=sys.stderr)
    _submit_pin_pr(table, reports, source, branch, pull, published)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.pull < 1:
        parser.error("--pull must name a positive PR number")
    bot = None if args.dry_run else _configured_bot()
    configure_coreweave_s3()
    planned = plan_refresh(SNOWBALL)
    if bot is None:
        print(json.dumps({"refresh": [item.preset.name for item in planned]}))
        return
    _refresh_controls(planned, pull=args.pull, bot=bot)


if __name__ == "__main__":
    main()
