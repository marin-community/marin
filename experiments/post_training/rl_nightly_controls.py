# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run nightly control cohorts through each model's regular launcher."""

import argparse
import json
import os
import subprocess
import sys
from html import escape
from pathlib import Path

from finestore.rl.rl_progress import completed_controls
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training import rl_compose, rl_gate, rl_index
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_presets import Preset
from experiments.post_training.snowball_rl import SNOWBALL


def select_presets(selection: str) -> list[tuple[str, Preset]]:
    gated = {
        f"{table.model}/{preset.name}": (table.model, preset)
        for table in (SNOWBALL, MOCHI)
        for preset in table.presets
        if preset.gated
    }
    selected = set()
    for token in selection.split():
        matches = {key for key, (model, _) in gated.items() if token == key or token == model}
        if not matches:
            raise ValueError(f"Unknown or ungated RL preset: {token!r}")
        selected.update(matches)
    return [value for key, value in gated.items() if not selected or key in selected]


def dispatch_comment(event_path: Path) -> None:
    event = json.loads(event_path.read_text())
    command = event["comment"]["body"].split(maxsplit=1)
    if not command or command[0] != "/rl-controls" or "pull_request" not in event["issue"]:
        return
    repository = f"repos/{os.environ['GITHUB_REPOSITORY']}"
    number = event["issue"]["number"]
    login = event["comment"]["user"]["login"]
    # The 2026-03-10 dispatch API returns the run ID and links in its response.
    api = ["gh", "api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10"]
    try:
        pull = json.loads(subprocess.check_output([*api, f"{repository}/pulls/{number}"]))
        if pull["merged"] is not True:
            raise ValueError("RL controls require a merged pull request")
        permission = json.loads(subprocess.check_output([*api, f"{repository}/collaborators/{login}/permission"]))
        if permission["permission"] not in {"write", "admin"}:
            raise ValueError("RL controls require repository write permission")
        if len(command) != 2:
            raise ValueError("Provide at least one model or model/preset after /rl-controls")
        selection = command[1]
        select_presets(selection)
    except json.JSONDecodeError:
        raise
    except ValueError as error:
        reply = f"🤖 <code>{escape(str(error))}</code>"
    else:
        response = subprocess.check_output(
            [*api, "-X", "POST", f"{repository}/actions/workflows/marin-skyrl-nightly.yaml/dispatches", "--input", "-"],
            input=json.dumps({"ref": "main", "inputs": {"controls": "true", "presets": selection}}).encode(),
        )
        if not response:
            raise RuntimeError("GitHub dispatch returned no run details")
        reply = f"🤖 [RL control run]({json.loads(response)['html_url']})"
    subprocess.check_output([*api, f"{repository}/issues/{number}/comments", "-f", f"body={reply}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch", help="Unique schedule or dispatch identifier.")
    source.add_argument("--comment-event", type=Path, help="GitHub issue_comment event payload.")
    source.add_argument("--check-presets", help="Validate a selector list before launch setup.")
    parser.add_argument("--presets", default="", help="Space-separated models or model/preset selectors.")
    args = parser.parse_args()
    try:
        if args.comment_event is not None:
            dispatch_comment(args.comment_event)
            return 0
        if args.check_presets is not None:
            select_presets(args.check_presets)
            return 0
        selected = select_presets(args.presets)
    except ValueError as error:
        parser.error(str(error))
    configure_coreweave_s3()
    records = rl_index.list_attempts()
    started = []
    failed = False
    for model, preset in selected:
        try:
            version = rl_gate.config_version(rl_gate.compose_preset(preset))
            cohort = completed_controls(records, model=model, preset=preset.name, version=version)
            count = max(1, rl_gate.CONTROL_COUNT - len(cohort))
            print(
                json.dumps({"model": model, "preset": preset.name, "version": version, "requested": count}), flush=True
            )
            started.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        f"experiments.post_training.{model}_rl",
                        "--preset",
                        preset.name,
                        "--control",
                        args.batch,
                        "--replicates",
                        str(count),
                        "--version",
                        rl_compose.RUN_ARTIFACT_VERSION,
                        "--run",
                        "--wait",
                    ]
                )
            )
        except (ValueError, OSError, RuntimeError) as error:
            print(f"{model}/{preset.name}: {error}", file=sys.stderr)
            failed = True
    for child in started:
        failed |= child.wait() != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
