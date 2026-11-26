# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-region postprocessing for one terminal loop-guards run.

Promoted from the ad-hoc ``submit-s1-audit.py``, ``submit-s0-v12-postprocess.py`` and
``submit-s0-report-refresh.py`` worker scripts. One subcommand chains the pieces a
terminal run needs; every step is also a standalone subcommand:

* ``audit`` — the panel audit (the existing ``audit`` module).
* ``--counterfactual`` / ``--census`` — the S0-style analyses, run under SkyRL's frozen
  CPU environment like their originals.
* ``retain-traces`` — promote the run's shared trajectory sink with ``fsutil
  verified-copy`` once the coordinator and GPU jobs are terminal.
* ``--preset-name`` — refresh the saved-data reporting for that stage preset.

    … cpu-submit --cpu 16 --memory 64GB --disk 128GB --timeout 14400 \\
        --env MARIN_PREFIX=s3://marin-us-east-08a/marin -- \\
      python -m experiments.post_training.mochi_loop_guards postprocess \\
      --run-output <run output> --output <audit prefix> --baseline <sealed audit> \\
      --retain-traces s3://…/traces/s1-stop --preset-name s1-stop

Wait for terminal job state (and the final cohort dump) before launching: sequencing
belongs to the operator and the job watchdog, not to this subcommand.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml
from marin.rl.skyrl import skyrl_cpu_command
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.mochi_loop_guards import audit

REGIONAL_ROOT = "s3://marin-us-east-08a/marin"
EXECUTION_REGION = "s3://marin-us-east-08a/"
PACKAGE = "marin-core"
FSUTIL_WORKERS = 8
RETENTION_MANIFEST_NAME = ".verified-copy-manifest.json"


def retention_source(run_output: str) -> str:
    """Read the run's shared trajectory-retention sink from its resolved launch document."""
    document = yaml.safe_load((StoragePath(run_output) / "resolved-launch.yaml").read_text())["config"]
    source = document["skyrl"]["generator"]["trajectory_retention"]["output_path"]
    if not source.startswith(EXECUTION_REGION):
        raise ValueError(f"Trajectory retention sink {source!r} is outside the execution region")
    return source


def verified_copy_command(source: str, destination: str) -> list[str]:
    """The ``fsutil verified-copy`` argv that promotes one same-region sink."""
    return [
        "uv",
        "run",
        "--package",
        PACKAGE,
        "fsutil",
        "verified-copy",
        source,
        destination,
        "--workers",
        str(FSUTIL_WORKERS),
    ]


def retain_traces(argv: list[str] | None = None) -> None:
    """Promote the run's trajectory-retention sink to its permanent prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True, help="Directory for the retention-promotion receipt")
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    source = retention_source(args.run_output)
    subprocess.run(verified_copy_command(source, args.destination), check=True)
    manifest = (StoragePath(args.destination) / RETENTION_MANIFEST_NAME).read_bytes()
    receipt = {
        "source": source,
        "destination": args.destination,
        "completion_manifest": f"{args.destination}/{RETENTION_MANIFEST_NAME}",
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "same_region": True,
    }
    (StoragePath(args.output) / "retention-promotion.json").write_text(json.dumps(receipt, indent=2) + "\n")


def report_refresh(argv: list[str] | None = None) -> None:
    """Refresh the saved-data evaluation status and attempt summary for one stage preset."""
    # The presets table is bound in the package launcher; importing it here keeps the
    # analysis modules free of a launcher import at module load.
    from experiments.post_training import rl_compose
    from experiments.post_training.mochi_loop_guards.__main__ import GUARDS
    from experiments.post_training.rl_measurements import evaluation_status, read_recorded_launch
    from experiments.post_training.rl_publication import build_record
    from marin.rl.skyrl_records import read_stored_skyrl_attempts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--preset-name", required=True, help="Stage preset name in the loop-guards table")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    rl_compose.SELECTION_ROOT = f"{REGIONAL_ROOT}/rl-data/task-selection"
    run = args.run_output
    attempt_id, attempts = read_stored_skyrl_attempts(run)
    attempt = attempts[attempt_id]
    assert attempt.native.result.state == "succeeded"
    preset = GUARDS.by_name[args.preset_name]
    status = evaluation_status(run, preset, attempt, read_recorded_launch(attempt))
    (StoragePath(args.output) / "refreshed-evaluation-status.json").write_text(status.model_dump_json(indent=2) + "\n")
    summary = build_record(run, preset, table=GUARDS, attempt=attempt)
    (StoragePath(args.output) / "refreshed-attempt-summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    receipt = {
        "run_output": run,
        "attempt_id": attempt_id,
        "model_trials_created": 0,
        "evaluation_error": status.evaluation_error,
        "evaluations": len(status.evaluations),
        "failed_facts": summary.failed_facts,
        "summary_status": summary.status,
        "summary_output": str(StoragePath(args.output) / "refreshed-attempt-summary.json"),
    }
    (StoragePath(args.output) / "report-refresh-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    assert status.evaluation_error is None, status.evaluation_error


def _skyrl_analysis_command(module: str, *arguments: str) -> list[str]:
    return skyrl_cpu_command("python", "-m", f"experiments.post_training.mochi_loop_guards.{module}", *arguments)


def postprocess(argv: list[str] | None = None) -> None:
    """Audit one terminal run and apply the registered postprocessing chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--exclude-cohort", action="append", default=[], choices=tuple(audit.MANIFEST_COHORTS.values()))
    parser.add_argument("--counterfactual", action="store_true", help="Run the registered counterfactual analysis")
    parser.add_argument("--census", action="store_true", help="Run the saved-feedback observation census")
    parser.add_argument("--retain-traces", metavar="DESTINATION", help="Promote the trajectory sink to this prefix")
    parser.add_argument("--preset-name", help="Refresh saved-data reporting for this stage preset")
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    run_output = args.run_output
    output = StoragePath(args.output)
    output.mkdirs()
    pattern = str(StoragePath(run_output) / "exports/**/dumped_evals/*/eval_only/*.jsonl")
    steps = ["audit"]
    audit.main(
        [
            "--run-output",
            run_output,
            "--output",
            args.output,
            *(["--baseline", args.baseline] if args.baseline else []),
            *sum((["--exclude-cohort", cohort] for cohort in args.exclude_cohort), []),
        ]
    )
    manifest_path = str(output / "resolved-manifest.json")
    counterfactual_path = Path(audit.__file__).with_name("counterfactual.py")
    if args.counterfactual:
        steps.append("counterfactual")
        (output / "regional-counterfactual.py").write_text(counterfactual_path.read_text())
        subprocess.run(
            _skyrl_analysis_command(
                "counterfactual", "--manifest", manifest_path, "--pattern", pattern, "--output", str(output / "counterfactual")
            ),
            check=True,
        )
    if args.census:
        steps.append("census")
        subprocess.run(
            _skyrl_analysis_command(
                "observation_census", "--manifest", manifest_path, "--pattern", pattern, "--output", str(output / "observation-census.json")
            ),
            check=True,
        )
    if args.retain_traces:
        steps.append("retain-traces")
        retain_traces(["--run-output", run_output, "--destination", args.retain_traces, "--output", args.output])
    if args.preset_name:
        steps.append("report-refresh")
        report_refresh(["--run-output", run_output, "--preset-name", args.preset_name, "--output", args.output])
    receipt = {
        "run_output": run_output,
        "baseline": args.baseline,
        "steps": steps,
        "model_trials_created": 0,
        "complete": True,
        "counterfactual_module_sha256": hashlib.sha256(counterfactual_path.read_bytes()).hexdigest()
        if args.counterfactual
        else None,
    }
    (output / "postprocessing-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main(argv: list[str] | None = None) -> None:
    handlers = {
        "postprocess": postprocess,
        "retain-traces": retain_traces,
        "report-refresh": report_refresh,
    }
    handler = handlers.get((argv or [None])[0])
    if handler is None:
        raise SystemExit(f"expected one of {sorted(handlers)} as the first argument")
    handler(argv[1:])


if __name__ == "__main__":
    main()
