# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the TMax reserve pilot: serve a catalog model on Iris and run the pilot Harbor policy in snapshot waves.

Runs as one CPU Iris coordinator job. It lowers the model exactly as the evaluation launcher does
(``inference_config_for_model`` + ``remote_inference``), waits for the endpoint, and runs the policy through
``run_in_waves`` on ``run_harbor_driver``. The model's agent kwargs come from ``harbor_model_agent_kwargs``, as
in the TBLite eval. The summary reports per-task pass@1 and pass@k, trial errors, live sandbox hours, and how
many trials ran apt (the Ubuntu mirror is the one egress Daytona's lockdown leaves open).
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml
from iris.rpc import job_pb2
from marin.evaluation.eval_env import DAYTONA_RUNTIME_ENV_KEYS
from marin.evaluation.harbor.agent_context import CANDIDATE_API_KEY_ENV
from marin.evaluation.harbor.driver_config import HarborRuntimeOverlay, preflight_harbor_configs
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret

from experiments.evaluation.evals import harbor_model_agent_kwargs
from experiments.evaluation.models import models
from experiments.post_training.tmax_reserve.null_run import trial_results
from experiments.post_training.tmax_reserve.waves import HeadroomGate, run_in_waves, wave_usage

logger = logging.getLogger(__name__)

HEADROOM_POLL_SECONDS = 300
SERVE_GPU = "H100"
SERVE_GPU_COUNT = 8
# An agent command that installs or refreshes apt packages; the mirror is reachable under the lockdown.
APT_COMMAND = re.compile(r"\bapt(?:-get)?\s+(?:-\S+\s+)*(?:install|update|upgrade)\b")
# apt's download lines name the mirror it fetched from.
APT_FETCH = re.compile(r"\bGet:\d+ https?://\S*ubuntu\.com")


def apt_usage(trial_dirs: list[Path]) -> dict:
    """Count trials whose agent ran an apt command in any attempt, and those whose terminal shows a mirror download."""
    ran, fetched = [], []
    for trial in trial_dirs:
        steps = [
            step
            for trajectory in sorted(trial.glob("attempts/*/agent/trajectory.json"))
            for step in json.loads(trajectory.read_text()).get("steps", [])
        ]
        keystrokes = " ".join(
            str(call.get("arguments", {}).get("keystrokes", ""))
            for step in steps
            for call in step.get("tool_calls") or []
        )
        screens = " ".join(
            str(result.get("content", ""))
            for step in steps
            for result in (step.get("observation") or {}).get("results") or []
        )
        if APT_COMMAND.search(keystrokes):
            ran.append(trial.name)
        if APT_FETCH.search(screens):
            fetched.append(trial.name)
    return {"trials_scanned": len(trial_dirs), "ran_apt": len(ran), "fetched_from_mirror": len(fetched)}


def pass_at_k(trials: list[dict]) -> dict:
    """Per-task pass@1 (mean success) and pass@k (any success) over scored trials; rewards > 0 count as passes."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for trial in trials:
        if trial["reward"] is not None:
            by_task[trial["task"]].append(trial["reward"] > 0)
    tasks = {task: {"attempts": len(passes), "passes": sum(passes)} for task, passes in sorted(by_task.items())}
    return {
        "tasks_scored": len(tasks),
        "mean_pass_at_1": (sum(t["passes"] / t["attempts"] for t in tasks.values()) / len(tasks)) if tasks else None,
        "pass_at_k": (sum(t["passes"] > 0 for t in tasks.values()) / len(tasks)) if tasks else None,
        "tasks": tasks,
    }


def summarize(trials: list[dict], seconds: float, apt: dict) -> dict:
    errors: dict[str, int] = defaultdict(int)
    for trial in trials:
        if trial["exception"]:
            errors[trial["exception"]] += 1
    return {
        "trials": len(trials),
        "trials_without_reward": sum(trial["reward"] is None for trial in trials),
        "errors": dict(sorted(errors.items())),
        "live_sandbox_hours": seconds / 3600,
        "apt": apt,
        **pass_at_k(trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, required=True, help="The pilot Harbor policy (local dataset path).")
    parser.add_argument("--model", required=True, help="Evaluation catalog model name.")
    parser.add_argument("--wave-size", type=int, required=True, help="Most tasks per wave (see waves.py).")
    parser.add_argument("--min-wave-size", type=int, required=True, help="Wait rather than run a smaller wave.")
    parser.add_argument("--snapshot-quota", type=int, required=True, help="The account's snapshot quota.")
    parser.add_argument("--headroom-wait", type=int, required=True, help="Seconds to wait for room per wave.")
    parser.add_argument("--max-sandbox-hours", type=float, required=True, help="Stop after the wave that passes it.")
    parser.add_argument("--output", type=Path, required=True, help="New local directory for jobs and the summary.")
    parser.add_argument("--upload", required=True, help="Remote directory that receives a copy of --output.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True)
    model = models()[args.model]
    model_kwargs = harbor_model_agent_kwargs(model)
    (config,) = preflight_harbor_configs([(args.policy, model_kwargs)], judge_model=None)
    dataset = (args.policy.parent / yaml.safe_load(args.policy.read_text())["datasets"][0]["path"]).resolve()
    inference = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type=SERVE_GPU, gpu_count=SERVE_GPU_COUNT),
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    gate = HeadroomGate(args.snapshot_quota, args.min_wave_size, HEADROOM_POLL_SECONDS, args.headroom_wait)
    try:
        with remote_inference(inference) as session:
            session.wait_until_ready()
            logger.info("serving %s at %s", model.name, session.model.endpoint.base_url)
            overlay = HarborRuntimeOverlay(
                job_name="tmax-pilot",
                jobs_dir=str(args.output / "jobs"),
                dataset_path=str(dataset),
                endpoint_url=session.model.endpoint.base_url,
                served_model=session.model.endpoint.model,
                model_identity=model.location,
                task_limit=None,
                model_agent_kwargs=model_kwargs,
                verifier_env={},
                archive_root=str(args.output / "archive"),
                archive_dataset="tmax-pilot",
            )
            driver_env = {
                "DAYTONA_API_KEY": require_secret("DAYTONA_API_KEY"),
                CANDIDATE_API_KEY_ENV: session.model.endpoint.api_key or "EMPTY",
            }
            driver_env.update({key: os.environ[key] for key in DAYTONA_RUNTIME_ENV_KEYS if key in os.environ})
            run_in_waves(
                config,
                overlay,
                driver_env,
                session.backend_state,
                args.wave_size,
                gate,
                args.output,
                args.max_sandbox_hours,
            )
    finally:
        jobs_dirs = sorted((args.output / "waves").glob("w*/jobs"))
        trials = trial_results(jobs_dirs)
        seconds = sum(wave_usage(jobs_dir)[0] for jobs_dir in jobs_dirs)
        apt = apt_usage(sorted(path.parent for jobs_dir in jobs_dirs for path in jobs_dir.glob("*/*/result.json")))
        summary = summarize(trials, seconds, apt)
        (args.output / "pilot-trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
        (args.output / "pilot-summary.json").write_text(json.dumps(summary, indent=1))
        fs, path = filesystem_for(args.upload)
        fs.put(str(args.output), path, recursive=True)


if __name__ == "__main__":
    main()
