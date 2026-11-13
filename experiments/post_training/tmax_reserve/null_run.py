# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-free integrity run of the pilot tasks through the same Harbor driver, policy and Terminus-2 as the pilot.

A scripted OpenAI-compatible server stands in for the model, in this process. Terminus-2 calls the model from the
Harbor process, so the sandbox never reaches it. For every task the script's first reply runs a probe in the agent's
tmux shell (tmux version, then a HEAD request to each ``EGRESS_TARGETS`` URL) and changes nothing; its next reply
marks the task complete. For a control task (``--controls``: task name -> known-good shell commands) the first reply
runs those commands instead. Checks, written to ``<output>/null-summary.json``:
- every non-control task scores 0 (an intact verifier on an unchanged environment);
- every control task scores 1 (the verifier passes on a known-good state);
- every probe shows tmux, and only iris.oa.dev answers.

Tasks run in waves (``waves.py``) so the snapshots fit the Daytona quota. Only Daytona sandboxes run; there is no
GPU or model.
"""

import argparse
import base64
import collections
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import uvicorn
import yaml
from marin.evaluation.eval_env import DAYTONA_RUNTIME_ENV_KEYS
from marin.evaluation.harbor.agent_context import CANDIDATE_API_KEY_ENV, served_model_info
from marin.evaluation.harbor.driver_config import HarborRuntimeOverlay, preflight_harbor_configs
from marin.inference.dashboard_server import bind_serving_socket
from marin.inference.types import InferenceBackendState
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from experiments.post_training.tmax_reserve.waves import HeadroomGate, run_in_waves

logger = logging.getLogger(__name__)

SERVED_MODEL = "scripted-probe"
ALLOWED_HOST = "https://iris.oa.dev/"
EGRESS_TARGETS = (
    "https://github.com/",
    "https://api.github.com/",
    "https://raw.githubusercontent.com/",
    "https://pypi.org/simple/pip/",
    "https://files.pythonhosted.org/",
    "http://archive.ubuntu.com/ubuntu/",
    "https://huggingface.co/",
    "https://www.google.com/",
    ALLOWED_HOST,
    # By IP address, without DNS: GitHub and Cloudflare's resolver.
    "https://140.82.112.3/",
    "https://1.1.1.1/",
)
_PROBE_PY = f"""
import ssl, urllib.request, urllib.error
ctx = ssl._create_unverified_context()
for url in {list(EGRESS_TARGETS)!r}:
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=5, context=ctx).status
    except urllib.error.HTTPError as e:
        r = e.code
    except Exception as e:
        r = type(e).__name__
    print("EGRESS", url, r)
"""
PROBE_KEYSTROKES = (
    'echo "TMUX $(tmux -V 2>&1 || echo missing)"; '
    f'python3 -c "$(echo {base64.b64encode(_PROBE_PY.encode()).decode()} | base64 -d)"; echo PROBE_DONE\n'
)
PROBE_DURATION = 60
HEADROOM_POLL_SECONDS = 300
EGRESS_LINE = re.compile(r"^EGRESS (\S+) (.+)$", re.MULTILINE)
TMUX_LINE = re.compile(r"^TMUX (.+)$", re.MULTILINE)


def _reply(commands: list[dict], complete: bool) -> str:
    return json.dumps(
        {"analysis": "scripted probe", "plan": "scripted probe", "commands": commands, "task_complete": complete}
    )


class ScriptedModel:
    """First reply: probe (or the control's commands). Later replies: task complete. Records probe output."""

    def __init__(self, instructions: dict[str, str], controls: dict[str, str]):
        self.instructions = instructions
        self.controls = controls
        self.probes: dict[str, dict] = {}
        self.lock = threading.Lock()

    def task_of(self, messages: list[dict]) -> str | None:
        first = next((m["content"] for m in messages if m["role"] == "user"), "")
        text = first if isinstance(first, str) else json.dumps(first)
        matches = [task for task, instruction in self.instructions.items() if instruction.strip() in text]
        return matches[0] if len(matches) == 1 else None

    def answer(self, messages: list[dict]) -> str:
        task = self.task_of(messages)
        turns = sum(m["role"] == "assistant" for m in messages)
        if turns == 0:
            if task in self.controls:
                return _reply([{"keystrokes": self.controls[task], "duration": PROBE_DURATION}], False)
            return _reply([{"keystrokes": PROBE_KEYSTROKES, "duration": PROBE_DURATION}], False)
        if turns == 1 and task is not None and task not in self.controls:
            seen = "\n".join(m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str))
            with self.lock:
                self.probes.setdefault(task, []).append(
                    {
                        "tmux": TMUX_LINE.findall(seen)[-1:] or None,
                        "egress": {url: result.strip() for url, result in EGRESS_LINE.findall(seen)},
                        "probe_done": "PROBE_DONE" in seen,
                    }
                )
        return _reply([], True)

    def app(self) -> Starlette:
        async def chat(request: Request) -> JSONResponse:
            body = await request.json()
            content = self.answer(body["messages"])
            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": SERVED_MODEL,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        async def tokenize(request: Request) -> JSONResponse:
            body = await request.json()
            return JSONResponse({"count": len(json.dumps(body)) // 4})

        async def models(_: Request) -> JSONResponse:
            return JSONResponse({"object": "list", "data": [{"id": SERVED_MODEL, "object": "model"}]})

        return Starlette(
            routes=[
                Route("/v1/chat/completions", chat, methods=["POST"]),
                Route("/tokenize", tokenize, methods=["POST"]),
                Route("/v1/models", models, methods=["GET"]),
            ]
        )


def serve(app: Starlette) -> str:
    sock = bind_serving_socket("127.0.0.1", 0)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
    deadline = time.monotonic() + 60
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("scripted model server did not start")
        time.sleep(0.2)
    return f"http://127.0.0.1:{sock.getsockname()[1]}/v1"


def null_policy(policy: Path, dest: Path) -> Path:
    """The pilot policy with one attempt per task, next to it so its relative dataset path still resolves."""
    config = yaml.safe_load(policy.read_text())
    config["n_attempts"] = 1
    dest.write_text(yaml.safe_dump(config, sort_keys=False))
    return dest


def trial_results(jobs_dirs: list[Path]) -> list[dict]:
    rows = []
    for result in sorted(path for jobs_dir in jobs_dirs for path in jobs_dir.glob("*/*/result.json")):
        data = json.loads(result.read_text())
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        exc = data.get("exception_info")
        rows.append(
            {
                "task": data["task_name"].rsplit("/", 1)[-1],  # Harbor prefixes the dataset name
                "reward": rewards.get("reward"),
                "exception": exc.get("exception_type") if exc else None,
            }
        )
    return rows


def summarize(trials: list[dict], probes: dict[str, list[dict]], controls: set[str]) -> dict:
    egress = collections.defaultdict(collections.Counter)
    for runs in probes.values():
        for run in runs:
            for url, result in run["egress"].items():
                egress[url]["reached" if result.isdigit() else "blocked"] += 1
    null = [t for t in trials if t["task"] not in controls]
    control = [t for t in trials if t["task"] in controls]
    tmux = collections.Counter(
        bool(run["tmux"]) and "missing" not in run["tmux"][0] for runs in probes.values() for run in runs
    )
    return {
        "trials": len(trials),
        "exceptions": dict(collections.Counter(t["exception"] for t in trials if t["exception"])),
        "null_reward_counts": dict(collections.Counter(str(t["reward"]) for t in null)),
        "null_tasks_scoring_nonzero": sorted(t["task"] for t in null if t["reward"]),
        "tasks_without_reward": sorted(t["task"] for t in trials if t["reward"] is None),
        "control_rewards": {t["task"]: t["reward"] for t in control},
        "probes": sum(len(runs) for runs in probes.values()),
        "probes_with_tmux": tmux[True],
        "probes_incomplete": sum(not run["probe_done"] for runs in probes.values() for run in runs),
        "egress": {url: dict(counts) for url, counts in sorted(egress.items())},
        "egress_violations": sorted(url for url, counts in egress.items() if url != ALLOWED_HOST and counts["reached"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, required=True, help="The pilot Harbor policy (local dataset path).")
    parser.add_argument("--wave-size", type=int, required=True, help="Most tasks per wave (see waves.py).")
    parser.add_argument("--min-wave-size", type=int, required=True, help="Wait rather than run a smaller wave.")
    parser.add_argument("--snapshot-quota", type=int, required=True, help="The account's snapshot quota.")
    parser.add_argument("--headroom-wait", type=int, required=True, help="Seconds to wait for room per wave.")
    parser.add_argument("--max-sandbox-hours", type=float, required=True, help="Stop after the wave that passes it.")
    parser.add_argument("--controls", type=Path, help="JSON {task name: known-good shell commands}.")
    parser.add_argument("--output", type=Path, required=True, help="New local directory for jobs and the summary.")
    parser.add_argument("--upload", help="Remote directory that receives a copy of --output at the end.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True)
    policy = null_policy(args.policy, args.policy.with_name(f"{args.policy.stem}-null.yaml"))
    model_kwargs = {"model_info": served_model_info(32768, 4096)}
    (config,) = preflight_harbor_configs([(policy, model_kwargs)], judge_model=None)
    dataset = (policy.parent / yaml.safe_load(policy.read_text())["datasets"][0]["path"]).resolve()
    instructions = {task.name: (task / "instruction.md").read_text() for task in sorted(dataset.iterdir())}
    controls = json.loads(args.controls.read_text()) if args.controls else {}
    if set(controls) - set(instructions):
        raise ValueError(f"controls not in the dataset: {sorted(set(controls) - set(instructions))}")
    scripted = ScriptedModel(instructions, controls)
    endpoint = serve(scripted.app())
    overlay = HarborRuntimeOverlay(
        job_name="tmax-reserve-null",
        jobs_dir=str(args.output / "jobs"),
        dataset_path=str(dataset),
        endpoint_url=endpoint,
        served_model=SERVED_MODEL,
        model_identity=SERVED_MODEL,
        task_limit=None,
        model_agent_kwargs=model_kwargs,
        verifier_env={},
        archive_root=str(args.output / "archive"),
        archive_dataset="tmax-reserve-null",
    )
    driver_env = {"DAYTONA_API_KEY": require_secret("DAYTONA_API_KEY"), CANDIDATE_API_KEY_ENV: "EMPTY"}
    driver_env.update({key: os.environ[key] for key in DAYTONA_RUNTIME_ENV_KEYS if key in os.environ})
    gate = HeadroomGate(args.snapshot_quota, args.min_wave_size, HEADROOM_POLL_SECONDS, args.headroom_wait)
    try:
        run_in_waves(
            config,
            overlay,
            driver_env,
            lambda: InferenceBackendState.READY,
            args.wave_size,
            gate,
            args.output,
            args.max_sandbox_hours,
        )
    finally:
        trials = trial_results(sorted((args.output / "waves").glob("w*/jobs")))
        summary = summarize(trials, scripted.probes, set(controls))
        (args.output / "null-trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
        (args.output / "null-probes.json").write_text(json.dumps(scripted.probes, indent=1))
        (args.output / "null-summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        if args.upload:
            fs, path = filesystem_for(args.upload)
            fs.put(str(args.output), path, recursive=True)


if __name__ == "__main__":
    main()
