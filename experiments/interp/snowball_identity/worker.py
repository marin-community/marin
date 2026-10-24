# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A long-lived harness worker that takes job specs from an S3 queue (task #298).

``serve`` loads the checkpoint once and then polls the queue. ``submit`` adds a spec; ``status`` shows the
queue. Only the built-in commands below run: a spec carries data and options, never code.

Queue layout under ``--queue`` (an s3:// or local prefix)::

    inbox/<id>.json        specs waiting to run (oldest first)
    running/<id>.json      the spec being run
    results/<id>/...       the command's outputs, then done.json or error.json
    worker.json            heartbeat: state, current job, weights, updated time
    stop.json              ask the worker to exit after the current job (removed when honored)

The worker exits after ``--idle-minutes`` (default 60) with no work. A spec left in ``running/`` by a worker
that died is moved to ``results/<id>/error.json`` when the next worker starts.

Spec format (JSON). ``id`` is optional; ``submit`` makes one from the time.

``{"command": "positions" | "capture" | "generate", "args": [...]}``
    ``args`` are ``harness.py`` flags (``--prompts``, ``--positions``, ``--add-vector``, ``--neuron``,
    ``--patch``, ``--max-new-tokens``, ...). ``--weights`` and ``--output`` are set by the worker.
``{"command": "lens", "prompts": PATH, "top_neurons": 20}``
    Logit lens and direct logit attribution at the last position (``analyses.lens``). Rows need
    ``targets``; with two targets the attribution is of their logit difference.
``{"command": "patch", "pairs": PATH, ...}``
    Activation patching from clean into corrupt runs (``analyses.patch``). Pair rows:
    ``{"id", "clean": <prompt row>, "corrupt": <prompt row>, "targets": [correct, wrong]}``. Options:
    ``align`` (``tail`` or ``assistant``), ``tail`` (K last positions), ``components`` (from
    ``resid_pre``, ``attn``, ``routed``, ``shared``, ``resid``), ``heads``, ``per_position``, ``batch``.
    Reports ``recovery = (patched - corrupt) / (clean - corrupt)`` of the target logit difference.
``{"command": "steer", "prompts": PATH, "vector": ..., "layers": [...], "alphas": [...], ...}``
    Adds ``alpha * v`` after each layer in the grid (``analyses.steer``). ``vector`` is ``{"path": PT}`` or
    ``{"mean_diff": {"a": PATH, "b": PATH}}`` (mean last-position residual of set A minus set B, per layer),
    with optional ``"unit": true``. Options: ``positions`` (``assistant`` or ``all``), ``generate``
    (``{"max_new_tokens": N}`` for greedy text), ``batch``.

Prompt rows use ``prompts.py``'s format: ``{"id", "messages" | "text", "chat_template_kwargs"?, "prefill"?,
"targets"?}``.
"""

import argparse
import json
import logging
import time
import traceback
import uuid

import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.interp.snowball_identity import analyses
from experiments.interp.snowball_identity.harness import (
    COMMANDS,
    Output,
    build_parser,
    load_harness,
    run_command,
    stop_tokens,
)
from experiments.interp.snowball_identity.prompts import SNOWBALL_V01, load_tokenizer

logger = logging.getLogger(__name__)

ANALYSES = {"lens": analyses.lens, "patch": analyses.patch, "steer": analyses.steer}
POLL_SECONDS = 15


class Queue:
    def __init__(self, url: str):
        self.url = url
        self.fs, self.root = filesystem_for(url)

    def path(self, *parts: str) -> str:
        return prefix_join(self.root, "/".join(parts))

    def read(self, *parts: str) -> dict | None:
        path = self.path(*parts)
        if not self.fs.exists(path):
            return None
        return json.loads(self.fs.cat_file(path))

    def write(self, value: dict, *parts: str) -> None:
        path = self.path(*parts)
        self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        self.fs.pipe_file(path, json.dumps(value, indent=1).encode())

    def remove(self, *parts: str) -> None:
        path = self.path(*parts)
        if self.fs.exists(path):
            self.fs.rm(path)

    def ids(self, folder: str) -> list[str]:
        """Spec ids in ``folder``, oldest first."""
        self.fs.invalidate_cache()
        path = self.path(folder)
        if not self.fs.exists(path):
            return []
        entries = [entry for entry in self.fs.ls(path, detail=True) if entry["name"].endswith(".json")]
        entries.sort(key=lambda entry: (str(entry.get("LastModified") or entry.get("mtime") or ""), entry["name"]))
        return [entry["name"].rsplit("/", 1)[-1].removesuffix(".json") for entry in entries]

    def result(self, job_id: str) -> Output:
        return Output(prefix_join(self.url, f"results/{job_id}"))


def check_spec(spec: dict) -> None:
    command = spec.get("command")
    if command not in (*COMMANDS, *ANALYSES):
        raise ValueError(f"Unknown command {command!r}; expected one of {[*COMMANDS, *ANALYSES]}")
    if command in COMMANDS:
        args = spec.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError('"args" must be a list of strings')
        if any(arg.split("=")[0] in ("--weights", "--output") for arg in args):
            raise ValueError("--weights and --output are set by the worker")


def run_spec(harness, tokenizer, weights: str, stop_ids: set[int], spec: dict, output: Output) -> None:
    check_spec(spec)
    command = spec["command"]
    if command in COMMANDS:
        args = build_parser().parse_args([command, *spec.get("args", []), "--weights", weights, "--output", output.url])
        run_command(harness, tokenizer, args, output, stop_ids)
    elif command == "steer":
        analyses.steer(harness, tokenizer, spec, output, stop_ids)
    else:
        ANALYSES[command](harness, tokenizer, spec, output)


def serve(queue: Queue, weights: str, idle_minutes: float) -> None:
    for job_id in queue.ids("running"):
        spec = queue.read("running", f"{job_id}.json")
        queue.write(
            {"error": "the worker running this spec stopped before it finished", "spec": spec},
            "results",
            job_id,
            "error.json",
        )
        queue.remove("running", f"{job_id}.json")
    started = time.time()
    heartbeat = {"state": "loading", "weights": weights, "started": started, "jobs_done": 0, "job": None}

    def beat(**changes) -> None:
        heartbeat.update(changes, updated=time.time())
        queue.write(heartbeat, "worker.json")

    beat()
    harness = load_harness(weights)
    tokenizer = load_tokenizer(weights)
    stop_ids = stop_tokens(weights)
    beat(state="idle", loaded_seconds=round(time.time() - started))
    last_work = time.time()
    while True:
        if queue.read("stop.json") is not None:
            queue.remove("stop.json")
            beat(state="stopped", reason="stop.json")
            return
        pending = queue.ids("inbox")
        if not pending:
            if time.time() - last_work > idle_minutes * 60:
                beat(state="stopped", reason=f"idle for {idle_minutes:g} minutes")
                return
            beat()
            time.sleep(POLL_SECONDS)
            continue
        job_id = pending[0]
        try:
            spec = queue.read("inbox", f"{job_id}.json")
        except json.JSONDecodeError as error:
            spec = None
            failure = f"spec is not valid JSON: {error}"
        else:
            failure = None
        queue.remove("inbox", f"{job_id}.json")
        if spec is None:
            queue.write({"error": failure or "spec vanished"}, "results", job_id, "error.json")
            continue
        queue.write(spec, "running", f"{job_id}.json")
        beat(state="running", job=job_id)
        logger.info("job %s: %s", job_id, json.dumps(spec)[:500])
        job_started = time.time()
        output = queue.result(job_id)
        try:
            run_spec(harness, tokenizer, weights, stop_ids, spec, output)
        except Exception as error:
            logger.exception("job %s failed", job_id)
            torch.cuda.empty_cache()
            output.write_json(
                "error.json",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "spec": spec,
                    "seconds": round(time.time() - job_started, 1),
                },
            )
        else:
            output.write_json("done.json", {"spec": spec, "seconds": round(time.time() - job_started, 1)})
        queue.remove("running", f"{job_id}.json")
        last_work = time.time()
        beat(state="idle", job=None, jobs_done=heartbeat["jobs_done"] + 1)


def read_spec(value: str) -> dict:
    if value.lstrip().startswith("{"):
        return json.loads(value)
    fs, local = filesystem_for(value)
    return json.loads(fs.cat_file(local))


def submit(queue: Queue, spec: dict, wait: bool) -> dict | None:
    check_spec(spec)
    job_id = (
        spec.get("id") or f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{spec['command']}-{uuid.uuid4().hex[:6]}"
    )
    if queue.read("inbox", f"{job_id}.json") or queue.read("results", job_id, "done.json"):
        raise ValueError(f"Spec id {job_id!r} is already in use")
    queue.write({**spec, "id": job_id}, "inbox", f"{job_id}.json")
    print(json.dumps({"id": job_id, "results": prefix_join(queue.url, f"results/{job_id}")}))
    while wait:
        for name in ("done.json", "error.json"):
            outcome = queue.read("results", job_id, name)
            if outcome is not None:
                print(json.dumps({"outcome": name.removesuffix(".json"), **outcome}, indent=1))
                return outcome
        time.sleep(POLL_SECONDS)
    return None


def status(queue: Queue) -> dict:
    return {
        "worker": queue.read("worker.json"),
        "inbox": queue.ids("inbox"),
        "running": queue.ids("running"),
        "stop_requested": queue.read("stop.json") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    serve_parser = commands.add_parser("serve", help="load the model and run queued specs")
    serve_parser.add_argument("--queue", required=True)
    serve_parser.add_argument("--weights", default=SNOWBALL_V01)
    serve_parser.add_argument("--idle-minutes", type=float, default=60)
    submit_parser = commands.add_parser("submit", help="queue one spec")
    submit_parser.add_argument("--queue", required=True)
    submit_parser.add_argument("--spec", required=True, help="spec JSON, or a path to it (s3:// or local)")
    submit_parser.add_argument("--wait", action="store_true", help="poll until done.json or error.json appears")
    for name, text in (("status", "show the worker heartbeat and the queue"), ("stop", "ask the worker to exit")):
        commands.add_parser(name, help=text).add_argument("--queue", required=True)
    args = parser.parse_args()

    queue = Queue(args.queue)
    if args.action == "serve":
        serve(queue, args.weights, args.idle_minutes)
    elif args.action == "submit":
        submit(queue, read_spec(args.spec), args.wait)
    elif args.action == "status":
        print(json.dumps(status(queue), indent=1))
    else:
        queue.write({"requested": time.time()}, "stop.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
