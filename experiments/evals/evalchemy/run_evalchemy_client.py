# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy OpenAI-client entrypoint, run inside the ``:evalchemy-tpu`` container.

The eval child runs this as a plain command under the image's own interpreter
(``/opt/openthoughts/.venv/bin/python``) -- the only interpreter in that image with ``eval``,
``lm_eval``, ``fsspec`` and ``gcsfs`` installed. It is deliberately a *command* entrypoint, not an
Iris ``from_callable`` one: the image's default/synced interpreter is a bare python with no
cloudpickle, so a cloudpickled callable cannot be deserialized there (issue #7267). Keeping this
script to the standard library plus ``fsspec`` lets that interpreter run it directly.

Config arrives as JSON in ``$EVALCHEMY_CLIENT_CONFIG`` (the parent builds it in
:mod:`experiments.evals.evalchemy.serve_and_eval`), so nothing marin-side needs to import here.
Each task runs through the evalchemy fork's ``eval.eval`` once (one invocation per task so each
carries its own ``num_fewshot``) with lm-eval's ``local-completions`` (or ``local-chat-completions``)
API model pointed at the served URL, and its native ``results_*.json`` tree is uploaded to
``out_path/<dir>/`` for :class:`~marin.evaluation.eval_result.EvalchemyResult` to read back.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import fsspec

CONFIG_ENV_KEY = "EVALCHEMY_CLIENT_CONFIG"


def build_model_args(config: dict) -> str:
    """lm-eval ``--model_args`` for the served OpenAI endpoint (comma-joined ``key=value`` list)."""
    endpoint_path = "chat/completions" if config["apply_chat_template"] else "completions"
    return ",".join(
        [
            f"model={config['model_id']}",
            f"base_url={config['base_url'].rstrip('/')}/{endpoint_path}",
            f"tokenizer={config['tokenizer']}",
            "tokenizer_backend=huggingface",
            # LOCAL (uncommitted): some tokenizers ship custom code (e.g. Moonlight/DeepseekV3) and
            # lm-eval refuses to load them without this; harmless (no-op) for tokenizers that don't.
            "trust_remote_code=True",
            "tokenized_requests=False",
            f"num_concurrent={config['num_concurrent']}",
        ]
    )


def build_command(config: dict, task: dict, output_path: str, python: str) -> list[str]:
    """The ``eval.eval`` argv for one task. ``python`` runs the evalchemy fork + lm-eval in its venv.

    One invocation per task so each carries its own ``num_fewshot`` (lm-eval's ``--num_fewshot`` is a
    single global override). Chat vs completion route follows ``apply_chat_template``.
    """
    model = "local-chat-completions" if config["apply_chat_template"] else "local-completions"
    cmd = [
        python,
        "-m",
        "eval.eval",
        "--model",
        model,
        "--model_args",
        build_model_args(config),
        "--tasks",
        task["name"],
        "--gen_kwargs",
        f"max_gen_toks={config['max_gen_toks']}",
        "--output_path",
        output_path,
        "--log_samples",
        "--verbosity",
        "INFO",
    ]
    if task["num_fewshot"]:
        cmd += ["--num_fewshot", str(task["num_fewshot"])]
    if task.get("seed") is not None:  # LOCAL: AIME24 10-seed μ±σ (one process per seed)
        cmd += ["--seed", str(task["seed"])]
    if config["max_eval_instances"] is not None:
        cmd += ["--limit", str(config["max_eval_instances"])]
    if config["apply_chat_template"]:
        cmd.append("--apply_chat_template")
        # LOCAL (uncommitted): evalchemy chat_benchmarks (MATH500/AIME24/…) read `--max_tokens`, NOT
        # `--gen_kwargs max_gen_toks` (that only bounds lm-eval NATIVE tasks). Unpinned → they truncate
        # at a tiny default (~256 tok) → thinking CoT never reaches the boxed answer → score 0 (POLICY
        # §3/§4: pin --max_tokens). Pin it to the generation budget.
        cmd += ["--max_tokens", str(config["max_gen_toks"])]
    return cmd


def main() -> None:
    config = json.loads(os.environ[CONFIG_ENV_KEY])
    tasks = config["tasks"]
    if not tasks:
        raise SystemExit("run_evalchemy_client requires at least one task")

    out_path = config["out_path"].rstrip("/")
    # Raw fsspec, not rigging's StoragePath: the eval image carries only fsspec/gcsfs, not rigging.
    # out_path is region-local (the eval child is pinned to the serve region), so no cross-region copy.
    # LOCAL (uncommitted): for a CoreWeave LOTA s3 out_path (marin-us-east-02a), s3fs must use the
    # injected in-pod endpoint + VIRTUAL addressing (LOTA rejects path-style — .claude/ops/iris/ops.md
    # §Scheduling). The iris-task-env Secret injects AWS_* + AWS_ENDPOINT_URL into the pod.
    storage_options: dict = {}
    if out_path.startswith("s3://") and os.environ.get("AWS_ENDPOINT_URL"):
        storage_options = {
            "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL"]},
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
        }
        # LOCAL (uncommitted): s3fs lives in the workspace venv, NOT the eval image's own venv
        # (/opt/eval/evalchemy/.venv, which runs this script) → install it on demand (pod has egress).
        try:
            import s3fs  # noqa: F401,PLC0415
        except ImportError:
            for installer in (
                ["uv", "pip", "install", "--python", sys.executable, "s3fs"],
                [sys.executable, "-m", "pip", "install", "s3fs"],
            ):
                try:
                    subprocess.run(installer, check=True)
                    break
                except Exception:  # noqa: BLE001
                    continue
    # s3 write is best-effort: a durable-store hiccup must NEVER lose a task's scores (the
    # EVALCHEMY_RESULT stdout print is the reliable, Mac-reachable harvest source). out_fs=None
    # disables the put; the run still completes + prints every score.
    try:
        out_fs, _ = fsspec.core.url_to_fs(out_path, **storage_options)
    except Exception as e:  # noqa: BLE001
        print(f"EVALCHEMY_S3_SETUP_ERROR out_path={out_path} err={e!r} "
              "(scores still harvestable from EVALCHEMY_RESULT log lines)", flush=True)
        out_fs = None
    # LOCAL (uncommitted): some evalchemy chat_benchmarks import deps missing from the image venv
    # (HumanEvalPlus needs `fire`) → their registration throws → "Task not found". Install on demand
    # (pod has egress). Non-fatal: a still-missing dep just skips that benchmark (loop is non-fatal).
    for _dep in ("fire",):
        try:
            __import__(_dep)
        except Exception:  # noqa: BLE001
            for installer in (["uv", "pip", "install", "--python", sys.executable, _dep],
                              [sys.executable, "-m", "pip", "install", _dep]):
                try:
                    subprocess.run(installer, check=True)
                    break
                except Exception:  # noqa: BLE001
                    continue
    for task in tasks:
        with tempfile.TemporaryDirectory() as local_out:
            # sys.executable is the evalchemy image's interpreter, so ``-m eval.eval`` resolves the
            # fork + lm-eval baked into its venv.
            cmd = build_command(config, task, local_out, sys.executable)
            print(f"running evalchemy: {' '.join(cmd)}", flush=True)
            # LOCAL (uncommitted): NON-FATAL per benchmark — one broken task (e.g. a chat_benchmark
            # whose module fails to import a missing dep) must NOT abort the whole suite. Log + skip;
            # the working benchmarks still score + upload. (Was check=True → a single failure killed all.)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"EVALCHEMY_RESULT task={task['name']} STATUS=SUBPROCESS_FAILED rc={rc} "
                      "(skipped — non-fatal; remaining tasks continue)", flush=True)
                continue
            # LOCAL (uncommitted): parse the results JSON and PRINT the aggregate metrics to stdout so
            # they land in the durable finelog. out_path is often pod-local /tmp (lost when the pod
            # tears down) and evalchemy prints no results table — without this the scores are
            # unrecoverable, and an empty `results: {}` (the COMPLETED-not-success trap) is invisible.
            import glob  # noqa: PLC0415
            result_files = sorted(glob.glob(os.path.join(local_out, "**", "results*.json"), recursive=True))
            if not result_files:
                print(f"EVALCHEMY_RESULT task={task['name']} STATUS=NO_RESULT_FILE "
                      f"contents={os.listdir(local_out)}", flush=True)
            for rf in result_files:
                try:
                    data = json.load(open(rf))
                    res = data.get("results", data)
                    status = "EMPTY" if not res else "OK"
                    print(f"EVALCHEMY_RESULT task={task['name']} STATUS={status} "
                          f"results={json.dumps(res)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"EVALCHEMY_RESULT task={task['name']} STATUS=PARSE_ERROR "
                          f"file={rf} err={e!r}", flush=True)
            if out_fs is not None:
                try:
                    out_fs.put(local_out, f"{out_path}/{task['dir']}", recursive=True)
                except Exception as e:  # noqa: BLE001
                    print(f"EVALCHEMY_S3_PUT_ERROR task={task['name']} err={e!r}", flush=True)
    print(f"evalchemy client wrote results for {len(tasks)} task(s) to {out_path}", flush=True)
    # LOCAL (uncommitted): confirm the durable JSONs actually landed at the (LOTA) s3 prefix — an
    # in-pod listing proves the deliverable is readable, visible in the finelog.
    if out_fs is not None:
        try:
            landed = [p for p in out_fs.find(out_path) if p.endswith(".json")]
            print(f"EVALCHEMY_S3_LANDED count={len(landed)} prefix={out_path} files={landed}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"EVALCHEMY_S3_LANDED_ERROR prefix={out_path} err={e!r}", flush=True)


if __name__ == "__main__":
    main()
