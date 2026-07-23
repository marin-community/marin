# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy OpenAI-client entrypoint, run inside the ``:evalchemy-tpu`` container.

The eval child runs this as a plain command under the image's own interpreter
(``/opt/openthoughts/.venv/bin/python``) -- the only interpreter in that image with ``eval``,
``lm_eval`` and ``fsspec`` (plus the ``s3fs``/``gcsfs`` backends) installed. It is a command
entrypoint, not an Iris ``from_callable`` one: the image's default/synced interpreter is a bare
python with no cloudpickle, so a cloudpickled callable cannot be deserialized there (issue #7267).
Keeping this script to the standard library plus ``fsspec`` lets that interpreter run it directly.

Config arrives as JSON in ``$EVALCHEMY_CLIENT_CONFIG`` (the parent builds it in
:mod:`experiments.evals.evalchemy.serve_and_eval`), so nothing marin-side needs to import here.
Each task runs through the evalchemy fork's ``eval.eval`` once (one invocation per task so each
carries its own ``num_fewshot``) with lm-eval's ``local-completions`` (or ``local-chat-completions``)
API model pointed at the served URL. Its ``results_*.json`` tree is uploaded to ``out_path/<dir>/``
for :class:`~marin.evaluation.eval_result.EvalchemyResult` to read back. ``out_path`` is an
object-store URL the parent resolved under ``marin_prefix()``; for an ``s3://`` destination the pod's
injected ``FSSPEC_S3`` (endpoint + virtual-host addressing) is applied by fsspec automatically.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import fsspec

CONFIG_ENV_KEY = "EVALCHEMY_CLIENT_CONFIG"

# vLLM returns HTTP 400 when prompt_tokens + max_tokens exceeds the served context window. Reserve
# this many tokens for the prompt when shrinking a generation budget to fit a small served context.
_CONTEXT_PROMPT_RESERVE = 1024

# lm-eval truncates a prompt to max_length, but the served backend also counts the requested output
# tokens against its context window (a loglikelihood request adds one output token to a
# max_length-long prompt). Report a context this much below the true served window so prompt +
# output never crosses it; on a large-context model the shave is negligible.
_CONTEXT_MARGIN = 64


def generation_budget(max_gen_toks: int, max_length: int | None) -> int:
    """The per-request generation cap, shrunk to fit a served context smaller than the budget.

    A model whose context is smaller than the suite's generation budget (e.g. a 4k-context model
    under an 8k chat budget) 400s every request unless the requested ``max_tokens`` leaves room for
    the prompt within the context window.
    """
    if max_length is None or max_gen_toks + _CONTEXT_PROMPT_RESERVE <= max_length:
        return max_gen_toks
    return max(256, max_length - _CONTEXT_PROMPT_RESERVE)


def served_max_length(base_url: str) -> int | None:
    """The served model's context length, from the OpenAI ``/models`` card (vLLM reports ``max_model_len``).

    lm-eval's API model cannot see the server's context window and assumes 2048 tokens by default,
    left-truncating longer prompts -- which silently drops few-shot examples on tasks like 25-shot
    arc_challenge. Returns None when the server does not report a length (the lm-eval default stands).
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30) as resp:
            payload = json.load(resp)
    except Exception as exc:
        print(f"could not read {base_url}/models for max_model_len: {exc}", flush=True)
        return None
    for entry in payload.get("data", []):
        if entry.get("max_model_len"):
            return int(entry["max_model_len"])
    return None


def build_model_args(config: dict, use_chat: bool, max_length: int | None) -> str:
    """lm-eval ``--model_args`` for the served OpenAI endpoint (comma-joined ``key=value`` list)."""
    endpoint_path = "chat/completions" if use_chat else "completions"
    args = [
        f"model={config['model_id']}",
        f"base_url={config['base_url'].rstrip('/')}/{endpoint_path}",
        f"tokenizer={config['tokenizer']}",
        "tokenizer_backend=huggingface",
        "tokenized_requests=False",
        f"num_concurrent={config['num_concurrent']}",
        # The TPU vLLM prompt-logprobs path 500s in whole-batch bursts (every in-flight request at
        # once); one request exhausting its retries mid-burst closes lm-eval's shared session and
        # fails the whole task, so give each request enough headroom to ride out a burst.
        "max_retries=8",
        # lm-eval's per-request client timeout defaults to 300s; a long reasoning generation
        # (multi-thousand-token chat benchmark) can exceed that, and a spurious timeout retry-storms
        # the endpoint. 1800s covers a full max_gen_toks generation on a slow serve.
        "timeout=1800",
    ]
    if max_length is not None:
        args.append(f"max_length={max_length}")
    return ",".join(args)


def build_command(config: dict, task: dict, output_path: str, python: str, max_length: int | None) -> list[str]:
    """The ``eval.eval`` argv for one task. ``python`` runs the evalchemy fork + lm-eval in its venv.

    One invocation per task so each carries its own ``num_fewshot`` (lm-eval's ``--num_fewshot`` is a
    single global override). The chat route applies only to generation tasks of a chat-template model:
    loglikelihood (MCQ) tasks always go through the completions API, since chat endpoints cannot echo
    prompt logprobs (lm-eval rejects them with "Loglikelihood is not supported for chat completions").
    """
    # completion_only: code-infilling tasks score a raw continuation, which chat formatting breaks.
    use_chat = config["apply_chat_template"] and task["generation"] and not task["completion_only"]
    model = "local-chat-completions" if use_chat else "local-completions"
    gen_budget = generation_budget(config["max_gen_toks"], max_length)
    if gen_budget != config["max_gen_toks"]:
        print(
            f"clamped max_gen_toks {config['max_gen_toks']} -> {gen_budget} to fit served context {max_length}",
            flush=True,
        )
    cmd = [
        python,
        "-m",
        "eval.eval",
        "--model",
        model,
        "--model_args",
        build_model_args(config, use_chat, max_length),
        "--tasks",
        task["name"],
        "--gen_kwargs",
        f"max_gen_toks={gen_budget}",
        # Chat-native benchmarks (MATH500-style) size their generations from --max_tokens, not
        # gen_kwargs; lm-eval-native tasks ignore it.
        "--max_tokens",
        str(gen_budget),
        "--output_path",
        output_path,
        # Per-question jsonl (doc, prompt, responses, per-sample scores) next to the results JSON;
        # the parent converts each to parquet for drill-down analysis.
        "--log_samples",
        "--verbosity",
        "INFO",
    ]
    # Always pass --num_fewshot, including 0. evalchemy's parser defaults it to None, so omitting the
    # flag lets lm-eval fall back to the task YAML's own default (gsm8k defaults to 5-shot) and a
    # 0-shot request is silently ignored. Chat-native benchmarks record it in their config but do not
    # few-shot on it, so an explicit 0 is harmless there.
    cmd += ["--num_fewshot", str(task["num_fewshot"])]
    if task["unsafe_code"]:
        # code_eval tasks execute model-generated code; lm-eval refuses them without this opt-in.
        cmd.append("--confirm_run_unsafe_code")
    if config["max_eval_instances"] is not None:
        cmd += ["--limit", str(config["max_eval_instances"])]
    if use_chat:
        cmd.append("--apply_chat_template")
    return cmd


def scored_results(local_out: str) -> bool:
    """Whether any ``results_*.json`` under ``local_out`` holds a non-empty ``results`` payload.

    lm-eval exits 0 and writes an empty ``results`` dict when every request to the endpoint failed
    (e.g. the server crashed mid-task), so exit code and file presence alone cannot vouch for a task.
    """
    for dirpath, _, filenames in os.walk(local_out):
        for filename in filenames:
            if not (filename.startswith("results_") and filename.endswith(".json")):
                continue
            with open(os.path.join(dirpath, filename)) as handle:
                if json.load(handle).get("results"):
                    return True
    return False


def main() -> None:
    config = json.loads(os.environ[CONFIG_ENV_KEY])
    tasks = config["tasks"]
    if not tasks:
        raise SystemExit("run_evalchemy_client requires at least one task")

    out_path = config["out_path"].rstrip("/")
    # Raw fsspec, not rigging's StoragePath: the eval image carries fsspec + s3fs/gcsfs, not rigging.
    # For an s3:// destination the pod's injected FSSPEC_S3 (endpoint + virtual-host addressing) is
    # applied by fsspec, so url_to_fs needs no extra config. out_path is region-local (the eval child
    # is pinned to the serve region), so no cross-region copy.
    out_fs, _ = fsspec.core.url_to_fs(out_path)
    served = served_max_length(config["base_url"])
    max_length = served - _CONTEXT_MARGIN if served is not None else None
    print(f"served max_model_len: {served} (lm-eval max_length={max_length})", flush=True)
    failures: list[str] = []
    for task in tasks:
        dest = f"{out_path}/{task['dir']}"
        with tempfile.TemporaryDirectory() as local_out:
            # sys.executable is the evalchemy image's interpreter, so ``-m eval.eval`` resolves the
            # fork + lm-eval baked into its venv.
            cmd = build_command(config, task, local_out, sys.executable, max_length)
            print(f"running evalchemy: {' '.join(cmd)}", flush=True)
            # Upload whatever the task produced before reacting to its exit code, so one task's failure
            # does not discard another task's already-scored output.
            result = subprocess.run(cmd)
            produced = os.listdir(local_out)
            scored = scored_results(local_out)
            if produced:
                out_fs.put(local_out, dest, recursive=True)
                print(f"uploaded {len(produced)} path(s) to {dest}", flush=True)
        if result.returncode != 0:
            failures.append(f"{task['name']}: eval.eval exited {result.returncode}")
        elif not produced:
            failures.append(f"{task['name']}: produced no artifacts")
        elif not scored:
            failures.append(f"{task['name']}: results are empty (every request to the endpoint failed?)")
    print(f"evalchemy client wrote results for {len(tasks)} task(s) to {out_path}", flush=True)
    if failures:
        raise SystemExit("evalchemy task failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
