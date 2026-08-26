# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate configured tasks through an OpenAI-compatible endpoint.

Config arrives as JSON in ``$EVALCHEMY_CLIENT_CONFIG`` (the parent builds it in
:mod:`marin.evaluation.evalchemy.runner`), so nothing else in Marin needs to import here.
Each task runs through the evalchemy fork's ``evalchemy`` CLI once (one invocation per task so each
carries its own ``num_fewshot``) with lm-eval's ``local-completions`` (or ``local-chat-completions``)
API model pointed at the served URL. Its ``results_*.json`` tree is uploaded to ``out_path/<dir>/``
for :class:`~marin.evaluation.evalchemy.result.EvalchemyResult` to read back. ``out_path`` is an
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
from collections.abc import Callable, Sequence
from functools import wraps
from importlib import import_module
from pathlib import Path

import fsspec

CONFIG_ENV_KEY = "EVALCHEMY_CLIENT_CONFIG"

# Run Evalchemy through this file so Marin can normalize the pinned evaluator's outgoing request
# payload before its CLI imports the model registry. Keep the mode private to this subprocess.
_EVALCHEMY_CLI_MODE = "--marin-run-evalchemy"

# OpenAI-compatible completion endpoints accept at most four stop sequences. Pinned lm-eval already
# applies this limit to chat completions, but its LocalCompletionsAPI forwards an arbitrary-length
# task list unchanged. vLLM validates the same four-item contract.
_OPENAI_MAX_STOP_SEQUENCES = 4

# vLLM returns HTTP 400 when prompt_tokens + max_tokens exceeds the served context window. Reserve
# this many tokens for the prompt when shrinking a generation budget to fit a small served context.
_CONTEXT_PROMPT_RESERVE = 1024

# lm-eval truncates a prompt to max_length, but the served backend also counts the requested output
# tokens against its context window (a loglikelihood request adds one output token to a
# max_length-long prompt). Report a context this much below the true served window so prompt +
# output never crosses it; on a large-context model the shave is negligible.
_CONTEXT_MARGIN = 64


def _install_completion_stop_compatibility() -> Callable[[], None]:
    """Fit pinned lm-eval's completion stops to the OpenAI request limit without changing results.

    Evalchemy is an isolated external runtime, so import it only inside the wrapped subprocess. The
    endpoint receives the first four stops, matching lm-eval's chat implementation. If a task has
    more, truncate the returned text at the earliest configured stop and replace lm-eval's cached
    value too. That preserves the task's full visible stop policy even though the endpoint cannot
    represent it in one request.
    """
    completions_module = import_module("lm_eval.models.openai_completions")
    local_completions_api = completions_module.LocalCompletionsAPI
    original_create_payload = local_completions_api._create_payload
    original_generate_until = local_completions_api.generate_until

    @wraps(original_create_payload)
    def create_openai_compatible_payload(self, *args, **kwargs):
        payload = original_create_payload(self, *args, **kwargs)
        stop = payload.get("stop")
        if isinstance(stop, (list, tuple)) and len(stop) > _OPENAI_MAX_STOP_SEQUENCES:
            payload["stop"] = stop[:_OPENAI_MAX_STOP_SEQUENCES]
        return payload

    @wraps(original_generate_until)
    def generate_with_full_stop_semantics(self, requests, *args, **kwargs):
        generations = original_generate_until(self, requests, *args, **kwargs)
        corrected: list[object] = []
        for request, generation in zip(requests, generations, strict=True):
            request_args = getattr(request, "args", ())
            context = request_args[0] if request_args else None
            gen_kwargs = request_args[1] if len(request_args) >= 2 else None
            stops = gen_kwargs.get("until") if isinstance(gen_kwargs, dict) else None
            if not isinstance(generation, str) or not isinstance(stops, (list, tuple)):
                corrected.append(generation)
                continue
            if len(stops) <= _OPENAI_MAX_STOP_SEQUENCES:
                corrected.append(generation)
                continue

            indexes = [
                index for stop in stops if isinstance(stop, str) and stop and (index := generation.find(stop)) >= 0
            ]
            truncated = generation[: min(indexes)] if indexes else generation
            corrected.append(truncated)
            if truncated != generation and context is not None:
                # lm-eval caches inside its original generate_until. Replace that entry so a resumed
                # evaluation sees the same task-visible text as this run.
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), truncated)
        return corrected

    local_completions_api._create_payload = create_openai_compatible_payload
    local_completions_api.generate_until = generate_with_full_stop_semantics

    def restore() -> None:
        local_completions_api.generate_until = original_generate_until
        local_completions_api._create_payload = original_create_payload

    return restore


def run_evalchemy_cli(argv: Sequence[str]) -> None:
    """Run Evalchemy in a fresh process with Marin's OpenAI payload compatibility installed."""
    restore = _install_completion_stop_compatibility()
    original_argv = sys.argv
    try:
        sys.argv = ["evalchemy", *argv]
        import_module("eval.serve_eval.cli").main()
    finally:
        sys.argv = original_argv
        restore()


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
    args: dict[str, object] = {
        "model": config["model_id"],
        "base_url": f"{config['base_url'].rstrip('/')}/{endpoint_path}",
        "tokenizer": config["tokenizer"],
        "tokenizer_backend": "huggingface",
        "tokenized_requests": False,
        "num_concurrent": config["num_concurrent"],
        # The TPU vLLM prompt-logprobs path 500s in whole-batch bursts (every in-flight request at
        # once); one request exhausting its retries mid-burst closes lm-eval's shared session and
        # fails the whole task, so give each request enough headroom to ride out a burst.
        "max_retries": 8,
        # lm-eval's per-request client timeout defaults to 300s; a long reasoning generation
        # (multi-thousand-token chat benchmark) can exceed that, and a spurious timeout retry-storms
        # the endpoint. 1800s covers a full max_gen_toks generation on a slow serve.
        "timeout": 1800,
    }
    args.update(config.get("extra_model_args", {}))
    if max_length is not None:
        args["max_length"] = max_length
    return ",".join(f"{key}={value}" for key, value in args.items())


def build_command(config: dict, task: dict, output_path: str, python: str, max_length: int | None) -> list[str]:
    """The ``evalchemy`` argv for one task. ``python`` identifies the evaluator virtualenv.

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
    # Model-level extra sampler kwargs (skip_special_tokens, repetition_penalty, ...) ride on the same
    # --gen_kwargs list as the generation budget; lm-eval forwards them on both the completions and chat
    # routes (MCQ tasks ignore gen_kwargs). A per-model value overrides the max_gen_toks default only if
    # it keys "max_gen_toks", which the registry does not.
    gen_kwargs = ",".join(
        [f"max_gen_toks={gen_budget}", *(f"{key}={value}" for key, value in config.get("extra_gen_kwargs", {}).items())]
    )
    cmd = [
        python,
        str(Path(__file__).resolve()),
        _EVALCHEMY_CLI_MODE,
        "--model",
        model,
        "--model_args",
        build_model_args(config, use_chat, max_length),
        "--tasks",
        task["name"],
        "--gen_kwargs",
        gen_kwargs,
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
    # Pass every explicit shot count, including 0. A file-backed task may leave the value unset to use
    # the evaluator task's own default; explicit 0 must still override defaults such as gsm8k's 5-shot.
    if task["num_fewshot"] is not None:
        cmd += ["--num_fewshot", str(task["num_fewshot"])]
    if config.get("batch_size") is not None:
        cmd += ["--batch_size", str(config["batch_size"])]
    if config.get("seed") is not None:
        cmd += ["--seed", str(config["seed"])]
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


def upload_task_output(out_fs, local_out: str, dest: str) -> None:
    """Replace ``dest`` with the task tree ``local_out`` holds.

    fsspec ``put(local_out, dest, recursive=True)`` copies the tempdir's *contents* into ``dest`` only
    while ``dest`` does not yet exist; once it does — a retried evaluation reuses the same durable
    ``dest`` — ``put`` nests the tempdir under it as ``dest/tmp<random>/...``, leaving a second complete
    task tree that later reads must deduplicate. Removing ``dest`` first makes every attempt write
    exactly one tree.
    """
    if out_fs.exists(dest):
        out_fs.rm(dest, recursive=True)
    out_fs.put(local_out, dest, recursive=True)


def main() -> None:
    config = json.loads(os.environ[CONFIG_ENV_KEY])
    tasks = config["tasks"]
    if not tasks:
        raise SystemExit("run_evalchemy_client requires at least one task")

    out_path = config["out_path"].rstrip("/")
    # Raw fsspec, not rigging's StoragePath: the uvx environment carries fsspec + s3fs/gcsfs, not rigging.
    # For an s3:// destination the pod's injected FSSPEC_S3 (endpoint + virtual-host addressing) is
    # applied by fsspec, so url_to_fs needs no extra config. out_path is region-local (the eval child
    # is pinned to the serve region), so no cross-region copy.
    out_fs, _ = fsspec.core.url_to_fs(out_path)
    served = served_max_length(config["base_url"])
    available_context = served - _CONTEXT_MARGIN if served is not None else None
    configured_context = config.get("max_length")
    configured_lengths = [value for value in (available_context, configured_context) if value is not None]
    max_length = min(configured_lengths) if configured_lengths else None
    print(f"served max_model_len: {served} (lm-eval max_length={max_length})", flush=True)
    failures: list[str] = []
    for task in tasks:
        dest = f"{out_path}/{task['dir']}"
        with tempfile.TemporaryDirectory() as local_out:
            # Evalchemy is installed beside the uvx environment's interpreter.
            cmd = build_command(config, task, local_out, sys.executable, max_length)
            print(f"running evalchemy: {' '.join(cmd)}", flush=True)
            # Upload whatever the task produced before reacting to its exit code, so one task's failure
            # does not discard another task's already-scored output.
            result = subprocess.run(cmd)
            produced = os.listdir(local_out)
            scored = scored_results(local_out)
            if produced:
                upload_task_output(out_fs, local_out, dest)
                print(f"uploaded {len(produced)} path(s) to {dest}", flush=True)
        if result.returncode != 0:
            failures.append(f"{task['name']}: evalchemy exited {result.returncode}")
        elif not produced:
            failures.append(f"{task['name']}: produced no artifacts")
        elif not scored:
            failures.append(f"{task['name']}: results are empty (every request to the endpoint failed?)")
    print(f"evalchemy client wrote results for {len(tasks)} task(s) to {out_path}", flush=True)
    if failures:
        raise SystemExit("evalchemy task failures: " + "; ".join(failures))


if __name__ == "__main__":
    if sys.argv[1:2] == [_EVALCHEMY_CLI_MODE]:
        run_evalchemy_cli(sys.argv[2:])
    else:
        main()
