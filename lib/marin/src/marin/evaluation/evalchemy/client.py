# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate configured tasks through an OpenAI-compatible endpoint.

Config arrives as JSON in ``$EVALCHEMY_CLIENT_CONFIG``; the parent builds it in
:mod:`marin.evaluation.evalchemy.runner`. Each task runs through the evalchemy fork's ``evalchemy``
CLI once (one invocation per task so each carries its own ``num_fewshot``) with lm-eval's
``local-completions`` (or ``local-chat-completions``) API model pointed at the served URL. Evalchemy
writes its aggregate JSON, sample JSONL, and normalized sample rows directly to the FineStore archive
at ``out_path``. The ordinary ``--output_path`` is a temporary directory used for Evalchemy's local
completion check and is discarded after each task.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from importlib.util import find_spec
from pathlib import Path

CONFIG_ENV_KEY = "EVALCHEMY_CLIENT_CONFIG"
EVALCHEMY_RESULTS_PREFIX = "results_"
EVALCHEMY_RESULTS_SUFFIX = ".json"

# Without a configured cap, an lm-eval-native generation task gets the served context minus this
# prompt reserve, so the model config's context window sets its budget the way Evalchemy's own
# derivation does for its chat benchmarks. Few-shot prompts on these tasks (5-shot TriviaQA, 3-shot
# DROP) stay well inside it; lm-eval would otherwise fall back to its 256-token API default.
_NATIVE_PROMPT_RESERVE = 4096

# Smallest budget either context-derived clamp will request.
_MIN_GENERATION_BUDGET = 256

# vLLM returns HTTP 400 when prompt_tokens + max_tokens exceeds the served context window. Reserve
# this many tokens for the prompt when shrinking a generation budget to fit a small served context.
_CONTEXT_PROMPT_RESERVE = 1024

# lm-eval truncates a prompt to max_length, but the served backend also counts the requested output
# tokens against its context window (a loglikelihood request adds one output token to a
# max_length-long prompt). Report a context this much below the true served window so prompt +
# output never crosses it; on a large-context model the shave is negligible.
_CONTEXT_MARGIN = 64


def generation_budget(max_gen_toks: int | None, max_length: int | None) -> int | None:
    """The per-request generation cap, shrunk to fit a served context smaller than the budget.

    A model whose context is smaller than the suite's generation budget (e.g. a 4k-context model
    under an 8k chat budget) 400s every request unless the requested ``max_tokens`` leaves room for
    the prompt within the context window. ``None`` (no configured cap) stays ``None``: Evalchemy then
    derives each benchmark's budget from the served context and its stored longest prompt.
    """
    if max_gen_toks is None or max_length is None or max_gen_toks + _CONTEXT_PROMPT_RESERVE <= max_length:
        return max_gen_toks
    return max(_MIN_GENERATION_BUDGET, max_length - _CONTEXT_PROMPT_RESERVE)


def is_evalchemy_benchmark(task_name: str) -> bool:
    """Whether ``task_name`` is one of Evalchemy's own chat benchmarks rather than an lm-eval task.

    Evalchemy registers each chat benchmark from a directory of the same name under
    ``eval/chat_benchmarks``; everything else on ``--tasks`` resolves through lm-eval's registry.
    """
    spec = find_spec("eval")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("the evalchemy package (`eval`) is not installed beside this client")
    return any((Path(root) / "chat_benchmarks" / task_name).is_dir() for root in spec.submodule_search_locations)


def native_generation_budget(max_length: int | None) -> int | None:
    """The served-context budget for an lm-eval-native generation task with no configured cap."""
    if max_length is None:
        return None
    return max(_MIN_GENERATION_BUDGET, max_length - _NATIVE_PROMPT_RESERVE)


def budget_arguments(config: dict, task: dict, max_length: int | None) -> tuple[list[str], list[str]]:
    """The ``--gen_kwargs`` entries and extra argv that set one task's generation budget.

    A configured cap goes out as both ``max_gen_toks`` (read by lm-eval-native tasks) and
    ``--max_tokens`` (read by Evalchemy's chat benchmarks). Without one, a chat benchmark gets neither
    and sizes its own responses from the served context and its stored longest prompt, while an
    lm-eval-native generation task gets the served context minus a prompt reserve as ``max_gen_toks``.
    """
    gen_budget = generation_budget(config["max_gen_toks"], max_length)
    if gen_budget != config["max_gen_toks"]:
        print(
            f"clamped max_gen_toks {config['max_gen_toks']} -> {gen_budget} to fit served context {max_length}",
            flush=True,
        )
    if gen_budget is not None:
        return [f"max_gen_toks={gen_budget}"], ["--max_tokens", str(gen_budget)]
    if not task["generation"] or is_evalchemy_benchmark(task["name"]):
        return [], []
    native_budget = native_generation_budget(max_length)
    if native_budget is None:
        return [], []
    print(f"native task {task['name']}: max_gen_toks={native_budget} from served context {max_length}", flush=True)
    return [f"max_gen_toks={native_budget}"], []


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
    if use_chat:
        # The endpoint applies its own chat template, so the client needs no tokenizer. Loading one
        # rejects checkpoints whose tokenizer ships custom code (Kimi-Linear) or metadata the
        # client's Transformers cannot parse (Gemma 4). Mirrors marin-community/evalchemy#140.
        tokenizer_args: dict[str, object] = {"tokenizer_backend": "none"}
    else:
        # Loglikelihood scoring needs local token IDs, so load the checkpoint tokenizer and allow
        # its custom code.
        tokenizer_args = {
            "tokenizer": config["tokenizer"],
            "tokenizer_backend": "huggingface",
            "trust_remote_code": True,
        }
    args: dict[str, object] = {
        "model": config["model_id"],
        "base_url": f"{config['base_url'].rstrip('/')}/{endpoint_path}",
        **tokenizer_args,
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
    # Model-level extra sampler kwargs (skip_special_tokens, repetition_penalty, ...) ride on the same
    # --gen_kwargs list as the generation budget; lm-eval forwards them on both the completions and chat
    # routes (MCQ tasks ignore gen_kwargs). A per-model value overrides the max_gen_toks default only if
    # it keys "max_gen_toks", which the registry does not.
    budget_kwargs, budget_args = budget_arguments(config, task, max_length)
    gen_kwargs = budget_kwargs + [f"{key}={value}" for key, value in config.get("extra_gen_kwargs", {}).items()]
    cmd = [
        str(Path(python).with_name("evalchemy")),
        "--model",
        model,
        "--model_args",
        build_model_args(config, use_chat, max_length),
        "--tasks",
        task["name"],
        *(["--gen_kwargs", ",".join(gen_kwargs)] if gen_kwargs else []),
        *budget_args,
        "--output_path",
        output_path,
        # FineStore owns the durable native artifacts and normalized sample rows. Evalchemy keeps
        # its aggregate JSON in this temporary directory only for the completion check below.
        "--log_samples",
        "--finestore_output_path",
        config["out_path"],
        "--finestore_output_prefix",
        task["dir"],
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
            if not (filename.startswith(EVALCHEMY_RESULTS_PREFIX) and filename.endswith(EVALCHEMY_RESULTS_SUFFIX)):
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
    served = served_max_length(config["base_url"])
    available_context = served - _CONTEXT_MARGIN if served is not None else None
    configured_context = config.get("max_length")
    configured_lengths = [value for value in (available_context, configured_context) if value is not None]
    max_length = min(configured_lengths) if configured_lengths else None
    print(f"served max_model_len: {served} (lm-eval max_length={max_length})", flush=True)
    failures: list[str] = []
    for task in tasks:
        with tempfile.TemporaryDirectory() as local_out:
            # Evalchemy is installed beside the uvx environment's interpreter.
            cmd = build_command(config, task, local_out, sys.executable, max_length)
            print(f"running evalchemy: {' '.join(cmd)}", flush=True)
            result = subprocess.run(cmd)
            produced = os.listdir(local_out)
            scored = scored_results(local_out)
        if result.returncode != 0:
            failures.append(f"{task['name']}: evalchemy exited {result.returncode}")
        elif not produced:
            failures.append(f"{task['name']}: produced no artifacts")
        elif not scored:
            failures.append(f"{task['name']}: results are empty (every request to the endpoint failed?)")
    print(f"evalchemy client wrote {len(tasks)} task result(s) to FineStore at {out_path}", flush=True)
    if failures:
        raise SystemExit("evalchemy task failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
