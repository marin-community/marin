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
        "--verbosity",
        "INFO",
    ]
    if task["num_fewshot"]:
        cmd += ["--num_fewshot", str(task["num_fewshot"])]
    if config["max_eval_instances"] is not None:
        cmd += ["--limit", str(config["max_eval_instances"])]
    if config["apply_chat_template"]:
        cmd.append("--apply_chat_template")
    return cmd


def main() -> None:
    config = json.loads(os.environ[CONFIG_ENV_KEY])
    tasks = config["tasks"]
    if not tasks:
        raise SystemExit("run_evalchemy_client requires at least one task")

    out_path = config["out_path"].rstrip("/")
    # Raw fsspec, not rigging's StoragePath: the eval image carries only fsspec/gcsfs, not rigging.
    # out_path is region-local (the eval child is pinned to the serve region), so no cross-region copy.
    out_fs, _ = fsspec.core.url_to_fs(out_path)
    for task in tasks:
        with tempfile.TemporaryDirectory() as local_out:
            # sys.executable is the evalchemy image's interpreter, so ``-m eval.eval`` resolves the
            # fork + lm-eval baked into its venv.
            cmd = build_command(config, task, local_out, sys.executable)
            print(f"running evalchemy: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
            out_fs.put(local_out, f"{out_path}/{task['dir']}", recursive=True)
    print(f"evalchemy client wrote results for {len(tasks)} task(s) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
