# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Literal

import yaml
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.helmet.config import HelmetEvalName
from marin.evaluation.utils import is_remote_path
from marin.utils import fsspec_copy_path_into_dir, fsspec_exists
from marin.utils import remove_tpu_lockfile_on_exit


def _looks_like_hf_repo_id(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _model_config_for_vllm(*, run_name: str, model_name_or_path: str) -> ModelConfig:
    if _looks_like_hf_repo_id(model_name_or_path):
        return ModelConfig(name=model_name_or_path, path=None, engine_kwargs={})

    if is_remote_path(model_name_or_path):
        return ModelConfig(name=run_name, path=model_name_or_path, engine_kwargs={})

    return ModelConfig(name=model_name_or_path, path=None, engine_kwargs={})


def _sync_local_dir_to_output(local_dir: str, output_path: str) -> None:
    fsspec_copy_path_into_dir(src_path=local_dir, dst_path=output_path)


def _format_disk_usage(path: str) -> str:
    try:
        usage = shutil.disk_usage(path)
    except FileNotFoundError:
        return f"{path}: <missing>"
    except OSError as e:
        return f"{path}: <disk_usage failed: {e}>"

    total_gib = usage.total / (1024**3)
    used_gib = usage.used / (1024**3)
    free_gib = usage.free / (1024**3)
    used_pct = (usage.used / usage.total * 100.0) if usage.total else 0.0
    return f"{path}: total={total_gib:.1f}GiB used={used_gib:.1f}GiB free={free_gib:.1f}GiB used_pct={used_pct:.1f}%"


def _log_local_storage_state(prefix: str, *, tmpdir: str) -> None:
    print(prefix)
    for path in ("/", "/tmp", "/tmp/ray", tmpdir, "/dev/shm"):
        print(f"  {_format_disk_usage(path)}")


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.extend(_iter_strings(key))
            out.extend(_iter_strings(value))
        return out
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            out.extend(_iter_strings(item))
        return out
    return out


_DATA_PATH_RE = re.compile(r"(?:^|/)\.?/data/(?P<rel>.+)$")


def _split_data_relpaths(value: str) -> list[str]:
    # HELMET YAMLs often store file lists as comma-separated strings.
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    return parts or [value]


def _collect_data_relpaths_from_config(cfg_path: str) -> set[str]:
    """Best-effort scan of a HELMET YAML config for `data/...` references.

    HELMET configs typically contain strings like `data/<subdir>/...jsonl`. We scan
    the config structure recursively and extract any suffixes under `data/`.
    """
    with open(cfg_path, "r") as f:
        payload = yaml.safe_load(f)

    relpaths: set[str] = set()
    for value in _iter_strings(payload):
        if not value:
            continue

        # Common forms: "data/...", "./data/...", ".../data/..."
        if value.startswith("data/"):
            for part in _split_data_relpaths(value):
                if part.startswith("data/"):
                    relpaths.add(part.removeprefix("data/"))
            continue
        if value.startswith("./data/"):
            for part in _split_data_relpaths(value):
                if part.startswith("./data/"):
                    relpaths.add(part.removeprefix("./data/"))
            continue

        match = _DATA_PATH_RE.search(value)
        if match:
            for part in _split_data_relpaths(match.group("rel")):
                relpaths.add(part)

    normalized: set[str] = set()
    for rel in relpaths:
        rel = rel.strip().lstrip("/")
        if not rel:
            continue
        # Avoid escaping the data root.
        if rel.startswith(".."):
            continue
        normalized.add(rel)
    return normalized


def _stage_data_paths(
    *,
    remote_data_root: str,
    local_data_root: str,
    relpaths: set[str],
    staged_relpaths: set[str],
) -> None:
    missing = sorted(relpaths - staged_relpaths)
    if not missing:
        return

    print(f"HELMET: staging {len(missing)} data path(s) into {local_data_root}...")
    start = time.time()
    for rel in missing:
        rel = rel.rstrip("/")
        if not rel:
            continue
        remote_path = f"{remote_data_root.rstrip('/')}/{rel}"
        local_dst_dir = os.path.join(local_data_root, os.path.dirname(rel))
        os.makedirs(local_dst_dir, exist_ok=True)
        if not fsspec_exists(remote_path):
            print(f"HELMET: data path missing in dataset (skipping): {remote_path}")
            continue

        print(f"HELMET:   copying {remote_path} -> {local_dst_dir}")
        fsspec_copy_path_into_dir(src_path=remote_path, dst_path=local_dst_dir)
        staged_relpaths.add(rel)

    print(f"HELMET: staging step done in {time.time() - start:.1f}s")


def _collect_data_relpaths_from_eval_py_args(eval_py_args: tuple[str, ...]) -> set[str]:
    relpaths: set[str] = set()
    if not eval_py_args:
        return relpaths

    # We only parse args we care about for staging. These are comma-separated file lists.
    keys = {"--test_files", "--demo_files"}
    it = iter(range(len(eval_py_args)))
    for i in it:
        key = eval_py_args[i]
        if key not in keys:
            continue
        if i + 1 >= len(eval_py_args):
            continue
        value = eval_py_args[i + 1]
        # Skip the next index since we consumed its value.
        next(it, None)
        for part in str(value).split(","):
            part = part.strip()
            if part.startswith("data/"):
                relpaths.add(part.removeprefix("data/"))
            elif part.startswith("./data/"):
                relpaths.add(part.removeprefix("./data/"))
    return relpaths


def _apply_helmet_compat_patches(repo_dir: str) -> None:
    """Apply small, deterministic patches to upstream HELMET to keep our runner working.

    We intentionally keep this narrowly scoped and only patch when we detect known,
    released upstream issues.
    """
    model_utils_path = os.path.join(repo_dir, "model_utils.py")
    if not os.path.exists(model_utils_path):
        return

    with open(model_utils_path, "r") as f:
        src = f.read()

    # HELMET main at least once had an argparse mismatch between `stop_new_line` (arg)
    # and `stop_newline` (usage), which crashes before any evaluation begins.
    needle = "stop_newline=args.stop_newline,"
    replacement = "stop_newline=getattr(args, 'stop_newline', getattr(args, 'stop_new_line', False)),"
    if needle in src and replacement not in src:
        patched = src.replace(needle, replacement)
        with open(model_utils_path, "w") as f:
            f.write(patched)
        print("HELMET: applied compatibility patch for stop_new_line/stop_newline argparse mismatch.")

    # HELMET's TgiVllmModel inherits OpenAIModel, whose `generate()` always uses the
    # chat-completions API. For base models without a tokenizer chat_template, vLLM
    # returns HTTP 400 ("must provide a chat template..."). When `use_chat_template`
    # is False, we can instead use the plain completions endpoint with a string
    # prompt derived from the messages.
    with open(model_utils_path, "r") as f:
        src = f.read()

    # Note: we must check for a `generate()` override specifically inside the
    # TgiVllmModel class, not just anywhere in the file (OpenAIModel defines it).
    if "class TgiVllmModel(OpenAIModel):" in src and "\nclass TgiVllmModel(OpenAIModel):\n    def generate" not in src:
        marker = "    def generate_batch(self, inputs=None, prompt=None, **kwargs):"
        if marker in src:
            generate_impl = (
                "    def generate(self, inputs=None, prompt=None, **kwargs):\n"
                "        if inputs is None:\n"
                "            # For serving models, upstream prepare_inputs returns chat messages.\n"
                "            # Fall back to a raw prompt string if provided.\n"
                "            if prompt is None:\n"
                "                raise ValueError('Either inputs or prompt must be provided')\n"
                "            inputs_text = prompt\n"
                "        else:\n"
                "            # Convert chat messages into a deterministic string prompt.\n"
                "            parts = [f\"Role: {x['role']}\\nContent: {x['content']}\" for x in inputs]\n"
                '            inputs_text = "\\n".join(parts)\n'
                "\n"
                "        if self.use_chat_template:\n"
                "            func = functools.partial(\n"
                "                self.model.chat.completions.create,\n"
                "                model=self.model_name,\n"
                "                messages=inputs,\n"
                "                max_tokens=self.generation_max_length,\n"
                "                temperature=self.temperature if self.do_sample else 0.0,\n"
                "                top_p=self.top_p,\n"
                "                stop=self.stops,\n"
                "                seed=self.seed,\n"
                "                **kwargs,\n"
                "            )\n"
                "            output = call_api(func)\n"
                "            if output is not None:\n"
                "                if output.choices[0].message.content is None:\n"
                "                    return None\n"
                "                return {\n"
                "                    'output': output.choices[0].message.content,\n"
                "                    'input_len': output.usage.prompt_tokens,\n"
                "                    'output_len': output.usage.completion_tokens,\n"
                "                    'input_text': inputs,\n"
                "                }\n"
                "            return None\n"
                "\n"
                "        func = functools.partial(\n"
                "            self.model.completions.create,\n"
                "            model=self.model_name,\n"
                "            prompt=inputs_text,\n"
                "            max_tokens=self.generation_max_length,\n"
                "            temperature=self.temperature if self.do_sample else 0.0,\n"
                "            top_p=self.top_p,\n"
                "            stop=self.stops,\n"
                "            seed=self.seed,\n"
                "            **kwargs,\n"
                "        )\n"
                "        output = call_api(func)\n"
                "        if output is not None:\n"
                "            return {\n"
                "                'output': output.choices[0].text,\n"
                "                'input_len': output.usage.prompt_tokens,\n"
                "                'output_len': output.usage.completion_tokens,\n"
                "                'input_text': inputs_text,\n"
                "            }\n"
                "        return None\n"
                "\n"
            )
            patched = src.replace(marker, generate_impl + marker)
            with open(model_utils_path, "w") as f:
                f.write(patched)
            print("HELMET: applied compatibility patch to use /v1/completions when use_chat_template=False.")

    # Increase OpenAI client timeout for local serving. HELMET uses the OpenAI python
    # client even for vLLM/TGI backends, and the default timeout is often too short
    # for TPU compilation / long-context tasks.
    #
    # We keep this configurable via env since it's used in Ray jobs.
    with open(model_utils_path, "r") as f:
        src = f.read()

    timeout_expr = 'timeout=float(os.environ.get("MARIN_HELMET_API_TIMEOUT_SECS", "300"))'
    if (("OpenAI(" in src or "AsyncOpenAI(" in src) and timeout_expr not in src) or (
        "MARIN_HELMET_API_TIMEOUT_SECS" in src and timeout_expr not in src
    ):
        # Ensure `os` is imported if we use `os.environ` in the injected timeout.
        if "import os" not in src and "from os " not in src:
            lines = src.splitlines()
            insert_at = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Skip module docstring (best effort).
                    # Find the closing delimiter.
                    delim = stripped[:3]
                    for j in range(i + 1, len(lines)):
                        if delim in lines[j]:
                            insert_at = j + 1
                            break
                    continue
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_at = i
                    break
            lines.insert(insert_at, "import os")
            src = "\n".join(lines) + ("\n" if src.endswith("\n") else "")

        def _inject_timeout(call_name: str, text: str) -> str:
            needle = f"{call_name}("
            out = []
            i = 0
            while True:
                j = text.find(needle, i)
                if j == -1:
                    out.append(text[i:])
                    break
                out.append(text[i : j + len(needle)])
                # Heuristic: if this call already sets timeout soon after, don't inject.
                lookahead = text[j : j + 500]
                if "timeout=" in lookahead:
                    i = j + len(needle)
                    continue
                out.append(f"{timeout_expr}, ")
                i = j + len(needle)
            return "".join(out)

        patched = _inject_timeout("OpenAI", src)
        patched = _inject_timeout("AsyncOpenAI", patched)
        if patched != src:
            with open(model_utils_path, "w") as f:
                f.write(patched)
            print(
                "HELMET: increased OpenAI client timeout (set MARIN_HELMET_API_TIMEOUT_SECS to override; "
                "default=300s)."
            )

    # Improve diagnostics around API failures/timeouts: log relevant request metadata (prompt/messages size,
    # max_tokens, model) and raise with the last exception instead of returning None. Returning None makes
    # failures look like "unknown error" and later crashes obscure the root cause.
    with open(model_utils_path, "r") as f:
        src = f.read()

    if "def call_api(" in src and "MARIN_HELMET_CALL_API_DEBUG" not in src:
        lines = src.splitlines(keepends=True)
        start_idx: int | None = None
        for i, line in enumerate(lines):
            if line.startswith("def call_api("):
                start_idx = i
                break
        if start_idx is not None:
            end_idx = len(lines)
            for j in range(start_idx + 1, len(lines)):
                if lines[j].startswith("def ") and not lines[j].startswith("def call_api("):
                    end_idx = j
                    break

            fn_lines = lines[start_idx:end_idx]
            # Insert state capture right after the def line.
            if len(fn_lines) >= 2 and "last_exception" not in "".join(fn_lines[:3]):
                fn_lines.insert(
                    1,
                    (
                        "    # MARIN_HELMET_CALL_API_DEBUG: capture last exception + log request context for timeouts.\n"
                        "    last_exception = None\n"
                    ),
                )

            patched_fn_lines: list[str] = []
            inserted_except = False
            for line in fn_lines:
                patched_fn_lines.append(line)
                stripped = line.lstrip()
                if stripped.startswith("except ") and " as e" in stripped and stripped.rstrip().endswith(":"):
                    indent = line[: len(line) - len(stripped)]
                    patched_fn_lines.append(f"{indent}    last_exception = e\n")
                    inserted_except = True

            # Replace the final return None with a raise so the caller sees the root cause.
            for k in range(len(patched_fn_lines) - 1, -1, -1):
                if patched_fn_lines[k].strip() == "return None":
                    patched_fn_lines[k] = (
                        "    raise RuntimeError(\n"
                        '        "API call failed after retries; last_exception="\n'
                        "        + repr(last_exception)\n"
                        "    ) from last_exception\n"
                    )
                    break

            # Add one-time request context logging near the start of the function body (best-effort, no try/except).
            # We log only cheap derived info to avoid huge logs.
            context_snippet = [
                '    kw = getattr(func, "keywords", None) or {}\n',
                '    prompt = kw.get("prompt")\n',
                '    messages = kw.get("messages")\n',
                "    if prompt is not None:\n",
                "        prompt_str = str(prompt)\n",
                "        logger.info(\n",
                "            (\n",
                '                "MARIN call_api context: "\n',
                "                f\"model={kw.get('model')!r} \"\n",
                '                f"prompt_chars={len(prompt_str)} "\n',
                "                f\"max_tokens={kw.get('max_tokens')!r}\"\n",
                "            )\n",
                "        )\n",
                "    elif isinstance(messages, list):\n",
                "        msg_chars = 0\n",
                "        for msg in messages:\n",
                "            if isinstance(msg, dict):\n",
                "                msg_chars += len(str(msg.get('content', '')))\n",
                "        logger.info(\n",
                "            (\n",
                '                "MARIN call_api context: "\n',
                "                f\"model={kw.get('model')!r} \"\n",
                '                f"messages={len(messages)} "\n',
                '                f"message_chars={msg_chars} "\n',
                "                f\"max_tokens={kw.get('max_tokens')!r}\"\n",
                "            )\n",
                "        )\n",
            ]

            # Insert after our injected header (line 1) if present, else after def line.
            insert_at = 2 if len(patched_fn_lines) >= 2 and "MARIN_HELMET_CALL_API_DEBUG" in patched_fn_lines[1] else 1
            # Avoid double-inserting if call_api already logs this context.
            if "MARIN call_api context:" not in "".join(patched_fn_lines):
                patched_fn_lines[insert_at:insert_at] = context_snippet

            if inserted_except:
                lines[start_idx:end_idx] = patched_fn_lines
                patched_src = "".join(lines)
                with open(model_utils_path, "w") as f:
                    f.write(patched_src)
                print("HELMET: patched call_api to log request context and raise with last exception.")

    eval_py_path = os.path.join(repo_dir, "eval.py")
    if os.path.exists(eval_py_path):
        with open(eval_py_path, "r") as f:
            eval_src = f.read()

        # Fail-fast on generation failures rather than silently skipping examples and writing partial outputs.
        # This avoids "successful" runs with silently corrupted/incomplete results.
        none_needle = (
            "        if output is None:\n"
            '            logger.info(f"skipping example {idx+1} because the model returned None")\n'
            "            continue\n"
        )
        none_replacement = (
            "        if output is None:\n"
            "            raise RuntimeError(\n"
            '                f"Model returned None output at example {idx+1}; aborting to avoid partial results. "\n'
            '                f"(dataset={dataset}, test_file={test_file})"\n'
            "            )\n"
        )
        patched_eval = eval_src
        if none_needle in patched_eval and none_replacement not in patched_eval:
            patched_eval = patched_eval.replace(none_needle, none_replacement)
            print("HELMET: applied strictness patch (abort on None outputs).")

        # Fail-fast on per-dataset exceptions (HELMET upstream logs and continues by default).
        # Our executor relies on failures to be explicit so retries don't silently mask issues.
        exc_needle = (
            "        except Exception as e:\n"
            "            # in case we run into some kind of error\n"
            "            logger.exception(e)\n"
            '            logger.error(f"Error in {dataset}, continuing...")\n'
            "            if args.debug:\n"
            "                raise e\n"
        )
        exc_replacement = (
            "        except Exception as e:\n"
            "            # in case we run into some kind of error\n"
            "            logger.exception(e)\n"
            "            raise\n"
        )
        if exc_needle in patched_eval and exc_replacement not in patched_eval:
            patched_eval = patched_eval.replace(exc_needle, exc_replacement)
            print("HELMET: applied strictness patch (abort on dataset errors).")

        if patched_eval != eval_src:
            with open(eval_py_path, "w") as f:
                f.write(patched_eval)


def _run_cmd(cmd: list[str], *, cwd: str | None = None, desc: str | None = None) -> None:
    if desc:
        print(desc)
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {cmd}\n"
            f"cwd={cwd!r}\n"
            f"--- stdout ---\n{stdout[-8000:]}\n"
            f"--- stderr ---\n{stderr[-8000:]}"
        )


def _fetch_helmet_repo_at_sha(*, repo_url: str, repo_sha: str, repo_dir: str, max_attempts: int = 3) -> None:
    """Fetch HELMET at an exact SHA with minimal disk usage.

    Avoids `git clone` (which can fail/flap on busy clusters) by doing a shallow fetch
    of the specific commit.
    """
    os.makedirs(repo_dir, exist_ok=True)
    _run_cmd(["git", "init"], cwd=repo_dir, desc=f"HELMET: initializing git repo in {repo_dir}...")
    _run_cmd(["git", "remote", "add", "origin", repo_url], cwd=repo_dir, desc=f"HELMET: adding remote {repo_url}...")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"HELMET: fetching commit {repo_sha} (attempt {attempt}/{max_attempts})...")
            _run_cmd(["git", "fetch", "--depth", "1", "origin", repo_sha], cwd=repo_dir)
            _run_cmd(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir, desc=f"HELMET: checking out {repo_sha}...")
            return
        except Exception as e:
            last_error = e
            sleep_s = min(30, 2**attempt)
            print(f"HELMET: git fetch failed (attempt {attempt}/{max_attempts}): {e}")
            print(f"HELMET: retrying after {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch HELMET at {repo_sha} after {max_attempts} attempts.") from last_error


@dataclass(frozen=True)
class HelmetRunConfig:
    run_name: str
    model_name_or_path: str

    helmet_repo_url: str
    helmet_repo_sha: str

    helmet_data_output_path: str
    """Executor-resolved output path of the data step (may be remote)."""

    evals: tuple[HelmetEvalName, ...]
    config_variant: Literal["full", "short"] = "full"

    use_chat_template: bool = False
    seed: int = 42
    tag: str = "v1"

    output_path: str = field(default_factory=str)
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)

    vllm_serve_args: tuple[str, ...] = ()
    eval_py_args: tuple[str, ...] = ()


def run_helmet(config: HelmetRunConfig) -> None:
    """Executor step entrypoint: launch a single TPU job to run 1+ HELMET configs."""

    def _run():
        with remove_tpu_lockfile_on_exit():
            _run_on_tpu(config)

    env_vars: dict[str, str] = {
        # Ensure HELMET uses the Docker sidecar path (no `vllm-tpu` install needed).
        "MARIN_VLLM_MODE": "docker",
    }
    docker_image = os.environ.get("MARIN_VLLM_DOCKER_IMAGE")
    if docker_image:
        env_vars["MARIN_VLLM_DOCKER_IMAGE"] = docker_image

    job_request = JobRequest(
        name=f"helmet:{config.run_name}",
        entrypoint=Entrypoint.from_callable(_run),
        resources=config.resource_config,
        environment=EnvironmentConfig.create(
            extras=["eval", "tpu", "helmet"],
            env_vars=env_vars,
        ),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)


def _run_on_tpu(config: HelmetRunConfig) -> None:
    start = time.time()

    success_path = f"{config.helmet_data_output_path.rstrip('/')}/_SUCCESS"
    if not fsspec_exists(success_path):
        raise RuntimeError(f"HELMET data directory is not ready: missing {success_path}")

    with tempfile.TemporaryDirectory(prefix="helmet_repo_") as tmpdir:
        _log_local_storage_state("HELMET: starting TPU job; initial local storage state:", tmpdir=tmpdir)

        repo_dir = os.path.join(tmpdir, "HELMET")
        _fetch_helmet_repo_at_sha(
            repo_url=config.helmet_repo_url,
            repo_sha=config.helmet_repo_sha,
            repo_dir=repo_dir,
        )

        _apply_helmet_compat_patches(repo_dir)

        local_data_root = os.path.join(tmpdir, "helmet-data")
        os.makedirs(local_data_root, exist_ok=True)
        remote_data_root = f"{config.helmet_data_output_path.rstrip('/')}/data"
        staged_relpaths: set[str] = set()

        data_src = local_data_root
        data_dst = os.path.join(repo_dir, "data")
        if os.path.lexists(data_dst):
            if os.path.islink(data_dst) or os.path.isfile(data_dst):
                os.remove(data_dst)
            else:
                shutil.rmtree(data_dst)
        os.symlink(data_src, data_dst)

        model = _model_config_for_vllm(run_name=config.run_name, model_name_or_path=config.model_name_or_path)
        model_name_or_path = VllmTpuEvaluator.download_model(model)

        print("HELMET: starting vLLM server...")
        vllm_server = VllmTpuEvaluator.start_vllm_server_in_background(
            model=model,
            host="127.0.0.1",
            port=None,
            timeout_seconds=3600,
            extra_args=list(config.vllm_serve_args) if config.vllm_serve_args else None,
        )
        endpoint_url = f"{vllm_server.server_url}/"
        print(f"HELMET: vLLM ready at {endpoint_url}")

        local_output_dir = os.path.join(tmpdir, "output")

        def config_path(eval_name: HelmetEvalName) -> str:
            suffix = "" if config.config_variant == "full" else "_short"
            return os.path.join(repo_dir, "configs", f"{eval_name}{suffix}.yaml")

        ran = []
        try:
            for eval_name in config.evals:
                print(f"HELMET: running {eval_name}...")
                cfg = config_path(eval_name)
                if not os.path.exists(cfg):
                    raise FileNotFoundError(f"Missing HELMET config: {cfg}")

                relpaths = _collect_data_relpaths_from_config(cfg) | _collect_data_relpaths_from_eval_py_args(
                    config.eval_py_args
                )
                if relpaths:
                    _log_local_storage_state("HELMET: before per-config data staging:", tmpdir=tmpdir)
                    _stage_data_paths(
                        remote_data_root=remote_data_root,
                        local_data_root=local_data_root,
                        relpaths=relpaths,
                        staged_relpaths=staged_relpaths,
                    )
                    _log_local_storage_state("HELMET: after per-config data staging:", tmpdir=tmpdir)
                else:
                    print(
                        f"HELMET: no explicit data/ paths found in {cfg}; "
                        "running without additional staging (this may fail if the config loads local data files)."
                    )

                subprocess.run(
                    [
                        sys.executable,
                        "eval.py",
                        "--config",
                        cfg,
                        "--seed",
                        str(config.seed),
                        "--output_dir",
                        local_output_dir,
                        "--tag",
                        config.tag,
                        "--model_name_or_path",
                        model_name_or_path,
                        "--use_chat_template",
                        str(bool(config.use_chat_template)),
                        "--use_vllm_serving",
                        "--endpoint_url",
                        endpoint_url,
                        "--api_key",
                        "EMPTY",
                        *config.eval_py_args,
                        "--overwrite",
                        "--no_cuda",
                    ],
                    check=True,
                    cwd=repo_dir,
                )
                ran.append(eval_name)
        finally:
            VllmTpuEvaluator.cleanup(model, vllm_server=vllm_server)

        os.makedirs(local_output_dir, exist_ok=True)
        with open(os.path.join(local_output_dir, "marin_metadata.json"), "w") as f:
            json.dump(
                {
                    "run_name": config.run_name,
                    "model_name_or_path": config.model_name_or_path,
                    "helmet_repo_url": config.helmet_repo_url,
                    "helmet_repo_sha": config.helmet_repo_sha,
                    "helmet_data_output_path": config.helmet_data_output_path,
                    "evals": list(ran),
                    "config_variant": config.config_variant,
                    "use_chat_template": config.use_chat_template,
                    "eval_py_args": list(config.eval_py_args),
                    "seed": config.seed,
                    "tag": config.tag,
                    "wall_time_seconds": time.time() - start,
                },
                f,
                indent=2,
            )

        _sync_local_dir_to_output(local_output_dir, config.output_path)
