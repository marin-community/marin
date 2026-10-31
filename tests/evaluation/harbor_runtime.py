# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Harbor boundary tests in its isolated, locked environment."""

import json
import os
import subprocess
import textwrap
from pathlib import Path

from marin.evaluation.harbor.driver_config import _driver_environment

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_RUNTIME_PROJECT = _ROOT / "vendor/harbor"


def external_python(
    *args: str,
    hash_seed: str = "0",
    check: bool = True,
    extra_python_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = _driver_environment()
    environment["PYTHONHASHSEED"] = hash_seed
    if extra_python_path is not None:
        environment["PYTHONPATH"] = os.pathsep.join((str(extra_python_path), environment["PYTHONPATH"]))
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            str(_RUNTIME_PROJECT),
            "--extra",
            "runtime",
            "--no-default-groups",
            "--frozen",
            "python",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if check:
        completed.check_returncode()
    return completed


def write_runtime_overlay(path: Path, tmp_path: Path, **overrides: object) -> None:
    overlay = {
        "job_name": "runtime-job",
        "jobs_dir": str(tmp_path / "jobs"),
        "dataset_path": None,
        "endpoint_url": "https://iris.example/capability/v1",
        "served_model": "served-model",
        "model_identity": "org/original-model",
        "task_limit": 1,
        "model_agent_kwargs": {},
        "verifier_env": {},
        "archive_root": str(tmp_path / "archive"),
        "archive_dataset": "terminal-bench",
    }
    path.write_text(json.dumps({**overlay, **overrides}))


def run_single_turn_aime_agent(
    tmp_path: Path,
    *,
    content: str,
    finish_reason: str | None = None,
    transient_failures: int = 0,
    model_alias: str | None = None,
) -> dict[str, object]:
    """Exercise the agent in Harbor's frozen environment with fake HTTP and sandbox I/O boundaries."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    answer_path = tmp_path / "answer.txt"
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    response_bytes = json.dumps({"choices": [choice]}).encode()
    script = textwrap.dedent(
        f"""
        import asyncio
        import io
        import json
        import subprocess
        import urllib.error
        from types import SimpleNamespace

        from marin.evaluation.harbor import single_turn_aime_agent as agent_module


        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.close()


        class UrlOpen:
            def __init__(self):
                self.attempts = 0
                self.models = []

            def __call__(self, request, timeout):
                self.attempts += 1
                self.models.append(json.loads(request.data)["model"])
                if self.attempts <= {transient_failures}:
                    raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", None, None)
                return Response({response_bytes!r})


        class Environment:
            async def exec(self, command, **kwargs):
                completed = subprocess.run(
                    ["/bin/sh", "-c", command],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return SimpleNamespace(
                    return_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )


        urlopen = UrlOpen()
        agent_module.urllib.request.urlopen = urlopen
        agent = agent_module.SingleTurnAimeAgent(
            logs_dir=agent_module.Path({str(logs_dir)!r}),
            model_name="hosted_vllm/iceball-micro",
            model_alias={model_alias!r},
            api_base="https://inference.example/v1",
            answer_path={str(answer_path)!r},
            max_tokens=256,
            request_retry_initial=0.001,
        )
        async def drive_agent():
            environment = Environment()
            await agent.setup(environment=environment)
            await agent.run(
                instruction="Solve this AIME problem",
                environment=environment,
                context=object(),
            )

        asyncio.run(drive_agent())
        print(json.dumps({{
            "answer": agent_module.Path({str(answer_path)!r}).read_text(),
            "response": agent_module.Path({str(logs_dir / "response.txt")!r}).read_text(),
            "attempts": urlopen.attempts,
            "models": urlopen.models,
        }}))
        """
    )
    return json.loads(external_python("-c", script).stdout)


def preflight(
    tmp_path: Path,
    requests: list[tuple[Path, dict[str, object]]],
    *,
    hash_seed: str = "0",
    check: bool = True,
    extra_python_path: Path | None = None,
    judge_model: str | None = None,
) -> subprocess.CompletedProcess[str]:
    request_path = tmp_path / f"requests-{hash_seed}.json"
    request_path.write_text(
        json.dumps(
            [{"path": str(path), "model_agent_kwargs": kwargs, "judge_model": judge_model} for path, kwargs in requests],
            separators=(",", ":"),
        )
    )
    return external_python(
        str(_DRIVER),
        "preflight",
        str(request_path),
        hash_seed=hash_seed,
        check=check,
        extra_python_path=extra_python_path,
    )
