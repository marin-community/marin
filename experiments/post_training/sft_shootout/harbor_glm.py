# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Harbor task folders with GLM 5.3 from the internal bulk service as the agent's model.

The GLM relay is reachable only inside its cluster and needs a bearer token, while installed
agents (mini-swe-agent) call the model from inside the Daytona sandbox. This driver therefore
runs a small proxy next to Harbor that:

- forwards ``/v1/*`` to the relay, adding ``GLM_BULK_TOKEN`` so the token never enters a sandbox;
- logs every request's sampling settings exactly as sent, with usage, latency and the full reply
  (reasoning included), to ``requests.jsonl`` (the record of effective generation settings per client);
- serves vLLM's ``/tokenize`` (``count`` and ``max_model_len``) with GLM 5.3's own tokenizer,
  because the relay's router does not, and Harbor's Mini-SWE context limiter requires it.

The proxy is registered as an Iris link endpoint and reached through a capability URL. Harbor runs
from the campaign-pinned isolated project ``config/external/harbor``. Results and the request log
are uploaded to ``--output``.

Submit to the relay's cluster (cw-us-east-02a):

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --enable-extra-resources \
      --cpu 4 --memory 16GB --disk 64GB --job-name glm-harbor-<name> \
      -e GLM_BULK_TOKEN "$(gcloud secrets versions access latest --secret GLM_BULK_TOKEN \
          --project hai-gcp-models)" \
      -e DAYTONA_API_KEY "$(gcloud secrets versions access latest --secret DAYTONA_EVAL_API_KEY \
          --project hai-gcp-models)" \
      -- uv run python experiments/post_training/sft_shootout/harbor_glm.py \
      --agent terminus-2 --tasks experiments/post_training/sft_shootout/dryrun_tasks/tmax \
      --output s3://marin-us-east-02a/marin/experiments/sft-shootout/generation/<name>
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
import yaml
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import EndpointAccess
from marin.inference.dashboard_server import bind_serving_socket
from rigging.connect import capability_path
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from transformers import AutoTokenizer

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url

logger = logging.getLogger(__name__)

GLM_TOKENIZER = "zai-org/GLM-5.3"
GLM_TOKENIZER_REVISION = "ba32f8c19c8738e2c57c71798c601d477a7bf844"
CAPABILITY_ORIGIN = "https://iris-cw-us-east-02a.oa.dev"
HARBOR_PROJECT = "config/external/harbor"
SERVED_MODEL = f"hosted_vllm/{GLM_MODEL}"
UNLOGGED_REQUEST_KEYS = frozenset({"messages", "tools", "prompt"})

# Retry list and timeouts copied from the eval campaign policies
# (experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/harbor-configs).
RETRY = {
    "max_retries": 10,
    "include_exceptions": [
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "LLMRequestTimeoutError",
        "SandboxBuildFailedError",
        "ContainerRegistryThrottledError",
        "AgentSetupTimeoutError",
        "DaytonaSandboxStopError",
        "DaytonaProcessError",
        "VerifierRuntimeError",
    ],
    "exclude_exceptions": [],
    "wait_multiplier": 2.0,
    "min_wait_sec": 1.0,
    "max_wait_sec": 60.0,
}
AGENTS = {
    "oracle": {"name": "oracle"},
    # tb2-recovery.yaml: Terminus-2, 65,536 input / 16,384 output tokens.
    "terminus-2": {
        "name": "terminus-2",
        "override_setup_timeout_sec": 360,
        "override_timeout_sec": 1800,
        "kwargs": {"model_info": {"max_input_tokens": 65536, "max_output_tokens": 16384}},
    },
    # swebench-verified.yaml: mini-swe-agent 2.1.0, 32,768-token total context.
    "mini-swe-agent": {
        "name": "mini-swe-agent",
        "override_setup_timeout_sec": 360,
        "override_timeout_sec": 1800,
        "kwargs": {"version": "2.1.0", "model_info": {"max_input_tokens": 32768, "max_output_tokens": 16384}},
    },
}


class RequestLog:
    """Append-only JSONL log shared by the proxy's request handlers."""

    def __init__(self, path: Path):
        self._file = path.open("a")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()


def template_message(message: dict) -> dict:
    """Parse tool-call arguments from JSON strings to dicts, as vLLM does before rendering."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    parsed = []
    for call in calls:
        function = call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function = {**function, "arguments": json.loads(arguments)}
            except json.JSONDecodeError:
                pass
        parsed.append({**call, "function": function})
    return {**message, "tool_calls": parsed}


def build_proxy(upstream: str, token: str, max_model_len: int, log: RequestLog) -> Starlette:
    tokenizer = AutoTokenizer.from_pretrained(GLM_TOKENIZER, revision=GLM_TOKENIZER_REVISION)
    client = httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=30))
    auth = {"Authorization": f"Bearer {token}"}

    def count(body: dict) -> list[int]:
        rendered = tokenizer.apply_chat_template(
            [template_message(m) for m in body["messages"]],
            tools=body.get("tools") or None,
            add_generation_prompt=body.get("add_generation_prompt", True),
            tokenize=False,
            **(body.get("chat_template_kwargs") or {}),
        )
        return tokenizer.encode(rendered, add_special_tokens=False)

    async def tokenize(request: Request) -> Response:
        body = await request.json()
        tokens = await asyncio.to_thread(count, body)
        return JSONResponse({"count": len(tokens), "max_model_len": max_model_len, "tokens": tokens})

    async def forward(request: Request) -> Response:
        path = request.url.path
        raw = await request.body()
        headers = {"Content-Type": request.headers.get("content-type", "application/json"), **auth}
        if request.method == "GET":
            response = await client.get(f"{upstream}{path.removeprefix('/v1')}", headers=auth)
            return Response(response.content, response.status_code, media_type=response.headers.get("content-type"))
        body = json.loads(raw) if raw else {}
        record = {
            "ts": time.time(),
            "path": path,
            "agent": request.headers.get("user-agent", ""),
            "settings": {k: v for k, v in body.items() if k not in UNLOGGED_REQUEST_KEYS},
            "n_messages": len(body.get("messages", [])),
            "n_tools": len(body.get("tools") or []),
        }
        started = time.monotonic()
        if body.get("stream"):
            upstream_request = client.build_request("POST", f"{upstream}{path.removeprefix('/v1')}", content=raw, headers=headers)
            upstream_response = await client.send(upstream_request, stream=True)

            async def relay():
                try:
                    async for chunk in upstream_response.aiter_raw():
                        yield chunk
                finally:
                    await upstream_response.aclose()
                    log.write({**record, "status": upstream_response.status_code, "latency_s": time.monotonic() - started})

            return StreamingResponse(relay(), upstream_response.status_code, media_type="text/event-stream")
        response = await client.post(f"{upstream}{path.removeprefix('/v1')}", content=raw, headers=headers)
        record.update(status=response.status_code, latency_s=time.monotonic() - started)
        record["glm_headers"] = {k: v for k, v in response.headers.items() if k.startswith("x-glm")}
        if response.status_code == 200:
            payload = response.json()
            record["usage"] = payload.get("usage")
            record["finish_reasons"] = [c.get("finish_reason") for c in payload.get("choices", [])]
            # The full reply, reasoning included: Mini-SWE's trajectory drops the service's `reasoning` field.
            record["response_messages"] = [c.get("message") for c in payload.get("choices", [])]
            record["response_id"] = payload.get("id")
            if "messages" in body:
                record["local_prompt_tokens"] = len(await asyncio.to_thread(count, body))
        else:
            record["error"] = response.text[:2000]
        log.write(record)
        return Response(response.content, response.status_code, media_type=response.headers.get("content-type"))

    return Starlette(
        routes=[
            Route("/tokenize", tokenize, methods=["POST"]),
            Route("/v1/{rest:path}", forward, methods=["GET", "POST"]),
        ]
    )


def serve(app: Starlette, host: str) -> int:
    """Serve the proxy on a free port of ``host`` and return the port."""
    sock = bind_serving_socket(host, 0)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
    deadline = time.monotonic() + 120
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("proxy did not start")
        time.sleep(0.2)
    return sock.getsockname()[1]


def scrub(root: Path, capability: str) -> None:
    """Replace the capability URL (a bearer credential for the proxy) in every uploaded text file."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if capability in text:
            path.write_text(text.replace(capability, "<capability-url>"))


def job_config(args: argparse.Namespace, api_base: str | None, jobs_dir: Path) -> dict:
    agent = json.loads(json.dumps(AGENTS[args.agent]))
    if api_base is None:
        return job_dict(args, agent, jobs_dir)
    agent["model_name"] = SERVED_MODEL
    agent["kwargs"]["api_base"] = api_base
    agent["kwargs"]["model_info"]["max_input_tokens"] = args.context_window or agent["kwargs"]["model_info"]["max_input_tokens"]
    agent["kwargs"]["model_info"].update(input_cost_per_token=0.0, output_cost_per_token=0.0)
    if args.agent == "terminus-2":
        agent["kwargs"]["llm_call_kwargs"] = {"max_tokens": agent["kwargs"]["model_info"]["max_output_tokens"]}
    return job_dict(args, agent, jobs_dir)


def job_dict(args: argparse.Namespace, agent: dict, jobs_dir: Path) -> dict:
    dataset: dict = {"path": str(Path(args.tasks).resolve())}
    if args.task_names:
        dataset["task_names"] = args.task_names
    return {
        "job_name": args.name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": args.n_attempts,
        "n_concurrent_trials": args.n_concurrent,
        "retry": RETRY,
        "environment": {"type": "daytona"},
        "agents": [agent],
        "datasets": [dataset],
        "verifier": {"override_timeout_sec": 600},
    }


def run_harbor(args: argparse.Namespace, config: dict, work: Path) -> subprocess.CompletedProcess:
    config_path = work / "job.yaml"
    config_path.write_text(yaml.safe_dump(config))
    logger.info("Harbor job %s: agent=%s tasks=%s attempts=%d", args.name, args.agent, args.tasks, args.n_attempts)
    env = {k: v for k, v in os.environ.items() if k != GLM_BULK_TOKEN_ENV}
    return subprocess.run(
        ["uv", "run", "--isolated", "--project", HARBOR_PROJECT, "--frozen", "harbor", "run", "-c", str(config_path), "-q"],
        env=env,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--tasks", required=True, help="Local folder of Harbor task folders, or a remote prefix.")
    parser.add_argument("--task-names", nargs="*", help="Restrict to these task folder names.")
    parser.add_argument("--output", required=True, help="Remote prefix for the job folder and request log.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--context-window", type=int, default=None, help="Override the policy's max_input_tokens.")
    parser.add_argument("--relay-job", default=DEFAULT_GLM_RELAY_JOB)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.name = args.name or f"{args.agent}-{uuid.uuid4().hex[:6]}"

    work = Path(tempfile.mkdtemp(prefix="glm-harbor-"))
    if StoragePath(args.tasks).is_remote:
        local_tasks = work / "tasks"
        StoragePath(args.tasks).download_to(str(local_tasks), recursive=True)
        args.tasks = str(local_tasks)

    if args.agent == "oracle":
        # Runs each task's solution/solve.sh; no model, so no proxy. Used for task and visibility probes.
        config = job_config(args, None, work / "jobs")
        result = run_harbor(args, config, work)
        capability = None
    else:
        upstream = resolve_glm_base_url(args.relay_job)
        log = RequestLog(work / "requests.jsonl")
        max_model_len = AGENTS[args.agent]["kwargs"]["model_info"]["max_input_tokens"]
        app = build_proxy(upstream, os.environ[GLM_BULK_TOKEN_ENV], args.context_window or max_model_len, log)
        host = get_job_info().advertise_host
        port = serve(app, host)
        endpoint_name = f"/glm-harbor/{args.name}-{uuid.uuid4().hex[:8]}"
        ctx = iris_ctx()
        with ctx.registry.registered(endpoint_name, f"http://{host}:{port}", {}, access=EndpointAccess.ENDPOINT_ACCESS_LINK):
            minted = ctx.client.mint_endpoint_token(endpoint_name, ttl=Duration.from_hours(24))
            capability = minted.capability_url or f"{CAPABILITY_ORIGIN}{capability_path(endpoint_name, minted.token)}"
            config = job_config(args, f"{capability.rstrip('/')}/v1", work / "jobs")
            result = run_harbor(args, config, work)

    job_dir = work / "jobs" / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "driver_config.json").write_text(json.dumps(config, indent=2))
    if capability is not None:
        scrub(job_dir, capability.rstrip("/"))
    StoragePath(args.output).upload_from(str(job_dir), recursive=True)
    if (work / "requests.jsonl").exists():
        StoragePath(f"{args.output.rstrip('/')}/requests.jsonl").upload_from(str(work / "requests.jsonl"))
    logger.info("Uploaded results to %s (harbor exit %d)", args.output, result.returncode)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
