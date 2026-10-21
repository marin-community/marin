# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Harbor task folders with GLM 5.3 (the internal bulk service) as the model, optionally as the
teacher in a relay with Snowball v0 as the student.

The GLM relay is reachable only inside its cluster and needs a bearer token, while mini-swe-agent
calls the model from inside the Daytona sandbox. The driver therefore runs ``glm_proxy`` next to
Harbor, registers it as an Iris link endpoint, and gives the agents its capability URL. Harbor runs
from the campaign-pinned isolated project ``config/external/harbor`` with agent settings copied
from the campaign policies, through the ``harbor_agents/trial_agents.py`` subclasses that tag every
request with a trial id. With ``--relay`` the proxy also routes to v0 per ``relay.py``; v0 runs as its
own root job (``serve_student.py``, possibly in another cluster) reached through ``STUDENT_BASE_URL``.

Outputs under ``--output``: Harbor's job folder, ``requests.jsonl`` (the canonical trace), and
``driver_config.json``. Submit to the GLM relay's cluster (cw-us-east-02a):

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --enable-extra-resources \
      --cpu 8 --memory 32GB --disk 64GB --job-name glm-harbor-<name> \
      -e GLM_BULK_TOKEN "$(gcloud secrets versions access latest --secret GLM_BULK_TOKEN \
          --project hai-gcp-models)" \
      -e DAYTONA_API_KEY "$(gcloud secrets versions access latest --secret DAYTONA_EVAL_API_KEY \
          --project hai-gcp-models)" \
      -- uv run python experiments/post_training/sft_shootout/harbor_glm.py \
      --agent terminus-2 --family tmax --package-revision <source@revision> \
      --tasks <folder of task folders> --name <name> \
      --output s3://marin-us-east-02a/marin/experiments/sft-shootout/generation/<name>
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import tarfile
import tempfile
import threading
import time
import tomllib
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pyarrow.parquet as pq
import uvicorn
import yaml
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import EndpointAccess
from iris.rpc import job_pb2
from marin.evaluation.harbor.agent_context import (
    MAX_INPUT_TOKENS_KEY,
    MAX_OUTPUT_TOKENS_KEY,
    MODEL_INFO_KEY,
    reconciled_model_info,
    served_model_info,
)
from marin.evaluation.hardware import AcceleratorChoice, Platform, serving_geometry
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.evaluation.serving_config import inference_config_for_model, resolved_serve_config
from marin.inference.config import RemoteInferenceConfig
from marin.inference.dashboard_server import bind_serving_socket
from rigging.connect import capability_path
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from starlette.applications import Starlette

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.sft_shootout.glm_proxy import (
    Backend,
    RelayPolicy,
    RequestLog,
    build_proxy,
    load_tokenizer,
)
from experiments.post_training.sft_shootout.rebuild_traces import final_attempts
from experiments.post_training.sft_shootout.relay import Author
from experiments.post_training.sft_shootout.scrub_run import scrub_text

logger = logging.getLogger(__name__)

GLM_TOKENIZER = "zai-org/GLM-5.3"
GLM_TOKENIZER_REVISION = "ba32f8c19c8738e2c57c71798c601d477a7bf844"
CLUSTER = "cw-us-east-02a"
CAPABILITY_ORIGIN = "https://iris-cw-us-east-02a.oa.dev"
# The only host a sandbox may reach during the agent run: the host of the proxy's minted capability URL
# (checked when the URL is minted); see AgentPhaseNetwork in harbor_agents/trial_agents.py.
AGENT_DOMAIN_ALLOW_LIST = "iris.oa.dev"
HARBOR_PROJECT = "config/external/harbor"
AGENTS_PATH = Path(__file__).parent / "harbor_agents"
MINI_SWE_CONSTRAINTS = AGENTS_PATH / "mini-swe-agent-2.1.0-constraints.txt"
# Relay students: each one's serving config (a regional bucket copy; see its YAML) and the Hub revision the copy
# matches, recorded in driver_config. v0's 4eeb756b has the same weights as the eval config's 271c4b3c; the later
# commit only corrects the model card.
STUDENTS = {
    "v0": (
        Path("experiments/post_training/sft_shootout/students/snowball-v0.yaml"),
        "4eeb756b404db25bc04e4daf49dfb9817bceb741",
    ),
    "v0.1": (
        Path("experiments/post_training/sft_shootout/students/snowball-v0.1.yaml"),
        "6783e9743478bd25a9ce03ab3460e84d6a9814b0",
    ),
}
# The relay student's OpenAI base URL: a capability URL (a bearer credential) for serve_student.py's endpoint,
# passed only through the environment.
STUDENT_URL_ENV = "STUDENT_BASE_URL"
STUDENT_POLL_INTERVAL = 120

# Each family runs under its eval-campaign Harbor policy: retry list, verifier timeout, agent timeouts
# and agent kwargs come from the YAML, and model_info is reconciled the way the eval launcher does it.
POLICY_DIR = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/harbor-configs")
POLICIES = {"tmax": POLICY_DIR / "tb2-recovery.yaml", "swesmith": POLICY_DIR / "swebench-verified.yaml"}
# Generation-only change to the policy: a smaller context window (model_info max_input_tokens, the total window in
# Harbor) so a trajectory that fits it also fits 32,768 tokens in v0's chat template, the SFT row length. In the
# SWE-smith pilot, v0-template / GLM tokens for passing traces was median 1.038, p99 1.073, max 1.089, so
# 30,000 x 1.089 = 32,670. TMax keeps the policy window (the ratio there is not stable). --window-tokens sets a
# different SWE-smith window; above 30,000 most traces that use the window no longer fit a 32,768-token row.
GENERATION_WINDOW = {"swesmith": 30_000}
AGENTS = {
    # Runs each task's solution/solve.sh with no model; used for task and visibility probes.
    "oracle": {"name": "oracle"},
    # Changes nothing, so an intact task must score 0.
    "null": {"import_path": "probe_agents:NullAgent"},
    # mini-swe-agent 2.1.0's real setup, then a listing of its tool env (no model calls).
    "tool-env-probe": {
        "import_path": "probe_agents:ToolEnvProbe",
        "model_name": "hosted_vllm/none",
        "kwargs": {"version": "2.1.0"},
    },
    # A null agent plus reachability checks before, during and after the agent-phase network limit.
    "network-probe": {
        "import_path": "probe_agents:NetworkProbe",
        "kwargs": {"agent_domain_allow_list": AGENT_DOMAIN_ALLOW_LIST},
    },
    # Policy agents, run through subclasses that tag requests with the trial id.
    "terminus-2": {"import_path": "trial_agents:TrialTerminus2"},
    # Runs inside the sandbox, which reaches only the proxy during the run (no github.com/swesmith mirrors).
    "mini-swe-agent": {
        "import_path": "trial_agents:TrialMiniSweAgent",
        "kwargs": {"agent_domain_allow_list": AGENT_DOMAIN_ALLOW_LIST},
    },
}
# Written to --output only after Harbor exits 0 and everything is uploaded.
SUCCESS_MARKER = "_SUCCESS"
# A run fails (no _SUCCESS; its outputs go to <output>.failed-<id>) when more than this share of its trials raised
# without ever getting a 200 reply from the proxy: the proxy or a backend was down, not the agent.
MAX_UNSERVED_SHARE = 0.2
MODEL_FREE_AGENTS = {"oracle", "null", "network-probe", "tool-env-probe"}


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


@contextlib.contextmanager
def capability_endpoint(app: Starlette, name: str) -> Iterator[tuple[str, str]]:
    """Serve ``app`` as an Iris link endpoint; yield its capability URL (a bearer credential) and local address."""
    host = get_job_info().advertise_host
    port = serve(app, host)
    endpoint_name = f"/glm-harbor/{name}-{uuid.uuid4().hex[:8]}"
    ctx = iris_ctx()
    with ctx.registry.registered(endpoint_name, f"http://{host}:{port}", {}, access=EndpointAccess.ENDPOINT_ACCESS_LINK):
        minted = ctx.client.mint_endpoint_token(endpoint_name, ttl=Duration.from_hours(24))
        capability = minted.capability_url or f"{CAPABILITY_ORIGIN}{capability_path(endpoint_name, minted.token)}"
        yield capability.rstrip("/"), f"http://{host}:{port}"


def student_model(student: str) -> ModelConfig:
    return load_model_config(STUDENTS[student][0])


def student_provenance(student: str) -> dict:
    model = student_model(student)
    return {
        "config": str(STUDENTS[student][0]),
        "location": model.location,
        "hub_revision": STUDENTS[student][1],
        "tokenizer": model.tokenizer or model.location,
        "tokenizer_revision": model.effective_tokenizer_revision,
    }


def student_inference_config(student: str, cluster: str) -> RemoteInferenceConfig:
    """The student served exactly as the eval policy does: 8xH100 vLLM in ``cluster`` (see serve_student.py)."""
    model = student_model(student)
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster=cluster)
    accelerator = replace(accelerator, geometry=serving_geometry(resolved_serve_config(model), accelerator))
    # Iris defaults GPU jobs to NCCL_DEBUG=INFO, which floods the log forwarder; this vLLM path gets no
    # NCCL_DEBUG_SUBSYS (no gpu extra), so WARN here is the whole setting.
    return inference_config_for_model(
        model, accelerator, env_vars={"NCCL_DEBUG": "WARN"}, priority=job_pb2.PRIORITY_BAND_INHERIT
    )


def teacher_overrides(reasoning_effort: str | None) -> dict:
    """Fields the proxy adds to every GLM request: the chat template's reasoning effort, when one is chosen."""
    return {} if reasoning_effort is None else {"chat_template_kwargs": {"reasoning_effort": reasoning_effort}}


def served_student(student: str, base_url: str) -> Backend:
    """The student behind ``base_url``, a capability URL for serve_student.py's endpoint, once it answers."""
    base_url = base_url.rstrip("/")
    while True:
        try:
            response = httpx.get(f"{base_url}/models", timeout=30)
            if response.status_code == 200:
                break
            reason = f"HTTP {response.status_code}"
        except httpx.HTTPError as error:
            # The type only: an httpx error message can carry the request URL, which is a credential.
            reason = type(error).__name__
        logger.info("waiting for the student to serve (%s)", reason)
        time.sleep(STUDENT_POLL_INTERVAL)
    (served,) = response.json()["data"]
    model = student_model(student)
    logger.info("the student is serving %s", served["id"])
    return Backend(
        author=Author.STUDENT,
        base_url=base_url,
        model=served["id"],
        tokenizer=load_tokenizer(model.tokenizer or model.location, model.effective_tokenizer_revision),
        headers={},
        # The URL names the endpoint, not a host, so it works again once a preempted v0 re-registers.
        can_move=True,
    )


def extract_packages(parquet: str, names: list[str] | None, dest: Path) -> None:
    """Unpack pool packages (columns ``path``, ``task_binary``: one gzipped task folder per row) into ``dest``."""
    with StoragePath(parquet).open("rb") as stream:
        table = pq.read_table(stream, columns=["path", "task_binary"])
    wanted = set(names) if names else None
    found = set()
    for path, binary in zip(table.column("path").to_pylist(), table.column("task_binary").to_pylist(), strict=True):
        if wanted is not None and path not in wanted:
            continue
        with tarfile.open(fileobj=io.BytesIO(binary), mode="r:gz") as archive:
            archive.extractall(dest / path, filter="data")
        found.add(path)
    if wanted is not None and wanted - found:
        raise ValueError(f"not in {parquet}: {sorted(wanted - found)}")


def scrub_file(path: Path) -> None:
    """Remove Iris capability tokens from one text file (see scrub_run.py)."""
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return
    scrubbed, found = scrub_text(text)
    if found:
        path.write_text(scrubbed)


def scrub(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            scrub_file(path)


def policy(family: str) -> dict:
    return yaml.safe_load(POLICIES[family].read_text())


def policy_agent(args: argparse.Namespace, served_info: dict | None) -> dict:
    """The family policy's agent block, with model_info reconciled as the eval launcher does it."""
    (agent,) = policy(args.family)["agents"]
    if agent["name"] != args.agent:
        raise ValueError(f"the {args.family} policy runs {agent['name']}, not {args.agent}")
    kwargs = {**agent.get("kwargs", {}), **AGENTS[args.agent].get("kwargs", {})}
    kwargs[MODEL_INFO_KEY] = reconciled_model_info(served_info, kwargs.get(MODEL_INFO_KEY))
    if args.window_tokens is not None:
        if served_info is not None and args.window_tokens > served_info[MAX_INPUT_TOKENS_KEY]:
            raise ValueError(
                f"--window-tokens {args.window_tokens} exceeds the student's max_model_len "
                f"{served_info[MAX_INPUT_TOKENS_KEY]}"
            )
        kwargs[MODEL_INFO_KEY][MAX_INPUT_TOKENS_KEY] = args.window_tokens
    if args.agent == "terminus-2":
        kwargs["llm_call_kwargs"] = {
            **kwargs.get("llm_call_kwargs", {}),
            "max_tokens": kwargs[MODEL_INFO_KEY][MAX_OUTPUT_TOKENS_KEY],
        }
    return {**{k: v for k, v in agent.items() if k != "name"}, **AGENTS[args.agent], "kwargs": kwargs}


def in_sandbox(agent: dict) -> bool:
    """Whether the agent runs in the sandbox (it has an agent-phase allow-list) rather than in Harbor's process."""
    return "agent_domain_allow_list" in agent.get("kwargs", {})


def job_config(args: argparse.Namespace, agent: dict, api_base: str | None, jobs_dir: Path) -> dict:
    agent = json.loads(json.dumps(agent))
    if api_base is not None:
        agent["model_name"] = f"hosted_vllm/{GLM_MODEL}"
        agent["kwargs"]["api_base"] = api_base
    dataset: dict = {"path": str(Path(args.tasks).resolve())}
    if args.task_names:
        dataset["task_names"] = args.task_names
    family_policy = policy(args.family)
    return {
        "job_name": args.name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": args.n_attempts,
        "n_concurrent_trials": args.n_concurrent,
        "retry": family_policy["retry"],
        "environment": {"type": "daytona"},
        "agents": [agent],
        "datasets": [dataset],
        "verifier": family_policy["verifier"],
    }


def run_harbor(args: argparse.Namespace, config: dict, work: Path) -> int:
    config_path = work / "job.yaml"
    config_path.write_text(yaml.safe_dump(config))
    logger.info("Harbor job %s: agent=%s tasks=%s attempts=%d", args.name, args.agent, args.tasks, args.n_attempts)
    env = {k: v for k, v in os.environ.items() if k != GLM_BULK_TOKEN_ENV}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(AGENTS_PATH.resolve()), env.get("PYTHONPATH")]))
    command = [
        "uv",
        "run",
        "--isolated",
        "--project",
        HARBOR_PROJECT,
        "--frozen",
        "harbor",
        "run",
        "-c",
        str(config_path),
        "-q",
    ]
    return subprocess.run(command, env=env, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument(
        "--tasks", required=True, help="Folder of Harbor task folders (local or remote), or a pool packages .parquet."
    )
    parser.add_argument("--task-names", nargs="*", help="Restrict to these task folder names.")
    parser.add_argument(
        "--task-list", help="JSON file (local or remote) whose 'tasks' are the task folder names to run."
    )
    parser.add_argument("--output", required=True, help="Remote prefix for the job folder and request log.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--relay", action="store_true", help="Serve v0 as the student; GLM takes over on triggers.")
    parser.add_argument(
        "--budget-fraction", type=float, default=0.75, help="Relay context-budget trigger, as a fraction of the window."
    )
    parser.add_argument(
        "--window-tokens",
        type=int,
        help="Generation window (model_info max_input_tokens) for SWE-smith; default 30,000. TMax keeps its policy's.",
    )
    parser.add_argument("--glm-relay-job", default=DEFAULT_GLM_RELAY_JOB)
    parser.add_argument(
        "--student", choices=sorted(STUDENTS), default="v0", help="Relay student (recorded in every run)."
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high"],
        help="GLM 5.3 template reasoning effort for every teacher request (default: none sent, so the template's Max).",
    )
    parser.add_argument("--skip-if-done", action="store_true", help=f"Exit if --output holds {SUCCESS_MARKER}.")
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument(
        "--package-revision", required=True, help="Source and revision of the task packages, for provenance."
    )
    args = parser.parse_args()
    if args.family not in GENERATION_WINDOW and args.window_tokens is not None:
        parser.error(f"--window-tokens applies only to {sorted(GENERATION_WINDOW)}")
    if args.window_tokens is None:
        args.window_tokens = GENERATION_WINDOW.get(args.family)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request URL at INFO, and the student's URL is a capability URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.skip_if_done and StoragePath(f"{args.output.rstrip('/')}/{SUCCESS_MARKER}").exists():
        logger.info("%s already holds a finished run; skipping", args.output)
        return
    args.task_source = args.tasks
    if args.task_list:
        args.task_names = json.loads(StoragePath(args.task_list).read_text())["tasks"]
    work = Path(tempfile.mkdtemp(prefix="glm-harbor-"))
    if args.tasks.endswith(".parquet"):
        extract_packages(args.tasks, args.task_names, work / "tasks")
        args.tasks = str(work / "tasks")
    elif StoragePath(args.tasks).is_remote:
        local_tasks = work / "tasks"
        StoragePath(args.tasks).download_to(str(local_tasks), recursive=True)
        args.tasks = str(local_tasks)

    capability = None
    with contextlib.ExitStack() as stack:
        if args.agent in MODEL_FREE_AGENTS:
            agent = AGENTS[args.agent]
            config = job_config(args, agent, None, work / "jobs")
        else:
            student = student_model(args.student)
            served_info = (
                served_model_info(student.serve.max_model_len, student.generation.max_gen_toks) if args.relay else None
            )
            agent = policy_agent(args, served_info)
            window = agent["kwargs"][MODEL_INFO_KEY][MAX_INPUT_TOKENS_KEY]
            teacher = Backend(
                author=Author.TEACHER,
                base_url=resolve_glm_base_url(args.glm_relay_job),
                model=GLM_MODEL,
                tokenizer=load_tokenizer(GLM_TOKENIZER, GLM_TOKENIZER_REVISION),
                headers={"Authorization": f"Bearer {os.environ[GLM_BULK_TOKEN_ENV]}"},
                overrides=teacher_overrides(args.reasoning_effort),
            )
            relay = None
            if args.relay:
                relay = RelayPolicy(
                    student=served_student(args.student, os.environ[STUDENT_URL_ENV]),
                    budget_tokens=int(window * args.budget_fraction),
                )
            app = build_proxy(teacher, window, RequestLog(work / "requests.jsonl"), relay)
            capability, local = stack.enter_context(capability_endpoint(app, args.name))
            if urlparse(capability).hostname != AGENT_DOMAIN_ALLOW_LIST:
                raise RuntimeError(
                    f"the capability URL's host is not the agent-phase allow-list {AGENT_DOMAIN_ALLOW_LIST}"
                )
            # Agents in the sandbox (those with an agent-phase allow-list) reach the proxy only through the
            # capability URL. The rest run in this container's Harbor process and use the local address: every
            # gateway round trip costs seconds when cw-us-east-02a endpoints are busy (marin-community/marin#10318).
            api_base = capability if in_sandbox(agent) else local
            config = job_config(args, agent, f"{api_base}/v1", work / "jobs")
        returncode = run_harbor(args, config, work)

    job_dir = work / "jobs" / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "args": vars(args),
        "harbor_source": tomllib.loads((Path(HARBOR_PROJECT) / "pyproject.toml").read_text())["tool"]["uv"]["sources"][
            "harbor"
        ],
        # The shootout base and relay student, recorded for plain runs too.
        "student_model": student_provenance(args.student),
        "harbor": config,
        "policy": str(POLICIES[args.family]),
        "policy_changes": (
            None
            if args.window_tokens is None
            else {
                "model_info.max_input_tokens": {
                    "policy": reconciled_model_info(
                        None, policy(args.family)["agents"][0].get("kwargs", {}).get(MODEL_INFO_KEY)
                    )[MAX_INPUT_TOKENS_KEY],
                    "generation": args.window_tokens,
                    "why": "generation only: 30,000 keeps trajectories within 32,768 tokens in v0's chat template",
                }
            }
        ),
        "teacher": {
            "model": GLM_MODEL,
            "relay_job": args.glm_relay_job,
            "tokenizer": f"{GLM_TOKENIZER}@{GLM_TOKENIZER_REVISION}",
            # Clients send only model and max_tokens (see requests.jsonl), so the service defaults apply. GLM 5.3's
            # chat template renders "Reasoning Effort: Low" or "High" for those values of its reasoning_effort
            # variable and "Max" for anything else or nothing.
            "request_overrides": teacher_overrides(args.reasoning_effort),
            "effective_settings": (
                f"service defaults; template reasoning effort {(args.reasoning_effort or 'max').capitalize()}"
            ),
        },
        "mini_swe_agent_install": (
            None
            if args.agent not in ("mini-swe-agent", "tool-env-probe")
            else {
                "constraints": str(MINI_SWE_CONSTRAINTS),
                "constraints_sha256": hashlib.sha256(MINI_SWE_CONSTRAINTS.read_bytes()).hexdigest(),
                "python": "3.12",
            }
        ),
        "agent_network": {
            "agent_phase_domain_allow_list": agent.get("kwargs", {}).get("agent_domain_allow_list"),
            "setup_and_verifier": "task policy (allow_internet)",
        },
        "capability_url": {
            "agent_calls_proxy_at": "capability URL" if in_sandbox(agent) else "local address (not through the gateway)",
            "in_sandbox": (
                "env only (HOSTED_VLLM_API_BASE, OPENAI_API_BASE)"
                if args.agent == "mini-swe-agent"
                else "not in the sandbox (the agent runs outside it)"
            ),
            "note": (
                "the agent's commands can still read the env, so exposure is reduced, not closed; the guarantee "
                "is the decode-based scrub of every output (scrub_run.py) and the endpoint's lifetime (one per "
                "shard, unregistered when the shard ends)"
            ),
        },
        # Terminus-2 resends earlier reasoning only with interleaved_thinking; the eval policy leaves it off, so the
        # model never sees its earlier reasoning, at generation or at eval. The rows restore it for every turn.
        "earlier_reasoning_resent": (
            None
            if args.agent != "terminus-2"
            else {
                "interleaved_thinking": agent.get("kwargs", {}).get("interleaved_thinking", False),
                "source": (
                    "harbor 016dd78b: src/harbor/agents/terminus_2/terminus_2.py:415 (default False), "
                    "src/harbor/llms/chat.py assistant_message_from_response"
                ),
            }
        ),
        "relay": (
            None
            if not args.relay
            else {
                "student": args.student,
                "student_serving": "serve_student.py root job, reached through its capability URL",
                "budget_fraction": args.budget_fraction,
                "budget_tokens": int(window * args.budget_fraction),
                "budget_counted_with": "student tokenizer",
                "triggers": [
                    "format_error: one teacher repair turn",
                    "context_budget: teacher to the end",
                    "completion_claim: student reply withheld, teacher to the end",
                ],
            }
        ),
    }
    (job_dir / "driver_config.json").write_text(json.dumps(provenance, indent=2))
    # Agents can print the capability URL (e.g. an `env` listing), so it can reach any output, including
    # the request log's observations.
    scrub(job_dir)
    if (work / "requests.jsonl").exists():
        scrub_file(work / "requests.jsonl")
    unserved = unserved_trials(job_dir, work / "requests.jsonl") if capability else None
    output = args.output.rstrip("/")
    if unserved and unserved["share"] > MAX_UNSERVED_SHARE:
        output = f"{output}.failed-{uuid.uuid4().hex[:8]}"
    StoragePath(output).upload_from(str(job_dir), recursive=True)
    if (work / "requests.jsonl").exists():
        StoragePath(f"{output}/requests.jsonl").upload_from(str(work / "requests.jsonl"))
    logger.info("Uploaded results to %s (harbor exit %d, unserved %s)", output, returncode, json.dumps(unserved))
    if returncode != 0:
        raise SystemExit(returncode)
    if output != args.output.rstrip("/"):
        raise SystemExit(
            f"RUN_UNSERVED: {unserved['unserved']} of {unserved['trials']} trials raised with no proxy reply"
        )
    StoragePath(f"{output}/{SUCCESS_MARKER}").write_text("")


def unserved_trials(job_dir: Path, requests: Path) -> dict:
    """Trials whose final Harbor attempt raised and that never got a 200 reply from the proxy."""
    served = set()
    if requests.exists():
        for line in requests.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == 200:
                served.add(record.get("trial"))
    final = final_attempts(StoragePath(str(job_dir)))
    unserved = sum(1 for t in final.values() if t["exception"] and t["trial_id"] not in served)
    return {"trials": len(final), "unserved": unserved, "share": unserved / len(final) if final else 0.0}


if __name__ == "__main__":
    main()
