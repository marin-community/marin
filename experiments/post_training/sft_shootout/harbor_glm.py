# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Harbor task folders with GLM 5.3 (the internal bulk service) as the model, optionally as the
teacher in a relay with Snowball v0 as the student.

The GLM relay is reachable only inside its cluster and needs a bearer token, while mini-swe-agent
calls the model from inside the Daytona sandbox. The driver therefore runs ``glm_proxy`` next to
Harbor, registers it as an Iris link endpoint, and gives the agents its capability URL. Harbor runs
from the campaign-pinned isolated project ``config/external/harbor`` with agent settings copied
from the campaign policies, through the ``harbor_agents/trial_agents.py`` subclasses that tag every
request with a trial id. With ``--relay`` the driver also serves v0 with the policy's serving config
(an 8xH100 vLLM job in this cluster) and the proxy routes per ``relay.py``.

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
import uuid
from collections.abc import Iterator
from urllib.parse import urlparse
from dataclasses import replace
from pathlib import Path

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
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model, resolved_serve_config
from marin.inference.dashboard_server import bind_serving_socket
from marin.inference.iris import remote_inference
from rigging.connect import capability_path
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from starlette.applications import Starlette

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.sft_shootout.glm_proxy import Backend, RelayPolicy, RequestLog, build_proxy, load_tokenizer
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
# v0: the policy's serving config for Snowball v0, at the revision pinned for the sweep (4eeb756b has
# the same weights as the config's 271c4b3c; the later commit only corrects the model card).
STUDENT_CONFIG = Path(
    "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/"
    "open-athena-Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12.yaml"
)
STUDENT_REVISION = "4eeb756b404db25bc04e4daf49dfb9817bceb741"

# Each family runs under its eval-campaign Harbor policy: retry list, verifier timeout, agent timeouts
# and agent kwargs come from the YAML, and model_info is reconciled the way the eval launcher does it.
POLICY_DIR = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/harbor-configs")
POLICIES = {"tmax": POLICY_DIR / "tb2-recovery.yaml", "swesmith": POLICY_DIR / "swebench-verified.yaml"}
# Generation-only change to the policy: a smaller context window (model_info max_input_tokens, the total window in
# Harbor) so a trajectory that fits it also fits 32,768 tokens in v0's chat template, the SFT row length. In the
# SWE-smith pilot, v0-template / GLM tokens for passing traces was median 1.038, p99 1.073, max 1.089, so
# 30,000 x 1.089 = 32,670. TMax keeps the policy window (the ratio there is not stable).
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
def capability_endpoint(app: Starlette, name: str) -> Iterator[str]:
    """Serve ``app`` as an Iris link endpoint and yield its capability URL (a bearer credential)."""
    host = get_job_info().advertise_host
    port = serve(app, host)
    endpoint_name = f"/glm-harbor/{name}-{uuid.uuid4().hex[:8]}"
    ctx = iris_ctx()
    with ctx.registry.registered(endpoint_name, f"http://{host}:{port}", {}, access=EndpointAccess.ENDPOINT_ACCESS_LINK):
        minted = ctx.client.mint_endpoint_token(endpoint_name, ttl=Duration.from_hours(24))
        yield (minted.capability_url or f"{CAPABILITY_ORIGIN}{capability_path(endpoint_name, minted.token)}").rstrip("/")


@contextlib.contextmanager
def served_student() -> Iterator[Backend]:
    """Serve v0 exactly as the eval policy does (8xH100 vLLM in this cluster) for the run's lifetime."""
    model = replace(load_model_config(STUDENT_CONFIG), revision=STUDENT_REVISION)
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster=CLUSTER)
    accelerator = replace(accelerator, geometry=serving_geometry(resolved_serve_config(model), accelerator))
    config = inference_config_for_model(model, accelerator, env_vars={}, priority=job_pb2.PRIORITY_BAND_INHERIT)
    with remote_inference(config) as session:
        session.wait_until_ready()
        endpoint = session.model.endpoint
        logger.info("v0 is serving %s at %s", endpoint.model, endpoint.base_url)
        yield Backend(
            author=Author.STUDENT,
            base_url=endpoint.base_url.rstrip("/"),
            model=endpoint.model,
            tokenizer=load_tokenizer(model.tokenizer or model.location, STUDENT_REVISION),
            headers={},
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
    if args.family in GENERATION_WINDOW:
        kwargs[MODEL_INFO_KEY][MAX_INPUT_TOKENS_KEY] = GENERATION_WINDOW[args.family]
    if args.agent == "terminus-2":
        kwargs["llm_call_kwargs"] = {**kwargs.get("llm_call_kwargs", {}), "max_tokens": kwargs[MODEL_INFO_KEY][MAX_OUTPUT_TOKENS_KEY]}
    return {**{k: v for k, v in agent.items() if k != "name"}, **AGENTS[args.agent], "kwargs": kwargs}


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
    command = ["uv", "run", "--isolated", "--project", HARBOR_PROJECT, "--frozen", "harbor", "run", "-c", str(config_path), "-q"]
    return subprocess.run(command, env=env, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument(
        "--tasks", required=True, help="Folder of Harbor task folders (local or remote), or a pool packages .parquet."
    )
    parser.add_argument("--task-names", nargs="*", help="Restrict to these task folder names.")
    parser.add_argument("--task-list", help="JSON file (local or remote) whose 'tasks' are the task folder names to run.")
    parser.add_argument("--output", required=True, help="Remote prefix for the job folder and request log.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--relay", action="store_true", help="Serve v0 as the student; GLM takes over on triggers.")
    parser.add_argument(
        "--budget-fraction", type=float, default=0.75, help="Relay context-budget trigger, as a fraction of the window."
    )
    parser.add_argument("--glm-relay-job", default=DEFAULT_GLM_RELAY_JOB)
    parser.add_argument("--skip-if-done", action="store_true", help=f"Exit if --output holds {SUCCESS_MARKER}.")
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument("--package-revision", required=True, help="Source and revision of the task packages, for provenance.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
            student_model = load_model_config(STUDENT_CONFIG)
            served_info = (
                served_model_info(student_model.serve.max_model_len, student_model.generation.max_gen_toks)
                if args.relay
                else None
            )
            agent = policy_agent(args, served_info)
            window = agent["kwargs"][MODEL_INFO_KEY][MAX_INPUT_TOKENS_KEY]
            teacher = Backend(
                author=Author.TEACHER,
                base_url=resolve_glm_base_url(args.glm_relay_job),
                model=GLM_MODEL,
                tokenizer=load_tokenizer(GLM_TOKENIZER, GLM_TOKENIZER_REVISION),
                headers={"Authorization": f"Bearer {os.environ[GLM_BULK_TOKEN_ENV]}"},
            )
            relay = None
            if args.relay:
                relay = RelayPolicy(student=stack.enter_context(served_student()), budget_tokens=int(window * args.budget_fraction))
            app = build_proxy(teacher, window, RequestLog(work / "requests.jsonl"), relay)
            capability = stack.enter_context(capability_endpoint(app, args.name))
            if urlparse(capability).hostname != AGENT_DOMAIN_ALLOW_LIST:
                raise RuntimeError(f"the capability URL's host is not the agent-phase allow-list {AGENT_DOMAIN_ALLOW_LIST}")
            config = job_config(args, agent, f"{capability}/v1", work / "jobs")
        returncode = run_harbor(args, config, work)

    job_dir = work / "jobs" / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "args": vars(args),
        "harbor": config,
        "policy": str(POLICIES[args.family]),
        "policy_changes": None
        if args.family not in GENERATION_WINDOW
        else {
            "model_info.max_input_tokens": {
                "policy": reconciled_model_info(
                    None, policy(args.family)["agents"][0].get("kwargs", {}).get(MODEL_INFO_KEY)
                )[MAX_INPUT_TOKENS_KEY],
                "generation": GENERATION_WINDOW[args.family],
                "why": "generation only: keep trajectories within 32,768 tokens in v0's chat template",
            }
        },
        "teacher": {
            "model": GLM_MODEL,
            "relay_job": args.glm_relay_job,
            "tokenizer": f"{GLM_TOKENIZER}@{GLM_TOKENIZER_REVISION}",
            # Clients send only model and max_tokens (see requests.jsonl), so the service defaults apply;
            # GLM 5.3's chat template then renders "Reasoning Effort: Max".
            "effective_settings": "service defaults; template reasoning effort Max",
        },
        "mini_swe_agent_install": None
        if args.agent not in ("mini-swe-agent", "tool-env-probe")
        else {
            "constraints": str(MINI_SWE_CONSTRAINTS),
            "constraints_sha256": hashlib.sha256(MINI_SWE_CONSTRAINTS.read_bytes()).hexdigest(),
            "python": "3.12",
        },
        "agent_network": {
            "agent_phase_domain_allow_list": agent.get("kwargs", {}).get("agent_domain_allow_list"),
            "setup_and_verifier": "task policy (allow_internet)",
        },
        # Terminus-2 resends earlier reasoning only with interleaved_thinking; the eval policy leaves it off, so the
        # model never sees its earlier reasoning, at generation or at eval. The rows restore it for every turn.
        "earlier_reasoning_resent": None
        if args.agent != "terminus-2"
        else {
            "interleaved_thinking": agent.get("kwargs", {}).get("interleaved_thinking", False),
            "source": "harbor 016dd78b: src/harbor/agents/terminus_2/terminus_2.py:415 (default False), "
            "src/harbor/llms/chat.py assistant_message_from_response",
        },
        "relay": None
        if not args.relay
        else {
            "student": f"{STUDENT_CONFIG}@{STUDENT_REVISION}",
            "budget_fraction": args.budget_fraction,
            "budget_tokens": int(window * args.budget_fraction),
            "budget_counted_with": "student tokenizer",
            "triggers": ["format_error: one teacher repair turn", "context_budget: teacher to the end", "completion_claim: student reply withheld, teacher to the end"],
        },
    }
    (job_dir / "driver_config.json").write_text(json.dumps(provenance, indent=2))
    # Agents can print the capability URL (e.g. an `env` listing), so it can reach any output, including
    # the request log's observations.
    scrub(job_dir)
    if (work / "requests.jsonl").exists():
        scrub_file(work / "requests.jsonl")
    StoragePath(args.output).upload_from(str(job_dir), recursive=True)
    if (work / "requests.jsonl").exists():
        StoragePath(f"{args.output.rstrip('/')}/requests.jsonl").upload_from(str(work / "requests.jsonl"))
    logger.info("Uploaded results to %s (harbor exit %d)", args.output, returncode)
    if returncode != 0:
        raise SystemExit(returncode)
    StoragePath(f"{args.output.rstrip('/')}/{SUCCESS_MARKER}").write_text("")


if __name__ == "__main__":
    main()
