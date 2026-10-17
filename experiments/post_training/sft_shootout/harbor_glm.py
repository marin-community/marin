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
import re
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
COST = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
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
    # tb2-recovery.yaml: Terminus-2, 65,536 input / 16,384 output tokens.
    "terminus-2": {
        "import_path": "trial_agents:TrialTerminus2",
        "override_setup_timeout_sec": 360,
        "override_timeout_sec": 1800,
        "kwargs": {
            "model_info": {"max_input_tokens": 65536, "max_output_tokens": 16384, **COST},
            "llm_call_kwargs": {"max_tokens": 16384},
        },
    },
    # swebench-verified.yaml: mini-swe-agent 2.1.0, 32,768-token total context. The agent runs inside
    # the sandbox, which reaches only the proxy during the run (no github.com/swesmith mirrors).
    "mini-swe-agent": {
        "import_path": "trial_agents:TrialMiniSweAgent",
        "override_setup_timeout_sec": 360,
        "override_timeout_sec": 1800,
        "kwargs": {
            "version": "2.1.0",
            "model_info": {"max_input_tokens": 32768, "max_output_tokens": 16384, **COST},
            "agent_domain_allow_list": AGENT_DOMAIN_ALLOW_LIST,
        },
    },
}

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


def scrub(root: Path, capability: str) -> None:
    """Replace the capability URL's token in every text file under ``root``.

    Agents echo the URL in their logs, sometimes wrapped across lines (also JSON-escaped as ``\\n``
    inside result.json), so the token is matched with any line breaks between its characters.
    """
    token = max(urlparse(capability).path.split("/"), key=len)
    pattern = re.compile(r"(?:\s|\\n)*".join(re.escape(c) for c in token))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        scrubbed = pattern.sub("<capability-token>", text)
        if scrubbed != text:
            path.write_text(scrubbed)


def job_config(args: argparse.Namespace, api_base: str | None, jobs_dir: Path) -> dict:
    agent = json.loads(json.dumps(AGENTS[args.agent]))
    if api_base is not None:
        agent["model_name"] = f"hosted_vllm/{GLM_MODEL}"
        agent["kwargs"]["api_base"] = api_base
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
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument("--package-revision", required=True, help="Source and revision of the task packages, for provenance.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
            config = job_config(args, None, work / "jobs")
        else:
            window = AGENTS[args.agent]["kwargs"]["model_info"]["max_input_tokens"]
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
            config = job_config(args, f"{capability}/v1", work / "jobs")
        returncode = run_harbor(args, config, work)

    job_dir = work / "jobs" / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "args": vars(args),
        "harbor": config,
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
            "agent_phase_domain_allow_list": AGENTS[args.agent].get("kwargs", {}).get("agent_domain_allow_list"),
            "setup_and_verifier": "task policy (allow_internet)",
        },
        "relay": None
        if not args.relay
        else {
            "student": f"{STUDENT_CONFIG}@{STUDENT_REVISION}",
            "budget_fraction": args.budget_fraction,
            "budget_tokens": int(AGENTS[args.agent]["kwargs"]["model_info"]["max_input_tokens"] * args.budget_fraction),
            "budget_counted_with": "student tokenizer",
            "triggers": ["format_error: one teacher repair turn", "context_budget: teacher to the end", "completion_claim: student reply withheld, teacher to the end"],
        },
    }
    (job_dir / "driver_config.json").write_text(json.dumps(provenance, indent=2))
    if capability is not None:
        scrub(job_dir, capability)
    StoragePath(args.output).upload_from(str(job_dir), recursive=True)
    if (work / "requests.jsonl").exists():
        StoragePath(f"{args.output.rstrip('/')}/requests.jsonl").upload_from(str(work / "requests.jsonl"))
    logger.info("Uploaded results to %s (harbor exit %d)", args.output, returncode)
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
