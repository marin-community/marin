# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-use native evaluations for the adapted SWE-bench DEVELOPMENT cohort."""

import asyncio
import base64
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial
from marin.execution.artifact import artifact_record_identity
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.serve import local_inference
from minisweagent.config import get_config_from_spec
from minisweagent.utils.serialize import recursive_merge
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle
from shellbox.backends.qemu.bundle import guest_code_id
from shellbox.mini_agent import MINI_VERSION, NativeMiniAgent

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal, EvaluationJournal
from experiments.post_training.russell_rsi.sources import compact_json_sha256

TASK_IDS = (
    "django__django-16493",
    "sphinx-doc__sphinx-8265",
    "sphinx-doc__sphinx-7985",
    "django__django-16429",
    "matplotlib__matplotlib-22719",
    "django__django-13810",
    "django__django-15277",
    "pydata__xarray-7229",
)
HARBOR_COMMIT = "2666d6526477ae3e46030a8dc4f3f2c68fd7a84f"
TASK_COMMIT = "86723674f04e4209ac479d0fb75d9d9f44b4377e"
CONCURRENCY = 4
CONTEXT_TOKENS = 32768
DISK_BYTES = 10 * 1024**3
SCOPE = "adapted SWE-bench Verified DEVELOPMENT comparison"
GUEST_MEMORY_MB = 4096
AGENT_SETUP_TIMEOUT = 360
AGENT_TIMEOUT = 1800
VERIFIER_TIMEOUT = 300
LITELLM_VERSION = "1.104.0"
NATIVE_MODEL_RETRY_ATTEMPTS = 1
QUALIFICATION_JOB_TIMEOUT = 6 * 3600
EVALUATION_JOB_TIMEOUT = 2 * 3600
MODEL_ERRORS = frozenset(
    {
        "ContextLengthExceededError",
        "ModelAuthenticationError",
        "AuthenticationError",
        "APIConnectionError",
        "RateLimitError",
        "BadRequestError",
        "APITimeoutError",
        "ServiceUnavailableError",
        "InternalServerError",
    }
)


@dataclass(frozen=True)
class LoadedSourcePin:
    module: str
    file: PinnedFile


@dataclass(frozen=True)
class FrozenProducer:
    identity: str
    export_uri: str
    record: PinnedFile
    export_manifest: PinnedFile
    tokenizer: str
    tokenizer_revision: str
    chat_template: PinnedFile

    def validate(self) -> None:
        record = self.record.read_json()
        identity = artifact_record_identity(record)
        if identity != self.identity:
            raise ValueError("Checkpoint producer identity differs from its frozen record")
        match record["result_type"]:
            case "marin.training.training.LevanterCheckpoint":
                canonical_export = record["source"]
            case "marin.rl.skyrl.SkyRLRun":
                canonical_export = record["result"]["hf_model_uri"]
            case _:
                raise ValueError("Unsupported frozen checkpoint producer type")
        if canonical_export != self.export_uri:
            raise ValueError("Checkpoint export URI differs from its canonical producer record")
        manifest = self.export_manifest.read_json()
        if manifest["producer_identity"] != self.identity or manifest["export_uri"] != self.export_uri:
            raise ValueError("Checkpoint export does not bind the frozen producer")
        if (manifest["tokenizer"], manifest["tokenizer_revision"], manifest["chat_template_sha256"]) != (
            self.tokenizer,
            self.tokenizer_revision,
            self.chat_template.sha256,
        ):
            raise ValueError("Checkpoint tokenizer or template differs from the frozen export")
        self.chat_template.read_bytes()


@dataclass(frozen=True)
class DevelopmentPlan:
    source_manifest: PinnedFile
    image_manifest: PinnedFile
    runtime_bundle: RuntimeBundle
    task_archive: PinnedFile
    task_archive_size: int
    native_configs: tuple[PinnedFile, ...]
    resolved_native_config: dict
    source_pins: tuple[LoadedSourcePin, ...]
    producers: tuple[FrozenProducer, FrozenProducer]
    journal_path: str
    worker_cluster: str


def load_plan(value: dict) -> DevelopmentPlan:
    producers = tuple(
        FrozenProducer(
            identity=p["identity"],
            export_uri=p["export_uri"],
            record=PinnedFile(**p["record"]),
            export_manifest=PinnedFile(**p["export_manifest"]),
            tokenizer=p["tokenizer"],
            tokenizer_revision=p["tokenizer_revision"],
            chat_template=PinnedFile(**p["chat_template"]),
        )
        for p in value["producers"]
    )
    if len(producers) != 2 or producers[0].identity == producers[1].identity:
        raise ValueError("DEVELOPMENT requires two distinct frozen checkpoint producers")
    return DevelopmentPlan(
        source_manifest=PinnedFile(**value["source_manifest"]),
        image_manifest=PinnedFile(**value["image_manifest"]),
        runtime_bundle=RuntimeBundle(**value["runtime_bundle"]),
        task_archive=PinnedFile(**value["task_archive"]),
        task_archive_size=value["task_archive_size"],
        native_configs=tuple(PinnedFile(**p) for p in value["native_configs"]),
        resolved_native_config=value["resolved_native_config"],
        source_pins=tuple(LoadedSourcePin(p["module"], PinnedFile(**p["file"])) for p in value["source_pins"]),
        producers=(producers[0], producers[1]),
        journal_path=value["journal_path"],
        worker_cluster=value["worker_cluster"],
    )


def cohort(plan: DevelopmentPlan) -> tuple[dict, ...]:
    source = plan.source_manifest.read_json()
    images = plan.image_manifest.read_json()
    tasks = source["tasks"]
    prepared = images if isinstance(images, list) else images["images"]
    if tuple(t["task_id"] for t in tasks) != TASK_IDS or tuple(t["task_id"] for t in prepared) != TASK_IDS:
        raise ValueError("The DEVELOPMENT cohort order or membership changed")
    if source["task_source_commit"] != TASK_COMMIT or source["harbor_commit"] != HARBOR_COMMIT:
        raise ValueError("Task or Harbor source differs from the fixed DEVELOPMENT sources")
    entries = []
    for task, image in zip(tasks, prepared, strict=True):
        if task["dockerfile_sha256"] != image["dockerfile_sha256"]:
            raise ValueError("Prepared image does not bind the canonical Dockerfile")
        entries.append({"task": task, "image": image})
    return tuple(entries)


def binding(plan: DevelopmentPlan, entries: tuple[dict, ...]) -> dict:
    pins = asdict(plan)
    settings = {
        "scope": SCOPE,
        "harbor_commit": HARBOR_COMMIT,
        "mini_version": MINI_VERSION,
        "litellm_version": LITELLM_VERSION,
        "concurrency": CONCURRENCY,
        "context_tokens": CONTEXT_TOKENS,
        "attempts_per_slot": 1,
        "agent_setup_timeout": AGENT_SETUP_TIMEOUT,
        "agent_timeout": AGENT_TIMEOUT,
        "verifier_timeout": VERIFIER_TIMEOUT,
        "guest_memory_mb": GUEST_MEMORY_MB,
        "guest_network": "deny",
        "model_retries": NATIVE_MODEL_RETRY_ATTEMPTS - 1,
        "harbor_retries": 0,
        "job_failure_retries": 0,
        "job_preemption_retries": 0,
        "qualification_job_timeout": QUALIFICATION_JOB_TIMEOUT,
        "evaluation_job_timeout": EVALUATION_JOB_TIMEOUT,
    }
    task_hashes = {e["task"]["task_id"]: compact_json_sha256(e) for e in entries}
    return json.loads(
        json.dumps(
            {
                "plan": pins,
                "settings": settings,
                "attempts": {"producer-0": task_hashes, "producer-1": task_hashes},
            }
        )
    )


def native_config_specs(plan: DevelopmentPlan, directory: Path) -> list[str]:
    provenance = json.loads(importlib.metadata.distribution("harbor").read_text("direct_url.json") or "{}")
    commit = provenance.get("vcs_info", {}).get("commit_id")
    if commit is None and provenance.get("url", "").startswith("file:"):
        installed = Path(unquote(urlparse(provenance["url"]).path))
        commit = subprocess.run(
            ["git", "-C", str(installed), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    if commit != HARBOR_COMMIT:
        raise ValueError("The installed Harbor source differs from the pinned campaign commit")
    if importlib.metadata.version("mini-swe-agent") != MINI_VERSION:
        raise ValueError("DEVELOPMENT requires the actual mini-swe-agent 2.1.0 installation")
    if importlib.metadata.version("litellm") != LITELLM_VERSION:
        raise ValueError("DEVELOPMENT requires the frozen LiteLLM installation")
    specs = []
    for index, pin in enumerate(plan.native_configs):
        path = directory / f"native-{index}.yaml"
        path.write_bytes(pin.read_bytes())
        specs.append(str(path))
    resolved = recursive_merge(*(get_config_from_spec(spec) for spec in specs))
    if resolved != plan.resolved_native_config:
        raise ValueError("Native config resolution differs from the frozen config")
    kwargs = resolved["model"]["model_kwargs"]
    if kwargs["num_retries"] != 0:
        raise ValueError("Native model retries must be disabled")
    if not {"temperature", "max_tokens"}.issubset(kwargs) or "step_limit" not in resolved["agent"]:
        raise ValueError("Native config must freeze temperature, maximum tokens, and step limit")
    return specs


def worker_provenance(plan: DevelopmentPlan) -> dict:
    root = Path(__file__).resolve().parents[3]
    branch_modules = {
        "experiments.post_training.russell_rsi.agentic_development": Path(__file__).resolve(),
        "experiments.post_training.russell_rsi.evaluation_journal": (
            root / "experiments/post_training/russell_rsi/evaluation_journal.py"
        ),
        "shellbox.mini_agent": root / "lib/shellbox/src/shellbox/mini_agent.py",
        "shellbox.mini_environment": root / "lib/shellbox/src/shellbox/mini_environment.py",
        "shellbox.backends.qemu.environment": root / "lib/shellbox/src/shellbox/backends/qemu/environment.py",
        "shellbox.backends.qemu.machine": root / "lib/shellbox/src/shellbox/backends/qemu/machine.py",
        "rigging.runtime_bundle": root / "lib/rigging/src/rigging/runtime_bundle.py",
        "marin.inference.serve": root / "lib/marin/src/marin/inference/serve.py",
    }
    required = set(branch_modules) | {
        "harbor.trial.trial",
        "harbor.verifier.verifier",
        "minisweagent.agents.interactive",
        "minisweagent.models.litellm_model",
    }
    supplied = {pin.module: pin for pin in plan.source_pins}
    if not required.issubset(supplied) or len(supplied) != len(plan.source_pins):
        raise ValueError("DEVELOPMENT requires distinct pins for the loaded worker, adapter, runtime, and controllers")
    modules = {}
    for name, pin in supplied.items():
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError("A pinned worker module has no loaded source file")
        loaded = Path(module.__file__).resolve()
        if name in branch_modules and loaded != branch_modules[name].resolve():
            raise ValueError(f"Worker imported {name} outside the packaged branch")
        if file_sha256(loaded) != pin.file.sha256:
            raise ValueError(f"Loaded worker source differs from its frozen pin: {name}")
        modules[name] = {"path": str(loaded), "sha256": pin.file.sha256}
    return {"branch_root": str(root), "modules": modules}


def materialize_tasks(plan: DevelopmentPlan, entries: tuple[dict, ...], parent: Path) -> Path:
    """Extract the fixed task archive after its size, hash, and inventory checks."""
    archive = parent / "canonical-tasks.tar.gz"
    storage = StoragePath(plan.task_archive.uri)
    if storage.size() != plan.task_archive_size:
        raise ValueError("Canonical task archive size differs from its frozen pin")
    storage.download_to(str(archive))
    if file_sha256(archive) != plan.task_archive.sha256:
        raise ValueError("Canonical task archive hash differs from its frozen pin")
    expected = {}
    for entry in entries:
        task_id = entry["task"]["task_id"]
        for item in entry["task"]["source_files"]:
            relative = Path(item["path"]).relative_to(f"datasets/swebench-verified/{task_id}")
            expected[f"tasks/{task_id}/{relative}"] = item["size"]
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Canonical task archive has duplicate entries")
        if {m.name: m.size for m in members if m.isfile()} != expected:
            raise ValueError("Canonical task archive inventory differs from its source metadata")
        if len(members) > 128:
            raise ValueError("Canonical task archive has excessive entries")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "tasks":
                raise ValueError("Canonical task archive path escapes its data directory")
            if not member.isdir() and not member.isfile():
                raise ValueError("Canonical task archive contains an unsupported file type")
        source.extractall(parent, filter="data")
    root = parent / "tasks"
    for entry in entries:
        verify_task_files(entry, root)
    archive.unlink()
    return root


def verify_task_files(entry: dict, task_root: Path) -> Path:
    task_id = entry["task"]["task_id"]
    path = task_root / task_id
    for item in entry["task"]["source_files"]:
        relative = Path(item["path"])
        local = path / relative.relative_to(f"datasets/swebench-verified/{task_id}")
        if not local.resolve().is_relative_to(path.resolve()):
            raise ValueError("Task file escapes the fixed task directory")
        content = local.read_bytes()
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if len(content) != item["size"] or digest != item["sha"]:
            raise ValueError("Canonical task source bytes changed")
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    expected = {
        str(Path(item["path"]).relative_to(f"datasets/swebench-verified/{task_id}"))
        for item in entry["task"]["source_files"]
    }
    if actual != expected:
        raise ValueError("Canonical task source inventory changed")
    return path


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def install_guest_bundle(entry: dict, plan: DevelopmentPlan, parent: Path) -> Path:
    """Install one pinned data archive without changing RuntimeBundle limits."""
    image = entry["image"]
    manifest = PinnedFile(image["data_manifest_uri"], image["data_manifest_sha256"]).read_json()
    task_id = entry["task"]["task_id"]
    expected_metadata = {
        "task_id": task_id,
        "archive_top_directory": task_id,
        "rootfs_logical_bytes": DISK_BYTES,
        "memory_mb": GUEST_MEMORY_MB,
        "shared_runtime_sha256": plan.runtime_bundle.archive_sha256,
        "source_files": entry["task"]["source_files"],
        "bundle_uri": image["bundle_uri"],
        "bundle_sha256": image["bundle_sha256"],
        "bundle_bytes": image["bundle_size"],
        "derived_manifest_digest": image["derived_manifest_digest"],
        "original_dockerfile_sha256": image["dockerfile_sha256"],
        "derived_recipe_sha256": image["derived_recipe_sha256"],
        "image": image["image"],
        "task_source_commit": TASK_COMMIT,
        "harbor_commit": HARBOR_COMMIT,
    }
    changed = [key for key, value in expected_metadata.items() if manifest.get(key) != value]
    if changed:
        raise ValueError(f"Guest data manifest differs from frozen metadata: {changed}")
    if image["image"]["guest_code_id"] != guest_code_id():
        raise ValueError("Guest protocol differs from the loaded guest source")
    target = parent / task_id
    if target.exists():
        raise ValueError("Guest installation requires a fresh target directory")
    parent.mkdir(parents=True, exist_ok=True)
    storage = StoragePath(image["bundle_uri"])
    if storage.size() != image["bundle_size"]:
        raise ValueError("Guest archive size differs from the pinned size")
    archive = parent / f"{task_id}.tar.gz"
    storage.download_to(str(archive))
    if file_sha256(archive) != image["bundle_sha256"]:
        raise ValueError("Guest archive hash differs from its pin")
    expected = {f"{task_id}/{item['path']}": item for item in manifest["files"]}
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        files = {m.name: m for m in members if m.isfile()}
        if len(members) > len(expected) + 128 or len({m.name for m in members}) != len(members):
            raise ValueError("Guest archive has excessive or duplicate entries")
        if set(files) != set(expected):
            raise ValueError("Guest archive inventory differs from the data manifest")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != task_id:
                raise ValueError("Guest archive path escapes its data directory")
            if not member.isdir() and not member.isfile():
                raise ValueError("Guest data archive contains an unsupported file type")
            if member.isfile() and member.size != expected[member.name]["size"]:
                raise ValueError("Guest file size differs from the data manifest")
        source.extractall(parent, filter="data")
    for item in manifest["files"]:
        if file_sha256(target / item["path"]) != item["sha256"]:
            raise ValueError("Guest extracted file hash differs from the data manifest")
    if json.loads((target / "image.json").read_bytes()) != image["image"]:
        raise ValueError("Extracted guest image metadata differs from the prepared image")
    if (target / "rootfs.ext4").stat().st_size != DISK_BYTES:
        raise ValueError("Guest disk does not retain the declared 10 GiB capacity")
    archive.unlink()
    return target


def save_directory(source: Path, destination: StoragePath) -> dict:
    inventory = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            relative = str(path.relative_to(source))
            record = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "base64": base64.b64encode(content).decode(),
            }
            write_once(destination / f"{relative}.json", record)
            inventory[relative] = {"sha256": record["sha256"], "size": record["size"]}
    write_once(destination / "inventory.json", inventory)
    return inventory


async def persist_before_verifier(trial: Trial, attempt: AttemptJournal, native: bool) -> None:
    destination = attempt.directory / "pre-verifier"
    inventory = save_directory(Path(str(trial.paths.agent_dir)), destination / "agent")
    native_complete = not native or {"mini-swe-agent.trajectory.json", "mini-swe-agent.txt"}.issubset(inventory)
    environment = trial.agent_environment
    try:
        cwd = await environment.exec("pwd", timeout_sec=30)
    except Exception as error:
        cwd_record = {"exception_type": type(error).__name__, "exception_message": str(error)}
        workdir = None
    else:
        cwd_record = cwd.model_dump(mode="json")
        workdir = cwd.stdout.strip() if cwd.return_code == 0 and cwd.stdout else None
    commands = {
        "patch": "git diff --binary HEAD",
        "inventory": "git status --porcelain=v1 -z --untracked-files=all",
        "tracked": "git ls-files -s -z",
    }
    records = {"cwd": cwd_record}
    for name, command in commands.items():
        try:
            result = await environment.exec(command, cwd=workdir, timeout_sec=60)
        except Exception as error:
            records[name] = {"exception_type": type(error).__name__, "exception_message": str(error)}
        else:
            records[name] = result.model_dump(mode="json")
    write_once(destination / "workspace.json", {"cwd": workdir, "records": records, "full_workspace_recovery": False})
    write_once(
        destination / "complete.json",
        {
            "agent_inventory": inventory,
            "native_evidence_complete": native_complete,
            "workspace_recorded": True,
            "grade_only_recovery_permitted": False,
        },
    )


def error_category(serialized: dict) -> str | None:
    error = serialized.get("exception_info")
    if error is None:
        return None
    metadata = (serialized.get("agent_result") or {}).get("metadata") or {}
    native_exit = metadata.get("exit_status")
    name = error["exception_type"]
    if native_exit in MODEL_ERRORS or name in MODEL_ERRORS:
        return "model_error"
    if name in {"AgentTimeoutError", "NonZeroAgentExitCodeError", "TurnCapExhaustedError"}:
        return "agent_error"
    return "infrastructure_error"


async def run_trial_slot(
    attempt: AttemptJournal,
    trial_config: TrialConfig,
    *,
    native: bool,
) -> dict:
    """Reserve the slot before Harbor creates an environment or an agent."""

    async def operation() -> dict:
        trial = await Trial.create(trial_config)

        async def before_verifier(event: TrialHookEvent) -> None:
            await persist_before_verifier(trial, attempt, native)

        trial.add_hook(TrialEvent.VERIFICATION_START, before_verifier)
        result = await trial.run()
        serialized = result.model_dump(mode="json")
        write_once(attempt.directory / "canonical-trial-result.json", serialized)
        terminal_agent_inventory = save_directory(Path(str(trial.paths.agent_dir)), attempt.directory / "terminal-agent")
        verifier_inventory = save_directory(Path(str(trial.paths.verifier_dir)), attempt.directory / "verifier")
        verifier = serialized.get("verifier_result")
        rewards = verifier.get("rewards") if verifier else None
        category = error_category(serialized)
        valid = rewards is not None and set(rewards) == {"reward"} and rewards["reward"] in (0, 1)
        if valid and not (attempt.directory / "pre-verifier" / "complete.json").exists():
            raise RuntimeError("Canonical grade has no durable pre-verifier evidence")
        return {
            "status": "valid_grade" if valid else category or "invalid_grade",
            "error_category": category,
            "terminal_agent_inventory": terminal_agent_inventory,
            "grade": rewards["reward"] if valid and rewards is not None else None,
            "trial_result": serialized,
            "verifier_inventory": verifier_inventory,
        }

    return await attempt.run(operation)


def trial_config(task_path: Path, bundle: Path, trial_parent: Path, task_id: str, agent: AgentConfig) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=task_path),
        trial_name=task_id,
        trials_dir=trial_parent,
        agent=agent,
        environment=EnvironmentConfig(
            import_path="shellbox.backends.qemu.environment:QemuEnvironment",
            kwargs={"guest_bundle": str(bundle), "guest_memory_mb": GUEST_MEMORY_MB, "network_policy": "deny"},
        ),
        verifier=VerifierConfig(override_timeout_sec=VERIFIER_TIMEOUT),
    )


@contextmanager
def scripted_endpoint(commands: tuple[str, ...], response_status: int = 200):
    """Serve fixed tool responses on localhost without model inference."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(request)
            if index >= len(commands):
                self.send_error(409)
                return
            response = {
                "id": f"qualification-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": "scripted",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Execute the command.",
                            "tool_calls": [
                                {
                                    "id": f"command-{index}",
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": json.dumps({"command": commands[index]})},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            if response_status != 200:
                response = {
                    "error": {
                        "message": "Fixture authentication error",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                }
            body = json.dumps(response).encode()
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def qualify_native_contract(
    attempt: AttemptJournal, task: Path, bundle: Path, parent: Path, task_id: str, specs: list[str]
) -> dict:
    async def operation() -> dict:
        config = trial_config(task, bundle, parent, task_id, AgentConfig(name="nop"))
        config = config.model_copy(update={"verifier": VerifierConfig(disable=True)})
        trial = await Trial.create(config)
        commands = (
            'printf "%s:%s:" "$RUSSELL_CONTRACT_SENTINEL" "$PWD"; printf first; printf second >&2; printf third; exit 7',
            "printf partial; sleep 3",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo qualified",
        )
        with scripted_endpoint(commands) as (endpoint, requests):

            async def native_run(event: TrialHookEvent) -> None:
                agent = NativeMiniAgent(
                    logs_dir=Path(str(trial.paths.agent_dir)),
                    model_name="scripted-qualification",
                    model_alias="hosted_vllm/scripted",
                    api_base=endpoint,
                    model_retry_attempts=NATIVE_MODEL_RETRY_ATTEMPTS,
                    config_specs=[
                        *specs,
                        "environment.cwd=/tmp",
                        'environment.env={"RUSSELL_CONTRACT_SENTINEL":"guest-contract"}',
                        "environment.timeout=1",
                    ],
                )
                await agent.run(
                    "Execute the three fixed qualification commands.", trial.agent_environment, AgentContext()
                )

            trial.add_hook(TrialEvent.AGENT_START, native_run)
            result = await trial.run()
        serialized = result.model_dump(mode="json")
        write_once(attempt.directory / "canonical-trial-result.json", serialized)
        inventory = save_directory(Path(str(trial.paths.agent_dir)), attempt.directory / "agent")
        trajectory = Path(str(trial.paths.agent_dir)) / "mini-swe-agent.trajectory.json"
        native = json.loads(trajectory.read_bytes()) if trajectory.exists() else {}
        observations = [m for m in native.get("messages", []) if m["role"] == "tool"]
        qualified = (
            serialized.get("exception_info") is None
            and len(requests) == 3
            and len(observations) == 2
            and "guest-contract:/tmp:firstsecondthird" in observations[0]["content"]
            and observations[0]["extra"]["returncode"] == 7
            and observations[1]["extra"]["exception_type"] == "TimeoutExpired"
            and observations[1]["extra"]["raw_output"] == "partial"
            and native.get("info", {}).get("exit_status") == "Submitted"
            and native["info"]["submission"] == "qualified\n"
            and requests[0]["tools"][0]["function"]["name"] == "bash"
        )
        write_once(attempt.directory / "scripted-requests.json", {"requests": requests, "models_called": 0})
        return {"qualified": qualified, "agent_inventory": inventory, "models_called": 0, "trial_result": serialized}

    return await attempt.run(operation)


@dataclass(frozen=True)
class QualificationConfig:
    plan: DevelopmentPlan
    output_path: str


def run_qualification(config: QualificationConfig) -> None:
    plan = config.plan
    entries = cohort(plan)
    frozen = binding(plan, entries)
    directory = StoragePath(config.output_path)
    write_once(directory / "binding.json", frozen)
    install_runtime_bundle(plan.runtime_bundle)
    write_once(StoragePath(config.output_path) / "worker-import-provenance.json", worker_provenance(plan))
    with tempfile.TemporaryDirectory(prefix="russell-native-qualification-") as temporary:
        root = Path(temporary)
        specs = native_config_specs(plan, root)
        task_root = materialize_tasks(plan, entries, root)
        results = {}
        for entry in entries:
            task_id = entry["task"]["task_id"]
            task = task_root / task_id
            bundle = install_guest_bundle(entry, plan, root / "guests")
            contract = AttemptJournal(
                directory / "native-contract" / task_id,
                {"binding": frozen, "role": "native-contract", "task_id": task_id},
            )
            results[task_id] = {
                "native_contract": asyncio.run(
                    qualify_native_contract(contract, task, bundle, root / "native-contract", task_id, specs)
                )
            }
            for role, agent in (("baseline", "nop"), ("reference", "oracle")):
                attempt = AttemptJournal(
                    directory / role / task_id, {"binding": frozen, "role": role, "task_id": task_id}
                )
                trial = trial_config(
                    task,
                    bundle,
                    root / role,
                    task_id,
                    AgentConfig(
                        name=agent, override_setup_timeout_sec=AGENT_SETUP_TIMEOUT, override_timeout_sec=AGENT_TIMEOUT
                    ),
                )
                results[task_id][role] = asyncio.run(run_trial_slot(attempt, trial, native=False))
        passed = all(
            row["baseline"]["status"] == row["reference"]["status"] == "valid_grade"
            and row["baseline"]["grade"] == 0
            and row["reference"]["grade"] == 1
            and row["native_contract"]["qualified"]
            for row in results.values()
        )
        write_once(
            directory / "qualification.json",
            {
                "binding": frozen,
                "tasks": results,
                "qualified": passed,
                "cohort_held": not passed,
                "native_command_contract_qualified": all(
                    row["native_contract"]["qualified"] for row in results.values()
                ),
            },
        )


@dataclass(frozen=True)
class CheckpointEvaluationConfig:
    plan: DevelopmentPlan
    producer_index: int
    qualification_path: str
    output_path: str


async def evaluate_slots(
    plan: DevelopmentPlan,
    entries: tuple[dict, ...],
    journal: EvaluationJournal,
    index: int,
    bundles: dict[str, Path],
    specs: list[str],
    root: Path,
    task_root: Path,
    api_base: str,
    model_alias: str,
) -> dict:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def evaluate(entry: dict) -> tuple[str, dict]:
        task_id = entry["task"]["task_id"]
        attempt = journal.attempt(f"producer-{index}", task_id, compact_json_sha256(entry))
        saved = attempt.saved_result()
        if saved is not None:
            return task_id, saved
        async with semaphore:
            task_path = task_root / task_id
            agent = AgentConfig(
                import_path="shellbox.mini_agent:NativeMiniAgent",
                model_name=plan.producers[index].identity,
                model_alias=f"hosted_vllm/{model_alias}",
                override_setup_timeout_sec=AGENT_SETUP_TIMEOUT,
                override_timeout_sec=AGENT_TIMEOUT,
                kwargs={
                    "config_specs": specs,
                    "api_base": api_base,
                    "model_retry_attempts": NATIVE_MODEL_RETRY_ATTEMPTS,
                },
            )
            trial = trial_config(task_path, bundles[task_id], root / "trials", task_id, agent)
            return task_id, await run_trial_slot(attempt, trial, native=True)

    outcomes = await asyncio.gather(*(evaluate(entry) for entry in entries), return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    if failures:
        raise ExceptionGroup("DEVELOPMENT has incomplete durable slots; completed siblings remain saved", failures)
    results = {}
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
        task_id, result = outcome
        results[task_id] = result
    return results


def run_checkpoint_evaluation(config: CheckpointEvaluationConfig) -> None:
    plan = config.plan
    entries = cohort(plan)
    frozen = binding(plan, entries)
    journal = EvaluationJournal(StoragePath(plan.journal_path), frozen)
    journal.seal()
    qualification = json.loads((StoragePath(config.qualification_path) / "qualification.json").read_text())
    if (
        qualification["binding"] != frozen
        or not qualification["qualified"]
        or not qualification["native_command_contract_qualified"]
    ):
        raise ValueError("The whole DEVELOPMENT cohort and native command contract must qualify before inference")
    role = f"producer-{config.producer_index}"
    saved = {
        task_id: journal.attempt(role, task_id, frozen["attempts"][role][task_id]).saved_result() for task_id in TASK_IDS
    }
    if all(value is not None for value in saved.values()):
        write_once(
            StoragePath(config.output_path) / "checkpoint-results.json",
            {"binding": frozen, "producer_index": config.producer_index, "slots": saved},
        )
        return
    for producer in plan.producers:
        producer.validate()
    write_once(StoragePath(config.output_path) / "worker-import-provenance.json", worker_provenance(plan))
    install_runtime_bundle(plan.runtime_bundle)
    with tempfile.TemporaryDirectory(prefix="russell-native-development-") as temporary:
        root = Path(temporary)
        specs = native_config_specs(plan, root)
        task_root = materialize_tasks(plan, entries, root)
        bundles = {entry["task"]["task_id"]: install_guest_bundle(entry, plan, root / "guests") for entry in entries}
        producer = plan.producers[config.producer_index]
        template = root / "chat-template.jinja"
        template.write_bytes(producer.chat_template.read_bytes())
        with local_inference(
            ServedModelConfig(
                weights=producer.export_uri,
                api_model="russell-native-dev",
                tokenizer=producer.tokenizer,
                tokenizer_revision=producer.tokenizer_revision,
                max_model_len=CONTEXT_TOKENS,
                tensor_parallel_size=1,
            ),
            VllmEngineConfig(
                launcher=VllmLauncherType.CUDA,
                source=VllmSource.MARIN_FORK,
                max_num_batched_tokens=8192,
                max_num_seqs=8,
                extra_args=(
                    "--data-parallel-size",
                    "8",
                    "--enable-expert-parallel",
                    "--model-loader-extra-config",
                    '{"distributed":true}',
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "hermes",
                    "--chat-template",
                    str(template),
                ),
            ),
            num_chips=8,
        ) as server:
            results = asyncio.run(
                evaluate_slots(
                    plan,
                    entries,
                    journal,
                    config.producer_index,
                    bundles,
                    specs,
                    root,
                    task_root,
                    server.model.endpoint.base_url,
                    server.model.endpoint.model,
                )
            )
        write_once(
            StoragePath(config.output_path) / "checkpoint-results.json",
            {"binding": frozen, "producer_index": config.producer_index, "slots": results},
        )


@dataclass(frozen=True)
class PairedReportConfig:
    plan: DevelopmentPlan
    paths: tuple[str, str]
    output_path: str


def paired_report(config: PairedReportConfig) -> None:
    frozen = binding(config.plan, cohort(config.plan))
    rows = []
    conditions = []
    for index, path in enumerate(config.paths):
        record = json.loads((StoragePath(path) / "checkpoint-results.json").read_text())
        if record["binding"] != frozen or record["producer_index"] != index or set(record["slots"]) != set(TASK_IDS):
            raise ValueError("DEVELOPMENT report requires exactly sixteen distinct frozen slots")
        conditions.append(record["slots"])
        for task_id in TASK_IDS:
            slot = record["slots"][task_id]
            if slot["status"] == "valid_grade" and slot["grade"] not in (0, 1):
                raise ValueError("DEVELOPMENT slot has an invalid canonical grade")
            if slot["status"] not in (
                "valid_grade",
                "infrastructure_error",
                "agent_error",
                "model_error",
                "invalid_grade",
            ):
                raise ValueError("DEVELOPMENT slot has no terminal outcome")
            rows.append(
                {
                    "producer_index": index,
                    "producer_identity": config.plan.producers[index].identity,
                    "task_id": task_id,
                    "status": slot["status"],
                    "grade": slot["grade"],
                    "error_category": slot["error_category"],
                }
            )
    comparable = [
        task_id for task_id in TASK_IDS if all(condition[task_id]["status"] == "valid_grade" for condition in conditions)
    ]
    wins = [task_id for task_id in comparable if conditions[1][task_id]["grade"] > conditions[0][task_id]["grade"]]
    losses = [task_id for task_id in comparable if conditions[1][task_id]["grade"] < conditions[0][task_id]["grade"]]
    write_once(
        StoragePath(config.output_path) / "development-comparison.json",
        {
            "scope": SCOPE,
            "binding": frozen,
            "slots": rows,
            "slot_count": len(rows),
            "valid_grade_count": sum(row["status"] == "valid_grade" for row in rows),
            "infrastructure_error_count": sum(row["error_category"] == "infrastructure_error" for row in rows),
            "agent_error_count": sum(row["error_category"] == "agent_error" for row in rows),
            "model_error_count": sum(row["error_category"] == "model_error" for row in rows),
            "invalid_grade_count": sum(row["status"] == "invalid_grade" for row in rows),
            "error_counts_include_valid_grades": True,
            "paired_valid_count": len(comparable),
            "wins": wins,
            "losses": losses,
            "paired_net_gain": len(wins) - len(losses),
            "promotion_gate_changed": False,
        },
    )
