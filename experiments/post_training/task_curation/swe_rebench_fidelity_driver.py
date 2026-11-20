# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench V2.1 fidelity gate driver: Daytona EVAL declarative trials.

Standalone driver fetched and executed by the regional job (the maintained ingest-A launch
recipe): the job pins this file's sha256, verifies the bytes, and execs it. Per sampled row
(fidelity-sample-manifest.json, 200 tasks across the 10 consolidated groups) it runs nop +
oracle Daytona trials on two hardened packages that differ ONLY in their environment — the
row's upstream digest-pinned image versus the repair's shared base image plus the declared
task-start setup — under the same SWE_REBENCH shared-source contract, and requires identical
F2P/P2P outcomes. A mismatching group splits further or keeps its upstream image; tests are
never weakened.

Trials run through the landed Harbor machinery on the Daytona EVAL backend with declarative
provisioning ({'provisioning': 'declarative'}), exactly like the ingest lane's real native
trials: Daytona builds each package's declared Dockerfile remotely; no local sandbox
machinery. run_trial/trial_config are vendored from the landed scale-lego protected-QA
module (small, stable helpers; importing the QA module would drag local-backend-only
machinery into this driver's environment).
"""

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import resource
import shlex
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import tomlkit
from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    DaytonaConfig,
    Resources,
)
from harbor.host_capabilities import host_capabilities_from_environment
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret
from taskcompendium.harbor.shared_source import harden_shared_source
from taskcompendium.integrity import SharedDependencyState

from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    WORKSPACE,
    dockerfile,
    family_config,
    row_base,
    setup_command,
    swe_rebench_profile,
    task_toml,
    to_workspace,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream import task_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("swe-rebench-fidelity-driver")

ATTEMPT_TIMEOUT = 4 * 3600.0
TRIAL_WORKERS = int(os.environ.get("GATE_WORKERS", "8"))
DAYTONA_TARGET = "us"
# Gate-resource deviation (coordinator 2026-11-17: gate-only, NOT a release approval). The
# upstream declares 6 CPUs / 24 GB / 32 GB; the Daytona EVAL account caps sandboxes at
# 4 CPUs / 8 GB (and A's DCAgent trials ran 10 GB storage). Both variants declare this shape
# so the comparison holds resources constant; the release keeps the declared values and
# routes to a backend that can run them unless the user approves the reduction.
GATE_RESOURCES = {"cpus": 4, "memory_mb": 8192, "storage_mb": 10240}
VERIFIER_DEADLINE_SEC = 2400.0


def trial_environment() -> dict:
    """The declared trial executor: Daytona with declarative provisioning.

    The org follows ruling 71 (2026-11-20): the EVAL org is reserved for priority evals, so
    swe-repair trials run on the DATA org — the secret named by DAYTONA_KEY_SECRET, default
    DAYTONA_DATA_API_KEY. The kwargs are exactly the ingest lane's proven shape; nothing else
    (the Daytona target comes from the job's DAYTONA_TARGET environment). The connection
    fields themselves travel through the process environment the Trial machinery snapshots.
    """

    key = get_secret(os.environ.get("DAYTONA_KEY_SECRET", "DAYTONA_DATA_API_KEY"))
    if key is None:
        raise ValueError("Daytona trials require the org's Daytona API key secret")
    os.environ["DAYTONA_API_KEY"] = key
    os.environ.setdefault("DAYTONA_TARGET", "us")
    # The connection validator requires all three explicitly; log presence only, never values.
    LOGGER.info(
        "daytona connection presence: api_key=%s api_url=%s target=%s",
        bool(os.environ.get("DAYTONA_API_KEY")),
        bool(os.environ.get("DAYTONA_API_URL")),
        bool(os.environ.get("DAYTONA_TARGET")),
    )
    return {"type": "daytona", "kwargs": {"provisioning": "declarative"}}


def trial_config(package: Path, trials_dir: Path, agent: dict, name: str) -> dict:
    return {
        "task": {"path": str(package)},
        "agent": agent,
        "environment": trial_environment(),
        "trials_dir": str(trials_dir),
        "trial_name": name,
        "trial_attempt_timeout_sec": ATTEMPT_TIMEOUT,
    }


async def run_trial(config_path: Path) -> dict:
    """Run one production trial in-process and return its result summary."""

    config = TrialConfig.model_validate_json(config_path.read_text())
    trial = await Trial.create(
        config, host_capabilities=host_capabilities_from_environment(os.environ, model_authentication={})
    )
    result = await trial.run()
    verifier = result.verifier_result
    rewards = dict(verifier.rewards) if verifier else {}
    return {
        "rewards": rewards,
        "completion_recorded": result.completion is not None,
        "exception": result.exception_info.model_dump(mode="json") if result.exception_info else None,
        "verifier": {
            "stdout": verifier.stdout if verifier else None,
            "stderr": verifier.stderr if verifier else None,
            "exit_code": verifier.exit_code if verifier else None,
        },
    }


def _flatten(data: dict) -> dict:
    flat = dict(data)
    install_config = data.get("install_config") or {}
    flat["test_cmd"] = install_config.get("test_cmd")
    flat["log_parser"] = install_config.get("log_parser")
    flat["install"] = install_config.get("install") or []
    return flat


LAYER_SETUP = (
    # The registry-image path never builds the Dockerfile layer, so the protected identities
    # are created here (the image layer's own useradd lines, verbatim).
    "getent passwd 10001 >/dev/null || useradd --create-home --uid 10001 --user-group taskcompendium-agent\n"
    "getent passwd 10002 >/dev/null || useradd --create-home --uid 10002 --user-group taskcompendium-grade\n"
    "mkdir -p /home/taskcompendium-agent /opt/taskcompendium-shared-bootstrap\n"
    "cp /setup_files/taskcompendium-shared/control.py"
    " /setup_files/taskcompendium-shared/shared_pytest.py"
    " /opt/taskcompendium-shared-bootstrap/\n"
    "chown 10001:10001 /home/taskcompendium-agent\n"
    "chmod -R go-w /opt/taskcompendium-shared-bootstrap\n"
)
# The gate accommodation for upstream images without a qualifying interpreter: provision a
# python >= 3.9 at setup exactly like the authored bases do (disclosed; held constant across
# the comparison since the condensed base carries the same provisioning).
PROVISION_PYTHON = (
    "if ! command -v python3 >/dev/null 2>&1 "
    "|| ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then "
    "apt-get update -qq && apt-get -o Dpkg::Options::=--force-confdef "
    "-o Dpkg::Options::=--force-confold install -y -qq python3; fi\n"
)


def adapt_for_daytona(files: dict[str, bytes], workdir: str, image: str) -> None:
    """Move the hardened image layer into the pre-agent setup; declare the image directly.

    Vendored from the landed scale-lego protected-QA adaptation (the proven Daytona shape):
    the trial packages declare ``[environment].docker_image`` so Daytona pulls the pinned
    reference directly — no remote build — and the protected-root layer runs at setup time
    from ``setup_files/``. The Dockerfile stays as the declared provenance of the image.
    """
    dockerfile = files["environment/Dockerfile"].decode()
    files["environment/Dockerfile"] = dockerfile.split("\nUSER 0:0\n")[0].encode()
    files["setup_files/taskcompendium-shared/control.py"] = files.pop("environment/taskcompendium-shared/control.py")
    files["setup_files/taskcompendium-shared/shared_pytest.py"] = files.pop(
        "environment/taskcompendium-shared/shared_pytest.py"
    )
    files["environment/shellbox-images.json"] = (
        json.dumps({"main": {"image": image, "workdir": workdir}}, indent=2).encode() + b"\n"
    )
    task = tomlkit.parse(files["task.toml"].decode())
    task["environment"]["docker_image"] = image
    command = str(task["environment"]["setup"]["command"])
    head, marker, rest = command.partition("\n")
    if marker != "\n" or not head.startswith("set -euo pipefail"):
        raise ValueError("Unexpected hardened setup command shape")
    prefix = PROVISION_PYTHON if files.get("_provision_interpreter") else ""
    task["environment"]["setup"]["command"] = head + "\n" + prefix + LAYER_SETUP + rest
    files["task.toml"] = tomlkit.dumps(task).encode()


def row_variant_files(row: dict, variant: str, base_image: str) -> dict[str, bytes]:
    """One sampled row's hardened package for the declared variant."""
    flat = _flatten(row)
    base = (row.get("install_config") or {}).get("base_image_name") or groups.V1_PYTHON_BASE
    group = groups.GROUP_BY_BASE[base]
    upstream_files = task_files.task_files(flat)
    if variant == "upstream":
        # The upstream renderer's own task.toml (digest-pinned image, no task-start setup),
        # with the family config keys added so the same profile hardens both variants. The
        # vendored renderer declares only docker_image; the hardening contract requires a
        # Dockerfile, so the variant carries the minimal FROM for the row's pinned image.
        files = {name: content.encode() for name, content in upstream_files.items()}
        files["environment/Dockerfile"] = f"FROM {row['image_name']}\n".encode()
    elif variant == "condensed":
        setup = setup_command(flat, base, group)
        files = {
            "task.toml": task_toml(flat, base, group, setup).encode(),
            "environment/Dockerfile": dockerfile(group, base_image).encode(),
        }
        for name, content in upstream_files.items():
            if name == "task.toml":
                continue
            if name.startswith("solution/"):
                # The condensed variant's solve script runs in the uniform workspace; the
                # upstream variant keeps the row's original paths (its image bakes them).
                content = to_workspace(content, flat["workdir"])
            files[name] = content.encode()
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    task_document = tomlkit.parse(files["task.toml"])
    environment = task_document.setdefault("environment", tomlkit.table())
    for key, value in GATE_RESOURCES.items():
        environment[key] = value
    files["task.toml"] = tomlkit.dumps(task_document).encode()
    config = family_config(flat)
    if variant == "upstream":
        # The upstream digest image bakes the row's repo at its own workdir: the upstream
        # variant keeps the row's original paths (the /testbed rewrite belongs to the
        # condensed consolidation; ruling 2026-11-20).
        config["workdir"] = flat["workdir"]
        config["test_cmd"] = flat["test_cmd"]
    files["tests/config.json"] = json.dumps(config, indent=1, ensure_ascii=False).encode() + b"\n"
    harden_shared_source(files, profile=swe_rebench_profile(), dependency_state=SharedDependencyState.AGENT_MUTABLE)
    # The declared image: the row's own digest for the upstream variant, the group's shared
    # base for the condensed variant (both pull directly on Daytona; the Dockerfile remains
    # as provenance only). The upstream images carry no interpreter-provisioning layer, so the
    # gate provisions python >= 3.9 at setup for them (the adapter's controller requirement).
    declared = row["image_name"] if variant == "upstream" else base_image
    if variant == "upstream":
        files["_provision_interpreter"] = b"1"
    adapt_for_daytona(files, flat["workdir"] if variant == "upstream" else WORKSPACE, declared)
    files.pop("_provision_interpreter", None)
    # The trial must pull the declared image directly (the probe proves Daytona resolves the
    # internal ghcr package); a Dockerfile in the package would make the trial machinery build
    # the image Daytona-side, where the internal pull has no credentials. The Dockerfile stays
    # in the release package as provenance; the trial package drops it.
    files.pop("environment/Dockerfile", None)
    return files


def write_package(root: Path, name: str, files: dict[str, bytes], row: dict, variant: str, image: str) -> Path:
    """Materialize the package once per row+variant; the nop and oracle trials share it."""
    package = root / "packages" / name
    if package.exists():
        return package
    package.mkdir(parents=True)
    for relative, content in sorted(files.items()):
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {
        "task_id": row["instance_id"],
        "variant": variant,
        "declared_image": image,
        "upstream_revision": groups.UPSTREAM_REVISION,
        "files": {
            relative: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for relative, content in sorted(files.items())
        },
    }
    (root / "packages" / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return package


class OutcomeLedger:
    """Thread-safe per-trial outcomes with incremental S3 persistence and resume.

    Every completed trial lands in the ledger immediately (the coordinator requires partial
    records and early mismatch visibility, not an end-of-run dump). A trial whose outcome
    already carries a real reward is a completed trial and is never re-run.

    Memory discipline (the wave2-full OOM fix): the in-memory state holds ONLY the reward
    per key; each full outcome is APPENDED as one JSON line to ``{uri}.jsonl`` — never
    rewritten in full, never retained. Resume replays the canonical dict file plus the
    sidecar (last write wins per key). At close, the canonical dict file is re-consolidated
    by streaming the sidecar line by line.
    """

    CHECKPOINT_EVERY = 500

    def __init__(self, uri: str, local_dir: Path | None = None):
        self._uri = uri
        self._sidecar = uri + ".jsonl"
        self._lock = threading.Lock()
        self._rewards: dict[tuple[str, str, str], float] = {}
        self._records_since_checkpoint = 0
        # The sidecar appends to LOCAL disk (O(1) per record); s3 gets a checkpoint copy every
        # CHECKPOINT_EVERY records and at close. A crash loses only the unsynced tail — those
        # keys simply re-run on resume (per-row resume, no correctness loss).
        self._local_sidecar = (local_dir or Path("/tmp")) / "wave-outcomes.jsonl"
        self._handle = None
        for source in (uri, self._sidecar):
            try:
                raw = StoragePath(source).read_text()
            except Exception:
                continue
            if source == self._sidecar:
                entries = (item for line in raw.splitlines() if line.strip() for item in json.loads(line).items())
            else:
                entries = json.loads(raw).items()
            for key_str, outcome in entries:
                parts = key_str.split("|")
                if len(parts) != 3:
                    continue
                key = (parts[0], parts[1], parts[2])
                reward = ((outcome or {}).get("rewards") or {}).get("reward")
                if isinstance(reward, float):
                    self._rewards[key] = reward

    def completed(self, key: tuple[str, str, str]) -> bool:
        return key in self._rewards

    def reward(self, key: tuple[str, str, str]) -> dict:
        value = self._rewards.get(key)
        return {"reward": value} if value is not None else {}

    def record(self, key: tuple[str, str, str], outcome: dict) -> None:
        with self._lock:
            reward = ((outcome or {}).get("rewards") or {}).get("reward")
            if isinstance(reward, float):
                self._rewards[key] = reward
            if self._handle is None:
                # The fresh pod's output directory may not exist yet (the earlier wave
                # instance died exactly here); create it before the first append.
                self._local_sidecar.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self._local_sidecar.open("a", encoding="utf-8")
            self._handle.write(json.dumps({"|".join(key): outcome}, separators=(",", ":")) + "\n")
            self._handle.flush()
            self._records_since_checkpoint += 1
            if self._records_since_checkpoint >= self.CHECKPOINT_EVERY:
                self._checkpoint_locked()

    def _checkpoint_locked(self) -> None:
        if self._handle is not None:
            self._handle.flush()
        # The canonical dict file carries the MERGED rewards (every prior pod's durable
        # verdicts plus this pod's). A fresh pod starts with an empty local sidecar, so
        # checkpointing the pod-local sidecar alone would erase every prior pod's history
        # on each preemption (the wave-v11 churn stop lost ~3,700 graded rows this way).
        StoragePath(self._uri).write_text(
            json.dumps(
                {
                    "|".join(key): {"rewards": {"reward": reward}}
                    for key, reward in sorted(self._rewards.items())
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
        StoragePath(self._sidecar).write_text(self._local_sidecar.read_text())
        self._records_since_checkpoint = 0

    def close(self) -> None:
        with self._lock:
            self._checkpoint_locked()
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def consolidate(self) -> None:
        """Rebuild the canonical dict file from the sidecar, streaming line by line."""
        with self._lock:
            entries = []
            try:
                raw = StoragePath(self._sidecar).read_text()
            except Exception:
                raw = ""
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                entries.append(payload)
            StoragePath(self._uri).write_text(
                json.dumps({k: v for entry in entries for k, v in entry.items()}, indent=1) + "\n"
            )

    def all(self) -> dict[tuple[str, str, str], dict]:
        with self._lock:
            return {(k): {"rewards": {"reward": v}} for k, v in self._rewards.items()}


# Transient Daytona infrastructure failures a phase retry survives: the organization CPU
# limit saturating (sandbox creates fail with "Total CPU limit exceeded" and live sandboxes
# are reaped, surfacing as "Daytona environment is not running") and the connection-level
# provider errors. Grading verdicts and config errors are never retried. Each retry runs in
# a fresh attempt directory: the machinery creates a uniquely named sandbox per run, so a
# retry is a clean recreate — never a reattach to a reaped or stopped sandbox (idle Daytona
# sandboxes are purged across orgs; never assume one survives a long preemption).
TRANSIENT_RETRY_DELAYS_SEC = (60.0, 300.0)


def transient_daytona_text(text: str) -> bool:
    """Classify the retryable Daytona infrastructure failures."""
    return (
        "Daytona environment is not running" in text
        or "Total CPU limit exceeded" in text
        or "DaytonaConnectionError" in text
        or "DaytonaTimeoutError" in text
    )


def transient_daytona_error(error: BaseException) -> bool:
    return transient_daytona_text(f"{type(error).__name__}: {error}")


def trial(
    root: Path, row: dict, variant: str, phase: str, image: str, ledger: OutcomeLedger, *, force: bool = False
) -> dict:
    key = (row["instance_id"], variant, phase)
    if ledger.completed(key) and not force:
        LOGGER.info("[%s/%s] already completed; resuming", row["instance_id"], variant)
        return {"rewards": ledger.reward(key), "resumed": True}
    # Each phase gets its own package copy: the trial machinery's post-run artifact download
    # targets the task directory, so a shared package across phases lost files mid-run (the
    # nop phase's artifact pass removed task.toml before the oracle phase read it).
    name = f"{row['instance_id']}-{variant}-{phase}"
    package = write_package(root, name, row_variant_files(row, variant, image), row, variant, image)
    agent = {"name": "nop"} if phase == "nop" else {"name": "oracle"}
    summary = None
    last_error: Exception | None = None
    for retry_index, delay in enumerate((0.0, *TRANSIENT_RETRY_DELAYS_SEC)):
        if delay:
            LOGGER.warning(
                "[%s/%s] transient Daytona failure (attempt %d failed); retrying in %ds: %s",
                row["instance_id"],
                variant,
                retry_index,
                int(delay),
                last_error,
            )
            time.sleep(delay)
        attempt = root / "trials" / (f"{name}-{phase}" if not retry_index else f"{name}-{phase}-r{retry_index}")
        attempt.mkdir(parents=True, exist_ok=True)
        config = trial_config(package, attempt / "jobs", agent, f"{name}-{phase}")
        config_path = attempt / "trial.json"
        config_path.write_text(json.dumps(config))
        started = time.monotonic()
        try:
            summary = asyncio.run(
                asyncio.wait_for(_run_logged(config_path, attempt / "harbor.log"), timeout=ATTEMPT_TIMEOUT + 900)
            )
        except Exception as error:
            if not transient_daytona_error(error) or retry_index == len(TRANSIENT_RETRY_DELAYS_SEC):
                raise
            last_error = error
            continue
        # The machinery embeds provider failures in the summary instead of raising: a
        # transient Daytona failure recorded as reward=None must retry like a raised one.
        embedded = summary.get("exception") or {}
        embedded_text = f"{embedded.get('exception_type')}: {embedded.get('exception_message')}"
        if transient_daytona_text(embedded_text) and retry_index < len(TRANSIENT_RETRY_DELAYS_SEC):
            last_error = RuntimeError(embedded_text)
            summary = None
            continue
        break
    assert summary is not None  # the loop exits only through the break or the re-raise
    elapsed = time.monotonic() - started
    summary["elapsed_sec"] = elapsed
    summary["deadline_sensitive"] = phase == "oracle" and elapsed > VERIFIER_DEADLINE_SEC
    # Grading evidence: the verifier redirects the test_cmd output to a file in the sandbox and
    # the machinery downloads it to trial_dir/verifier/test-stdout.txt; inline its tail so the
    # ledger carries WHY a reward came out the way it did. The VerifierResult.stdout is the
    # wrapper's own minimal output (empty when the exec redirected to the file).
    verifier = summary.get("verifier") or {}
    for field in ("stdout", "stderr"):
        text = verifier.get(field)
        if text:
            summary[f"grading_{field}"] = text[-30000:]
    for candidate in sorted(attempt.rglob("test-stdout.txt")):
        text = candidate.read_text(errors="replace")
        if text.strip():
            summary["grading_stdout"] = text[-30000:]
            summary["grading_stdout_path"] = str(candidate.relative_to(attempt))
            break
    verifier_files = {}
    for candidate in sorted(attempt.rglob("verifier/*")):
        if candidate.is_file() and candidate.stat().st_size <= 20000:
            verifier_files[str(candidate.relative_to(attempt))] = candidate.read_text(errors="replace")[-20000:]
    if verifier_files:
        summary["verifier_files"] = verifier_files
    summary["trial_tree"] = [
        (str(p.relative_to(attempt)), p.stat().st_size) for p in sorted(attempt.rglob("*")) if p.is_file()
    ][:400]
    LOGGER.info(
        "[%s/%s] reward=%s elapsed=%.0fs grading_stdout=%s verifier_files=%s",
        name,
        phase,
        summary["rewards"].get("reward"),
        elapsed,
        len(summary.get("grading_stdout") or ""),
        sorted(verifier_files),
    )
    ledger.record(key, summary)
    return summary


async def _run_logged(config_path: Path, log_path: Path) -> dict:
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return await run_trial(config_path)


def compare(row: dict, upstream: dict, condensed: dict) -> dict:
    def outcomes(summary: dict) -> dict:
        # An errored trial has no reward; two errored variants are NOT a match — the gate
        # compares graded outcomes, and empty rewards prove nothing about either environment.
        # The exception dict is recorded verbatim so the real failure cause survives, and the
        # oracle elapsed time carries the CPU/deadline-sensitivity evidence the gate owes the
        # release decision.
        nop = summary.get("nop") or {}
        oracle = summary.get("oracle") or {}
        return {
            "nop_reward": (nop.get("rewards") or {}).get("reward"),
            "oracle_reward": (oracle.get("rewards") or {}).get("reward"),
            "oracle_elapsed_sec": oracle.get("elapsed_sec"),
            "oracle_deadline_sensitive": oracle.get("deadline_sensitive"),
            "nop_exception": nop.get("exception"),
            "oracle_exception": oracle.get("exception"),
            "nop_error": nop.get("error"),
            "oracle_error": oracle.get("error"),
        }

    up, con = outcomes(upstream), outcomes(condensed)
    complete = all(isinstance(up[key], float) and isinstance(con[key], float) for key in ("nop_reward", "oracle_reward"))
    return {
        "task_id": row["instance_id"],
        "group": groups.GROUP_BY_BASE[row_base(row)].group_id,
        "upstream": up,
        "condensed": con,
        "match": complete and up["nop_reward"] == con["nop_reward"] and up["oracle_reward"] == con["oracle_reward"],
        "complete": complete,
    }


def load_rows(dataset_dir: StoragePath) -> list[dict]:
    shards = sorted((dataset_dir / "data" / "train-*.parquet").glob(), key=str) or sorted(
        (dataset_dir / "train-*.parquet").glob(), key=str
    )
    rows: list[dict] = []
    for shard in shards:
        parquet = pq.ParquetFile(io.BytesIO(shard.read_bytes()))
        for batch in parquet.iter_batches(batch_size=2000):
            rows.extend(batch.to_pylist())
    if not rows:
        raise ValueError(f"No V2.1 parquet shards under {dataset_dir}")
    return rows


class CreatedSandboxRecorder:
    """Capture sandbox IDs this run's own trial logs recorded as CREATED.

    The shellbox Daytona machinery emits versioned lifecycle events (including `created` with
    the sandbox_id) through the `shellbox.backends.daytona.lifecycle` logger, in this same
    process. A logging handler collects exactly those IDs. Cleanup at run end deletes only
    these exact IDs — never selected by label, status or time window.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: dict[str, str | None] = {}
        self._machinery_deleted: set[str] = set()
        self._handler: logging.Handler | None = None

    def attach(self) -> None:
        recorder = self

        class Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    payload = json.loads(record.getMessage())
                except Exception:
                    return
                if not isinstance(payload, dict) or payload.get("kind") != "daytona_sandbox_lifecycle":
                    return
                event = payload.get("event")
                sandbox_id = payload.get("sandbox_id")
                if event not in ("created", "deleted") or not sandbox_id:
                    return
                with recorder._lock:
                    if event == "created":
                        recorder._ids[sandbox_id] = payload.get("requested_name")
                    else:
                        recorder._machinery_deleted.add(sandbox_id)

        self._handler = Handler(level=logging.INFO)
        logging.getLogger("shellbox.backends.daytona.lifecycle").addHandler(self._handler)
        logging.getLogger("shellbox.backends.daytona.lifecycle").setLevel(logging.INFO)

    def record(self, sandbox_id: str) -> None:
        with self._lock:
            self._ids.setdefault(sandbox_id, None)

    def created(self) -> dict[str, str | None]:
        with self._lock:
            return dict(self._ids)

    def machinery_deleted(self) -> list[str]:
        with self._lock:
            return sorted(self._machinery_deleted)

    def detach(self) -> None:
        if self._handler is not None:
            logging.getLogger("shellbox.backends.daytona.lifecycle").removeHandler(self._handler)
            self._handler = None


async def delete_recorded_sandboxes(recorder: CreatedSandboxRecorder) -> list[str]:
    """Delete exactly the sandbox IDs this run's own trial logs recorded as created.

    The machinery deletes each sandbox at trial teardown (lifecycle event ``deleted``); IDs
    already recorded as deleted are skipped — only still-alive created IDs are deleted here.
    """
    created = recorder.created()
    machinery_deleted = set(recorder.machinery_deleted())
    pending = {sid: name for sid, name in created.items() if sid not in machinery_deleted}
    if not pending:
        return []

    client = AsyncDaytona(
        DaytonaConfig(
            api_key=os.environ["DAYTONA_API_KEY"],
            api_url=os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
            target=os.environ.get("DAYTONA_TARGET", "us"),
        )
    )
    deleted = []
    for sandbox_id in sorted(pending):
        try:
            target = await client.get(sandbox_id)
            await target.delete()
            deleted.append(sandbox_id)
            LOGGER.info("owned sandbox deleted by exact ID: %s (requested_name %s)", sandbox_id, created[sandbox_id])
        except Exception as error:
            LOGGER.error("owned sandbox %s cleanup failed: %s", sandbox_id, error)
    return deleted


async def _create_diagnose_sandbox(client, base_image: str):
    """Create the diagnose sandbox the trials' own way: DECLARATIVE image provisioning, root.

    A raw ``CreateSandboxBaseParams(image=...)`` carries no image field at all (silently
    ignored — Daytona started its default image as a non-root user and every capture was
    useless). Snapshot provisioning is unavailable here: the trials run declarative and create
    no snapshots, and the organization's snapshot quota is exhausted — the declarative path is
    the same pull the trials perform, including its registry credentials.
    """

    return await client.create(
        CreateSandboxFromImageParams(
            image=base_image,
            resources=Resources(cpu=4, memory=8, disk=10),
            os_user="root",
            labels={"purpose": "swe-repair-fidelity-diagnose", "owner": "swe-repair"},
            auto_stop_interval=0,
            ttl_minutes=360,
        )
    )


async def diagnose_condensed(row: dict, base_image: str, output_uri: str, ledger_dir: Path) -> None:
    """Reproduce the condensed variant's setup, oracle and grading in ONE sandbox with full
    output capture — the diagnostic behind a condensed-oracle failure (the trial machinery
    discards the setup output that `|| true` masks)."""
    flat = _flatten(row)
    base = (row.get("install_config") or {}).get("base_image_name") or groups.V1_PYTHON_BASE
    group = groups.GROUP_BY_BASE[base]
    log: list[str] = []

    async def run(client, sandbox_id: str, label: str, command: str, *, cwd: str | None = None, timeout: int = 1800):
        response = await (await client.get(sandbox_id)).process.exec(command, cwd=cwd, timeout=timeout)
        text = f"===== {label} (exit {response.exit_code}) =====\n{response.result}"
        log.append(text)
        LOGGER.info("diagnose %s exit=%s (%d chars)", label, response.exit_code, len(response.result or ""))
        return response

    trial_environment()  # validates the secret is present

    client = AsyncDaytona(
        DaytonaConfig(
            api_key=os.environ["DAYTONA_API_KEY"],
            api_url=os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
            target=os.environ.get("DAYTONA_TARGET", "us"),
        )
    )
    sandbox = await _create_diagnose_sandbox(client, base_image)
    sandbox_id = sandbox.id
    LOGGER.info("diagnostic sandbox created: %s", sandbox_id)
    try:
        setup = setup_command(flat, base, group)
        await run(client, sandbox_id, "source setup (verbatim)", setup)
        await run(client, sandbox_id, "solution dir upload check", "mkdir -p /solution /tmp/diagnose")
        gold = flat["patch"]
        await run(
            client, sandbox_id, "write gold patch", "cat > /solution/gold.patch <<'GOLD_EOF'\n" + gold + "\nGOLD_EOF"
        )
        await run(
            client,
            sandbox_id,
            "apply gold patch",
            f"cd {WORKSPACE} && git apply --verbose /solution/gold.patch || "
            "git apply --verbose --ignore-space-change --ignore-whitespace /solution/gold.patch || "
            "patch --batch --fuzz=5 -p1 -i /solution/gold.patch || "
            "git apply --verbose --reject --ignore-whitespace /solution/gold.patch",
        )
        restore = "\n".join(
            f"if git ls-files --error-unmatch {shlex.quote(f)} >/dev/null 2>&1; "
            f"then git restore {shlex.quote(f)} 2>/dev/null; else rm -f {shlex.quote(f)}; fi"
            for f in flat["test_files"]
        )
        await run(client, sandbox_id, "restore test files", f"cd {WORKSPACE}\n{restore}")
        await run(
            client,
            sandbox_id,
            "write test patch",
            "cat > /tmp/test.patch <<'TEST_EOF'\n" + flat["test_patch"] + "\nTEST_EOF",
        )
        await run(
            client,
            sandbox_id,
            "apply test patch",
            f"cd {WORKSPACE} && git apply --verbose /tmp/test.patch || " "patch --batch --fuzz=5 -p1 -i /tmp/test.patch",
        )
        test_cmd = "cd {workdir} && timeout -k 10 1800 bash -lc {cmd}".format(
            workdir=WORKSPACE, cmd=shlex.quote(to_workspace(flat["test_cmd"], flat["workdir"]))
        )
        response = await run(client, sandbox_id, "grading test_cmd", test_cmd, timeout=1900)
        StoragePath(f"{output_uri}/diagnose-{row['instance_id']}.txt").write_text("\n".join(log) + "\n")
        LOGGER.info("diagnostic capture written for %s (grading exit %s)", row["instance_id"], response.exit_code)
    finally:
        try:
            target = await client.get(sandbox_id)
            await target.delete()
            LOGGER.info("diagnostic sandbox deleted by exact ID: %s", sandbox_id)
        except Exception as cleanup_error:
            LOGGER.error("diagnostic sandbox %s cleanup failed: %s", sandbox_id, cleanup_error)


def trial_row(root: Path, row: dict, image: str, ledger: OutcomeLedger, *, force: bool = False) -> dict:
    """Run one row's four trials SEQUENTIALLY on one worker.

    The nop and oracle packages of a row+variant are identical, so their Daytona sandbox names
    are identical; submitting them concurrently makes the two phases race on ONE sandbox
    (removing each other's working directory mid-clone). Serializing a row's trials while
    running rows in parallel keeps the 8-way capacity grant without the race.
    """
    summaries = {}
    for variant, phase in (("upstream", "nop"), ("upstream", "oracle"), ("condensed", "nop"), ("condensed", "oracle")):
        summaries[(variant, phase)] = trial(root, row, variant, phase, image, ledger, force=force)
    return summaries


def wave_row(root: Path, row: dict, image: str, ledger: OutcomeLedger) -> dict:
    """Wave-1 admission run: condensed nop + condensed oracle for one population row."""
    summaries = {}
    for phase in ("nop", "oracle"):
        summaries[phase] = trial(root, row, "condensed", phase, image, ledger)
    return summaries


CHURN_WINDOW = 50
CHURN_THRESHOLD = 0.2


class QuotaInfraStop(RuntimeError):
    """The typed fail-fast stop: the rolling trial churn exceeded the threshold."""


class ChurnMonitor:
    """The rolling reward=None/infra-error share over the last CHURN_WINDOW completions.

    Ruling 2026-11-20 (the wave-v9 churn): when more than about 20% of the trials in the
    rolling window end with reward=None or an infra error, the driver stops submitting,
    logs a typed QUOTA/INFRA stop, and exits non-zero so the watchdog catches saturation
    within minutes instead of hours.
    """

    def __init__(self, window: int = CHURN_WINDOW, threshold: float = CHURN_THRESHOLD):
        self._window = window
        self._threshold = threshold
        self._outcomes: deque[bool] = deque()
        self._failed = 0

    def record(self, failed: bool) -> None:
        self._outcomes.append(failed)
        self._failed += failed
        if len(self._outcomes) > self._window:
            self._failed -= self._outcomes.popleft()

    def tripped(self) -> bool:
        return len(self._outcomes) >= self._window and self._failed / len(self._outcomes) > self._threshold

    def describe(self) -> str:
        return f"{self._failed}/{len(self._outcomes)} of the last {len(self._outcomes)} trials ended reward=None/infra"


def run_wave(
    root: Path,
    rows_by_id: dict[str, dict],
    bindings: dict,
    output_uri: str,
) -> None:
    """Wave-1 full-population admission run for the declared groups (condensed only).

    Each kept row must score NoP 0.0 and oracle 1.0 in the converted environment to be a
    valid RL task; anything else is recorded with its evidence and left out of admission.
    """
    wave_groups = [g for g in os.environ.get("WAVE_GROUPS", "").split(",") if g]
    exclusions = set()
    exclusions_uri = os.environ.get("WAVE_EXCLUSIONS_URI", "")
    if exclusions_uri:
        exclusions = set(json.loads(StoragePath(exclusions_uri).read_text()))
    limit = int(os.environ.get("WAVE_LIMIT", "0"))
    ledger = OutcomeLedger(f"{output_uri}/fidelity-wave1-outcomes.json", local_dir=root)
    recorder = CreatedSandboxRecorder()
    recorder.attach()
    LOGGER.info("wave-1 groups=%s exclusions=%d limit=%d", wave_groups, len(exclusions), limit)
    counts: dict[str, dict[str, int]] = {}
    invalid: list[dict] = []
    scheduled = 0
    _start_rss_logger()
    try:
        with ThreadPoolExecutor(max_workers=TRIAL_WORKERS) as pool:
            future_to_row = {}
            for group_id in sorted(wave_groups):
                image = bindings["images"][group_id]
                members = sorted(
                    instance_id
                    for instance_id, row in rows_by_id.items()
                    if _wave_group_id(row) == group_id and instance_id not in exclusions
                )
                for instance_id in members:
                    if limit and scheduled >= limit:
                        break
                    scheduled += 1
                    future_to_row[pool.submit(wave_row, root, rows_by_id[instance_id], image, ledger)] = (
                        group_id,
                        instance_id,
                    )
            # Process futures AS THEY COMPLETE and drop each result immediately: retaining
            # completed futures (with their full summaries) until run end was the wave2-full
            # OOM at 375 workers. The ledger holds the sealed per-row evidence on disk.

            churn = ChurnMonitor()
            for future in as_completed(future_to_row):
                group_id, instance_id = future_to_row.pop(future)
                counts.setdefault(group_id, {"done": 0, "valid": 0, "invalid": 0})
                try:
                    summaries = future.result()
                except Exception as error:
                    LOGGER.error("wave row %s failed: %s", instance_id, error)
                    counts[group_id]["invalid"] += 1
                    invalid.append({"task_id": instance_id, "group": group_id, "error": str(error)[:400]})
                    churn.record(True)
                    if churn.tripped():
                        raise QuotaInfraStop(f"QUOTA/INFRA STOP: {churn.describe()}; submissions stopped") from error
                    continue
                nop = (summaries["nop"].get("rewards") or {}).get("reward")
                oracle = (summaries["oracle"].get("rewards") or {}).get("reward")
                churn.record(nop is None or oracle is None)
                if churn.tripped():
                    for pending in future_to_row:
                        pending.cancel()
                    raise QuotaInfraStop(f"QUOTA/INFRA STOP: {churn.describe()}; submissions stopped")
                counts[group_id]["done"] += 1
                if nop == 0.0 and oracle == 1.0:
                    counts[group_id]["valid"] += 1
                else:
                    counts[group_id]["invalid"] += 1
                    invalid.append(
                        {
                            "task_id": instance_id,
                            "group": group_id,
                            "nop_reward": nop,
                            "oracle_reward": oracle,
                            "nop_error": str((summaries["nop"].get("exception") or {}).get("exception_message") or "")[
                                :300
                            ],
                            "oracle_error": str(
                                (summaries["oracle"].get("exception") or {}).get("exception_message") or ""
                            )[:300],
                        }
                    )
                del summaries
    finally:
        ledger.close()
    recorder.detach()
    StoragePath(f"{output_uri}/fidelity-wave1-verdicts.json").write_text(
        json.dumps({"counts": counts, "invalid": invalid, "total_scheduled": scheduled}, indent=2) + "\n"
    )
    LOGGER.info("wave-1 verdicts: %s (scheduled=%d, invalid=%d)", json.dumps(counts), scheduled, len(invalid))
    try:
        created = recorder.created()
        machinery_deleted = recorder.machinery_deleted()
        StoragePath(f"{output_uri}/created-sandboxes.json").write_text(
            json.dumps(
                {"created": created, "deleted_by_machinery": machinery_deleted, "deleted_by_this_run": None}, indent=1
            )
            + "\n"
        )
        deleted = asyncio.run(delete_recorded_sandboxes(recorder))
        StoragePath(f"{output_uri}/created-sandboxes.json").write_text(
            json.dumps(
                {"created": created, "deleted_by_machinery": machinery_deleted, "deleted_by_this_run": deleted}, indent=1
            )
            + "\n"
        )
        LOGGER.info(
            "wave owned-sandbox cleanup: %d created, %d machinery-deleted, %d deleted",
            len(created),
            len(machinery_deleted),
            len(deleted),
        )
    except Exception as error:
        LOGGER.error("wave owned-sandbox cleanup failed: %s", error)


def _start_rss_logger() -> None:
    """Log the driver task's peak RSS every 60s (Linux ru_maxrss is KiB) — the sizing
    evidence for the wave runs' driver-task memory. Daemon thread; ends with the process."""

    def loop():
        while True:
            peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            LOGGER.info("driver rss: peak_rss_kib=%d", peak_kib)
            time.sleep(60)

    threading.Thread(target=loop, daemon=True).start()


def _wave_group_id(row: dict) -> str | None:
    try:
        return groups.GROUP_BY_BASE[row_base(row)].group_id
    except Exception:
        return None


def main() -> None:
    configure_coreweave_s3()
    sample_uri = os.environ["GATE_SAMPLE_URI"]
    bindings_uri = os.environ["GATE_BINDINGS_URI"]
    dataset_uri = os.environ["GATE_DATASET_URI"]
    output_uri = os.environ["GATE_OUTPUT_URI"]

    root = Path(os.environ["IRIS_OUTPUT_DIR"]) / "gate"
    manifest = json.loads(StoragePath(sample_uri).read_text())
    bindings = json.loads(StoragePath(bindings_uri).read_text())
    rows_by_id = {}
    for row in load_rows(StoragePath(dataset_uri)):
        rows_by_id[row["instance_id"]] = row
    missing = {
        instance_id
        for members in manifest["groups"].values()
        for instance_id in members
        if instance_id not in rows_by_id
    }
    if missing:
        raise ValueError(f"Sample references rows absent from the loaded population: {sorted(missing)[:5]}")

    only_groups = [group for group in os.environ.get("GATE_GROUPS", "").split(",") if group]
    wave_groups = [group for group in os.environ.get("WAVE_GROUPS", "").split(",") if group]
    only_instances = [i for i in os.environ.get("GATE_INSTANCE", "").split(",") if i]
    ledger = OutcomeLedger(f"{output_uri}/fidelity-trial-outcomes.json")
    if only_instances:
        LOGGER.info("canary mode: instances %s", only_instances)
    if wave_groups:
        run_wave(root, rows_by_id, bindings, output_uri)
        return
    recorder = CreatedSandboxRecorder()
    recorder.attach()
    if os.environ.get("GATE_DIAGNOSTIC") == "1" and only_instances:
        # Canary diagnostics: reproduce each condensed variant in ONE sandbox with full output
        # capture. No full gate; the diagnostic artifacts + sandbox proof are written and the
        # run exits 0.
        for instance_id in only_instances:
            row = rows_by_id[instance_id]
            base = (row.get("install_config") or {}).get("base_image_name") or groups.V1_PYTHON_BASE
            group = groups.GROUP_BY_BASE[base]
            LOGGER.info("diagnostic canary: %s", instance_id)
            asyncio.run(diagnose_condensed(row, bindings["images"][group.group_id], output_uri, root))
        recorder.detach()
        created = recorder.created()
        machinery_deleted = recorder.machinery_deleted()
        StoragePath(f"{output_uri}/created-sandboxes.json").write_text(
            json.dumps(
                {"created": created, "deleted_by_machinery": machinery_deleted, "deleted_by_this_run": None}, indent=1
            )
            + "\n"
        )
        try:
            deleted = asyncio.run(delete_recorded_sandboxes(recorder))
            StoragePath(f"{output_uri}/created-sandboxes.json").write_text(
                json.dumps(
                    {"created": created, "deleted_by_machinery": machinery_deleted, "deleted_by_this_run": deleted},
                    indent=1,
                )
                + "\n"
            )
        except Exception as error:
            LOGGER.error("owned-sandbox cleanup failed: %s", error)
        return
    records = []
    with ThreadPoolExecutor(max_workers=TRIAL_WORKERS) as pool:
        futures = {}
        for group_id, members in sorted(manifest["groups"].items()):
            if only_groups and group_id not in only_groups:
                continue
            image = bindings["images"][group_id]
            for instance_id in members:
                if only_instances and instance_id not in only_instances:
                    continue
                row = rows_by_id[instance_id]
                # One future per ROW: the row's four trials run sequentially on one worker
                # (identical packages for nop/oracle share a deterministic sandbox name, so
                # concurrent phases would race on one sandbox). Rows run in parallel.
                futures[instance_id] = pool.submit(trial_row, root, row, image, ledger, force=bool(only_instances))
        results = {}
        for instance_id, future in futures.items():
            try:
                results[instance_id] = future.result()
            except Exception as error:
                LOGGER.error("row %s trials failed: %s", instance_id, error)
                results[instance_id] = {
                    (variant, phase): {"rewards": {}, "error": str(error)}
                    for variant, phase in (
                        ("upstream", "nop"),
                        ("upstream", "oracle"),
                        ("condensed", "nop"),
                        ("condensed", "oracle"),
                    )
                }
        for instance_id in sorted(results):
            summaries = results[instance_id]
            upstream = {
                "nop": summaries[("upstream", "nop")],
                "oracle": summaries[("upstream", "oracle")],
            }
            condensed = {
                "nop": summaries[("condensed", "nop")],
                "oracle": summaries[("condensed", "oracle")],
            }
            records.append(compare(rows_by_id[instance_id], upstream, condensed))
        records_uri = StoragePath(f"{output_uri}/fidelity-records.json")
        records_uri.write_text(json.dumps(records, indent=2) + "\n")
    by_group: dict[str, dict[str, int]] = {}
    for record in records:
        verdict = by_group.setdefault(record["group"], {"tasks": 0, "match": 0, "mismatch": 0})
        verdict["tasks"] += 1
        verdict["match" if record["match"] else "mismatch"] += 1
    recorder.detach()
    # Cleanup: the machinery deletes its own sandboxes at trial teardown (lifecycle events
    # `deleted`); this run additionally deletes any still-alive created IDs by exact ID, and
    # writes the created/deleted ledger as the durable proof of ownership.
    created = recorder.created()
    machinery_deleted = recorder.machinery_deleted()
    StoragePath(f"{output_uri}/created-sandboxes.json").write_text(
        json.dumps(
            {"created": created, "deleted_by_machinery": machinery_deleted, "deleted_by_this_run": None}, indent=1
        )
        + "\n"
    )
    try:
        deleted = asyncio.run(delete_recorded_sandboxes(recorder))
        StoragePath(f"{output_uri}/created-sandboxes.json").write_text(
            json.dumps(
                {"created": created, "deleted_by_machinery": machinery_deleted, "deleted_by_this_run": deleted}, indent=1
            )
            + "\n"
        )
        LOGGER.info(
            "owned-sandbox cleanup: %d created, %d already deleted by the machinery, %d deleted by exact ID",
            len(created),
            len(machinery_deleted),
            len(deleted),
        )
    except Exception as error:
        LOGGER.error("owned-sandbox cleanup failed: %s", error)
    LOGGER.info("fidelity verdicts per group: %s", json.dumps(by_group))
    StoragePath(f"{output_uri}/fidelity-verdicts.json").write_text(json.dumps(by_group, indent=2) + "\n")
    # Mismatches are DATA, not a job failure: the gate exits 0 whenever it completed and wrote
    # its verdicts; the per-group ruling happens outside the run.
    mismatches = [record for record in records if not record["match"]]
    StoragePath(f"{output_uri}/fidelity-mismatches.json").write_text(json.dumps(mismatches, indent=2) + "\n")
    if mismatches:
        LOGGER.warning("%d rows mismatch; verdicts and mismatch records written", len(mismatches))


if __name__ == "__main__":
    main()
