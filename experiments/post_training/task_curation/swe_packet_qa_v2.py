# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Six-row corrected SWE QA consuming the frozen v2026.11.16.3 TaskSpec packets.

Per the ingest lane's sealed freeze map (9bf3e1f0..., seal d1a2145e): download
each row's actual frozen TaskSpec JSON, verify its SHA, reconstruct the native
packet directory verbatim from the TaskSpec resources, and run the three real
trials (nop, oracle, tamper) on those exact packet bytes through the Daytona
EVAL declarative route. Peak two physical sandboxes; bounded row batches; no
package reconstruction from a newer checkout. A blocked row records its
error and the run continues; completed rows' records upload incrementally so
a mid-run abort preserves evidence.
"""

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import shlex
import tarfile
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from harbor.agents import factory as agent_factory
from harbor.agents.base import BaseAgent
from harbor.host_capabilities import host_capabilities_from_environment
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.secrets import get_secret

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("packet-qa-v2")

DEFAULT_MAP_URI = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.16.1/swe-v3-qa-freeze-v1/qa-map.json"
DEFAULT_MAP_SHA256 = "9bf3e1f034ef6d3d940c160730491600694bf396c720aea604c6a0df7bda19b9"
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/shared-grader-adapters/2026.11.16.1"
ATTEMPT_TIMEOUT = 4 * 3600.0

TAMPER_SCRIPT = r"""
set -uo pipefail
python3 - <<'EOF'
import os, pathlib, subprocess
assert os.getuid() != 0, "the hardened package left the agent as root"
try:
    next(pathlib.Path('/opt/taskcompendium-shared').iterdir())
    raise SystemExit('agent listed the protected root')
except PermissionError:
    pass
except OSError as error:
    raise SystemExit(f'unexpected protected-root error: {error}')
if os.path.exists('/setup_files/taskcompendium-source-input.zlib'):
    raise SystemExit('private payload archive survived bootstrap')
if os.path.exists('/opt/taskcompendium-shared/materialize.patch'):
    raise SystemExit('materialization patch survived')
subprocess.run(['rm', '-rf', '/logs/verifier'], check=False)
subprocess.run(['mkdir', '-p', '/logs/verifier/reward.txt'], capture_output=True)
for command in (
    ['git', 'add', '-A'],
    ['git', '-c', 'user.email=tamper@example.test', '-c', 'user.name=tamper', 'commit', '-qm', 'tamper'],
):
    subprocess.run(command, capture_output=True)
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
if head:
    subprocess.run(['git', 'replace', '--graft', head], capture_output=True)
print('tamper-ready', head)
EOF
exit 0
"""

# Per-thread probe state: the tamper agent and the owning row's loop share the
# worker thread, so a threading.local cannot be crossed by a sibling row.
_probe = threading.local()


TAMPER_FAMILIES = frozenset({"scale", "lego"})


def row_phases(family: str) -> tuple[tuple[str, dict], ...]:
    """Functional QA contract per family.

    RST/Multi rows run nop + the original TaskSpec oracle through TaskSpecTrial
    (no family tamper tooling); Scale/Lego keep the declared three-phase probe.
    """
    oracle_agent = {"name": "oracle"}
    if family not in ("rst", "multi"):
        oracle_agent["kwargs"] = {"resource_usage_filename": "resource-usage.json"}
    phases: list[tuple[str, dict]] = [("nop", {"name": "nop"}), ("oracle", oracle_agent)]
    if family in TAMPER_FAMILIES:
        phases.append(("tamper", {"name": "tamper"}))
    return tuple(phases)


# Per-phase budget objects: the nop phase uses the nop-calibrated pair, the
# oracle phase the passing-original pair.
PHASE_BUDGET_KEYS = {
    "nop": ("nop_execution_requirements", "nop_verifier_execution_requirements"),
    "oracle": ("actor_execution_requirements", "verifier_execution_requirements"),
}


def fetch_row_runtime(root: Path, row: dict) -> dict:
    """Download the row's sealed source-runtime binding and verify its exact SHA.

    The binding (TaskSpecRuntimeBinding: task_sha256, runtime_sha256, runtime)
    is consumed through TaskSpecTrial's supported source_runtime field, which
    applies the original per-role resource limits and network declarations to
    the executed machines.
    """
    uri, expected = row["row_runtime_uri"], row["row_runtime_sha256"]
    path = artifact_path(root, row, "source-runtime", uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    copy(uri, str(path), no_clobber=True)
    digest = sha256_bytes(path.read_bytes())
    if digest != expected:
        raise ValueError(f"row runtime hash mismatch for {row['instance_id']}: {digest} != {expected}")
    binding = json.loads(path.read_text())
    missing = {"task_sha256", "runtime_sha256", "runtime"} - set(binding)
    if missing:
        raise ValueError(f"row runtime binding for {row['instance_id']} lacks {sorted(missing)}")
    return binding


def apply_phase_requirements(config: dict, row: dict, phase: str, binding: dict) -> None:
    """Apply the phase's maintained requirement fields to the trial config.

    The executor rejects timeout overrides for TaskSpec payloads (deadlines
    come from the TaskSpec shape), so the maintained application routes are:
    the source_runtime binding (original per-role resource limits and network
    declarations) and config.task.verifier_execution_requirements (validated
    by the executor against the TaskSpec allocation and the source runtime).
    Actor-side allocations ride the packet's frozen task environment plus the
    bound runtime; the map's actor budgets are recorded for the resolved-
    allocation cross-check and any mismatch is flagged, never forced.
    """
    config["task"]["source_runtime"] = binding
    _, verifier_key = PHASE_BUDGET_KEYS[phase]
    verifier_budgets = row.get(verifier_key)
    if verifier_budgets is not None:
        config["task"]["verifier_execution_requirements"] = verifier_budgets


def attempt_guard(row: dict) -> float:
    """Outer QA guard: never below the applied original phase limits.

    The guard caps the QA attempt only; it never loosens an original limit.
    """
    deadlines = []
    for key in (*PHASE_BUDGET_KEYS["nop"], *PHASE_BUDGET_KEYS["oracle"]):
        budgets = row.get(key) or {}
        for field in ("agent_timeout_sec", "verifier_timeout_sec"):
            value = budgets.get(field)
            if value is not None:
                deadlines.append(float(value))
    return max(ATTEMPT_TIMEOUT, max(deadlines, default=0.0) + 1800.0)


def crosscheck_allocations(root: Path, family: str, row: dict, phase: str) -> dict:
    """Original-vs-resolved role allocation cross-check for one phase.

    Actor allocations and deadlines are read from the on-disk frozen packet
    (task.toml environment and TaskSpec execution_requirements — exactly what
    the executor consumes); verifier allocations from the trial result's
    resolved config. Both must equal the map's calibrated budgets; any
    mismatch flags the row as a fidelity finding instead of passing silently.
    """
    actor_key, verifier_key = PHASE_BUDGET_KEYS[phase]
    actor_budget = row.get(actor_key) or {}
    verifier_budget = row.get(verifier_key) or {}
    if not actor_budget and not verifier_budget:
        return {"ok": True, "skipped": "no map budgets for this family"}
    packet = root / "packets" / f"{family}-{row['instance_id']}"
    environment = tomllib.loads((packet / "task.toml").read_text()).get("environment", {})
    execution = json.loads((packet / "task-spec.json").read_text()).get("execution_requirements") or {}
    result_path = (
        root
        / "trials"
        / f"{family}-{row['instance_id']}-{phase}"
        / "jobs"
        / f"{family}-{row['instance_id']}-{phase}"
        / "result.json"
    )
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    resolved_verifier = ((result.get("config") or {}).get("task") or {}).get("verifier_execution_requirements")
    checks = {}
    for name in ("cpus", "memory_mb", "storage_mb"):
        expected, resolved = actor_budget.get(name), environment.get(name)
        checks[f"actor_{name}"] = {"expected": expected, "resolved": resolved, "ok": expected == resolved}
    for name in ("agent_timeout_sec", "verifier_timeout_sec"):
        # The canonical phase deadlines are the executor's resolved shape:
        # TaskSpecTrial builds shape from the native task.toml agent/verifier
        # timeouts and raises unless the verifier shape matches, so the
        # persisted config.task.verifier_execution_requirements is the
        # executor-validated value. The baseline task-spec.json
        # execution_requirements is a different field (None on Multi rows)
        # and is informational only.
        expected = actor_budget.get(name)
        resolved = (resolved_verifier or {}).get(name)
        checks[f"deadline_{name}"] = {"expected": expected, "resolved": resolved, "ok": expected == resolved}
    checks["baseline_spec_execution_requirements"] = {
        "resolved": execution,
        "note": "informational; not the deadline source for native task.toml trials",
    }
    for name in ("cpus", "memory_mb", "storage_mb", "agent_timeout_sec", "verifier_timeout_sec"):
        expected, resolved = verifier_budget.get(name), (resolved_verifier or {}).get(name)
        checks[f"verifier_{name}"] = {"expected": expected, "resolved": resolved, "ok": expected == resolved}
    baseline = checks.pop("baseline_spec_execution_requirements")
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
        "baseline_spec_execution_requirements": baseline,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_map(root: Path, map_uri: str, map_sha256: str) -> dict:
    path = root / "qa-map.json"
    copy(map_uri, str(path), no_clobber=True)
    digest = sha256_bytes(path.read_bytes())
    if digest != map_sha256:
        raise ValueError(f"qa map hash mismatch: {digest}")
    return json.loads(path.read_text())


def artifact_path(root: Path, row: dict, kind: str, uri: str) -> Path:
    """Row-unique local path for a sealed artifact download; sealed URIs share basenames across rows."""
    suffix = PurePosixPath(uri).suffix or ".bin"
    return root / "packets" / f"{row['instance_id']}-{kind}{suffix}"


def fetch_task_spec_bytes(root: Path, row: dict) -> bytes:
    """Download the row's actual frozen TaskSpec and verify its exact SHA."""
    path = artifact_path(root, row, "task-spec", row["task_spec_uri"])
    path.parent.mkdir(parents=True, exist_ok=True)
    copy(row["task_spec_uri"], str(path), no_clobber=True)
    digest = sha256_bytes(path.read_bytes())
    if digest != row["task_spec_sha256"]:
        raise ValueError(f"TaskSpec hash mismatch for {row['instance_id']}: {digest} != {row['task_spec_sha256']}")
    return path.read_bytes()


def packet_from_package(root: Path, family: str, row: dict, task_spec_bytes: bytes) -> Path:
    """Unpack the frozen passing prepared-package archive verbatim and verify it.

    The archive carries the production calibrated task.toml (with the final
    native actor allocations), task-spec.json and source-runtime.json; the
    trial reads everything from disk, which applies the exact calibrated
    budgets without changing TaskSpec bytes or using forbidden overrides.
    The unpacked bytes are checked against the sealed TaskSpec and runtime
    evidence and against the original consumed task_checksum via the
    maintained Harbor inspection (Task.checksum).
    """
    from harbor.models.task.task import Task  # noqa: PLC0415 - optional heavy import, QA host only

    name = row["instance_id"]
    directory = root / "packets" / f"{family}-{name}"
    if directory.exists():
        raise FileExistsError(directory)
    archive_path = artifact_path(root, row, "package", row["package_uri"])
    copy(row["package_uri"], str(archive_path), no_clobber=True)
    digest = sha256_bytes(archive_path.read_bytes())
    if digest != row["package_sha256"]:
        raise ValueError(f"package hash mismatch for {name}: {digest} != {row['package_sha256']}")
    directory.mkdir(parents=True)
    with tarfile.open(archive_path) as bundle:
        for member in bundle.getmembers():
            target = (directory / member.name).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError(f"package member escapes the extraction root: {member.name}")
        bundle.extractall(directory, filter="tar")
    spec_path = directory / "task-spec.json"
    if spec_path.read_bytes() != task_spec_bytes:
        raise ValueError(f"archive task-spec.json differs from the sealed TaskSpec evidence for {name}")
    if row.get("row_runtime_uri"):
        runtime_path = directory / "source-runtime.json"
        runtime_bytes = artifact_path(root, row, "source-runtime", row["row_runtime_uri"])
        if not runtime_bytes.exists():
            fetch_row_runtime(root, row)
        if runtime_path.read_bytes() != runtime_bytes.read_bytes():
            raise ValueError(f"archive source-runtime.json differs from the sealed runtime evidence for {name}")
    checksum = Task(task_dir=directory).checksum
    expected_checksum = row.get("task_checksum")
    if not expected_checksum:
        raise ValueError(f"row {name} lacks the original consumed task_checksum")
    if checksum != expected_checksum:
        raise ValueError(f"task_checksum mismatch for {name}: {checksum} != {expected_checksum}")
    return directory


def packet_from_task_spec(directory: Path, task_spec: dict) -> Path:
    """Reconstruct the native packet directory verbatim from TaskSpec resources."""
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    for role in ("verifier", "oracle"):
        for resource in task_spec["resources"][role]:
            content = base64.b64decode(resource["source"]["content_base64"], validate=True)
            target = directory / resource["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            mode = resource.get("mode")
            if mode:
                os.chmod(target, int(mode, 8))
    manifest = {
        "task_id": task_spec["id"],
        "packet_source": "frozen TaskSpec packet bytes (map URI verified)",
        "files": {
            resource["path"]: sha256_bytes(base64.b64decode(resource["source"]["content_base64"], validate=True))
            for role in ("verifier", "oracle")
            for resource in task_spec["resources"][role]
        },
    }
    (directory / "packet-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return directory


async def run_trial(config_path: Path) -> dict:
    class TamperAgent(BaseAgent):
        """The QA's declared tamper probe, registered as a scripted agent."""

        SUPPORTS_WINDOWS = False

        @staticmethod
        def name() -> str:
            return "tamper"

        def version(self) -> str:
            return "qa-2"

        async def setup(self, environment) -> None:
            pass

        async def run(self, instruction, environment, context) -> None:
            probe = await environment.exec(
                command="python -c 'import sys; print(sys.version_info[:3])'", timeout_sec=120
            )
            _probe.testbed_python = probe.stdout.strip()
            result = await environment.exec(command=f"bash -c {shlex.quote(TAMPER_SCRIPT)}", timeout_sec=900)
            if result.return_code != 0:
                raise RuntimeError(f"tamper probe failed inside the agent phase: {result.stdout[-2000:]}")

    agent_factory.AgentFactory._AGENT_MAP["tamper"] = TamperAgent
    config = TrialConfig.model_validate_json(config_path.read_text())
    trial = await Trial.create(
        config, host_capabilities=host_capabilities_from_environment(os.environ, model_authentication={})
    )
    result = await trial.run()
    rewards = dict(result.verifier_result.rewards) if result.verifier_result else {}
    return {
        "rewards": rewards,
        "completion_recorded": result.completion is not None,
        "exception": result.exception_info.model_dump(mode="json") if result.exception_info else None,
    }


def trial_config(packet: Path, trials_dir: Path, agent: dict, name: str, attempt_timeout: float) -> dict:
    return {
        "task": {"path": str(packet)},
        "agent": agent,
        "environment": {"type": "daytona", "kwargs": {"provisioning": "declarative"}},
        "trials_dir": str(trials_dir),
        "trial_name": name,
        "trial_attempt_timeout_sec": attempt_timeout,
    }


async def _run_logged(config_path: Path, log_path: Path) -> dict:
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return await run_trial(config_path)


def qa_row(root: Path, family: str, row: dict) -> dict:
    name = row["instance_id"]
    LOGGER.info("[%s] consuming frozen packet", name)
    task_spec_bytes = fetch_task_spec_bytes(root, row)
    task_spec = json.loads(task_spec_bytes)
    if row.get("package_uri"):
        # Archive-verbatim mode: the passing prepared package carries the
        # production calibrated task.toml (final actor allocations),
        # task-spec.json and source-runtime.json; the trial reads everything
        # from disk. No reconstruction from baseline resources.
        packet = packet_from_package(root, family, row, task_spec_bytes)
        binding = None
    else:
        if family in ("rst", "multi"):
            raise ValueError(
                f"{family} row {name} lacks the frozen prepared-package archive;"
                " reconstruction from baseline resources is not allowed"
            )
        packet = packet_from_task_spec(root / "packets" / f"{family}-{name}", task_spec)
        binding = fetch_row_runtime(root, row) if row.get("row_runtime_uri") else None
    if binding is not None and binding["task_sha256"] != row["task_spec_sha256"]:
        raise ValueError(f"row runtime binding does not bind the executed TaskSpec for {name}")
    guard = attempt_guard(row)
    record = {
        "task_id": row["task_id"],
        "instance_id": name,
        "family": family,
        "failure_class": row.get("failure_class"),
        "pinned_reference": row.get("pinned_reference"),
        "task_spec_sha256": row["task_spec_sha256"],
        "source_task_sha256": row.get("source_task_sha256") or row.get("sourceTaskSHA"),
        "row_runtime_uri": row.get("row_runtime_uri"),
        "row_runtime_sha256": row.get("row_runtime_sha256") or row.get("runtimeSHA"),
        "bound_runtime_task_sha256": binding["task_sha256"] if binding else None,
        "package_uri": row.get("package_uri"),
        "package_sha256": row.get("package_sha256"),
        "task_checksum": row.get("task_checksum"),
        "actor_execution_requirements": row.get("actor_execution_requirements"),
        "verifier_execution_requirements": row.get("verifier_execution_requirements"),
        "nop_execution_requirements": row.get("nop_execution_requirements"),
        "nop_verifier_execution_requirements": row.get("nop_verifier_execution_requirements"),
        "oracle_trial_config": row.get("oracle_trial_config"),
        "nop_trial_config": row.get("nop_trial_config"),
        "attempt_guard_sec": guard,
    }
    for phase, agent in row_phases(family):
        _probe.testbed_python = None
        attempt = root / "trials" / f"{family}-{name}-{phase}"
        attempt.mkdir(parents=True)
        config = trial_config(packet, attempt / "jobs", agent, f"{family}-{name}-{phase}", guard)
        if binding is not None:
            apply_phase_requirements(config, row, phase, binding)
        config_path = attempt / "trial.json"
        config_path.write_text(json.dumps(config))
        summary = asyncio.run(asyncio.wait_for(_run_logged(config_path, attempt / "harbor.log"), timeout=guard + 900))
        if phase == "tamper":
            summary["testbed_python"] = getattr(_probe, "testbed_python", None)
        record[phase] = summary
        LOGGER.info("[%s/%s] reward=%s", name, phase, summary["rewards"].get("reward"))
    record["allocation_crosscheck"] = {
        phase: crosscheck_allocations(root, family, row, phase) for phase, _ in row_phases(family)
    }
    return record


def seal(run_name: str, root: Path, records: list, blocked: list) -> None:
    """Write verdicts and upload the evidence bundle before any exit."""
    verdicts = {}
    for record in records:
        verdict = {
            "failure_class": record.get("failure_class"),
            "nop_reward_0": record["nop"]["rewards"].get("reward") == 0.0,
            "oracle_reward_1": record["oracle"]["rewards"].get("reward") == 1.0,
            "testbed_python": record.get("tamper", {}).get("testbed_python"),
        }
        if "tamper" in record:
            verdict["tamper_reward_0"] = record["tamper"]["rewards"].get("reward") == 0.0
        crosscheck = record.get("allocation_crosscheck") or {}
        verdict["allocations_ok"] = all(entry.get("ok") for entry in crosscheck.values())
        verdict["qualified"] = row_qualified(record)
        verdicts[record["instance_id"]] = verdict
    for entry in blocked:
        verdicts[entry["instance_id"]] = {"failure_class": entry["failure_class"], "blocked": entry["error"]}
    (root / "qa-verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    (root / "qa-blocked.json").write_text(json.dumps(blocked, indent=2) + "\n")
    archive = root.parent / f"{run_name}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname=root.name)
    destination = StoragePath(OUTPUT_ROOT) / run_name
    copy(str(archive), str(destination / "evidence.tar.gz"))
    LOGGER.info("verdicts: %s", json.dumps(verdicts, indent=2))


def row_qualified(record: dict) -> bool:
    qualified = record["nop"]["rewards"].get("reward") == 0.0 and record["oracle"]["rewards"].get("reward") == 1.0
    if "tamper" in record:
        qualified = qualified and record["tamper"]["rewards"].get("reward") == 0.0
    crosscheck = record.get("allocation_crosscheck") or {}
    return qualified and all(entry.get("ok") for entry in crosscheck.values())


def main() -> None:
    run_name = os.environ.get("QA_RUN_NAME", "swe-v3-packet-qa-v1")
    map_uri = os.environ.get("QA_MAP_URI", DEFAULT_MAP_URI)
    map_sha256 = os.environ.get("QA_MAP_SHA256", DEFAULT_MAP_SHA256)
    root = Path(os.environ["IRIS_OUTPUT_DIR"]) / run_name
    root.mkdir(parents=True)
    configure_coreweave_s3()
    mapping = read_map(root, map_uri, map_sha256)
    revision = os.environ.get("QA_GIT_REVISION") or "bundle"
    (root / "qa-scope.json").write_text(
        json.dumps(
            {
                "git_revision": revision,
                "map_uri": map_uri,
                "map_sha256": map_sha256,
                "converter_revision": mapping.get("converter_revision"),
                "converter_version": mapping.get("converter_version"),
                "executor": "daytona-eval declarative (target=us, eval key as secret, no snapshots)",
                "packet_source": "frozen TaskSpec packet bytes consumed verbatim (no reconstruction)",
                "trial_contract": (
                    "TaskSpecTrial on frozen packet bytes; nop + original TaskSpec oracle for rst/multi"
                    " (functional, no family tamper tooling); nop/oracle/tamper for scale/lego"
                ),
                "budgets": (
                    "map-calibrated actor/verifier phase deadlines applied when present;"
                    " None declarations preserved (field unset)"
                ),
                "peak_physical_sandboxes": 2,
                "scope": "QA rows per sealed map only; not qualification authority",
            },
            indent=2,
        )
        + "\n"
    )
    if get_secret("DAYTONA_EVAL_API_KEY") is None:
        raise ValueError("Daytona QA requires the DAYTONA_EVAL_API_KEY secret")
    os.environ["DAYTONA_API_KEY"] = get_secret("DAYTONA_EVAL_API_KEY")
    selected = [(family, row) for family, section in mapping["families"].items() for row in section["rows"]]
    records: list = []
    blocked: list = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(qa_row, root, family, row): (family, row) for family, row in selected}
        for future in as_completed(futures):
            _, row = futures[future]
            try:
                record = future.result()
            except Exception as error:
                blocked.append(
                    {
                        "instance_id": row["instance_id"],
                        "family": row.get("family") or futures[future][0],
                        "failure_class": row.get("failure_class"),
                        "task_spec_sha256": row.get("task_spec_sha256"),
                        "error": repr(error),
                    }
                )
                LOGGER.error("[%s] blocked: %s", row["instance_id"], error)
                continue
            records.append(record)
            (root / "qa-records.json").write_text(json.dumps(records, indent=2) + "\n")
            # Durable incremental evidence: survive a mid-run abort.
            copy(str(root / "qa-records.json"), str(StoragePath(OUTPUT_ROOT) / run_name / "qa-records.json"))
    seal(run_name, root, records, blocked)
    failed = [record["instance_id"] for record in records if not row_qualified(record)]
    failed += [entry["instance_id"] for entry in blocked]
    if failed:
        raise SystemExit(f"QA failures: {failed}")
    LOGGER.info("QA PASSED for all %d rows", len(records))


if __name__ == "__main__":
    main()
