# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protected SHARED source-scorer adapter QA on real pinned images (Scale-SWE + SWE-Lego).

Small-sample fresh Harbor QA for the landed shared-source adapter: for each
reviewed sample row from ingest agent A's map, assemble the native package
exactly as the production converter does, harden it with the landed adapter,
and run three real LOCAL trials on the digest-pinned source image: nop (reward
0), oracle (reward 1), and a scripted tamper agent (reward 0). Scope:
small_real_image_QA_only_not_full_population_qualification.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shlex
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tomlkit
from harbor.agents import factory as agent_factory
from harbor.agents.base import BaseAgent
from harbor.host_capabilities import host_capabilities_from_environment
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.secrets import get_secret
from taskcompendium.harbor.shared_source import (
    SCALE_SWE_PROFILE,
    SWE_LEGO_PROFILE,
    harden_shared_source,
)
from taskcompendium.integrity import SharedDependencyState

from experiments.post_training.task_curation.datasets.prime_intellect import swe as swe_module
from experiments.post_training.task_curation.difficulty.runtime import worker_limits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("scale-lego-protected-qa")

MAP_URI = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.15.1/swe-image-bindings-v1/scale-lego-qa-sample-map-v1.json"
MAP_SHA256 = "cba0d4b8043a549d1b9b581508a658d1c27def4fc9da382f37dfb530a54d27db"
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/shared-grader-adapters/2026.11.15.1"
ATTEMPT_TIMEOUT = 4 * 3600.0
HARBOR_PROJECT = Path("vendor/harbor")
PROFILES = {"scale": SCALE_SWE_PROFILE, "lego": SWE_LEGO_PROFILE}
ORACLE_PATCH_COMMANDS = {
    "scale": (
        "git apply --verbose /solution/gold.patch",
        "git apply --verbose --ignore-space-change --ignore-whitespace /solution/gold.patch",
        "patch --batch --fuzz=5 -p1 -i /solution/gold.patch",
        "git apply --verbose --reject --ignore-whitespace /solution/gold.patch",
    ),
    "lego": (
        "git apply --whitespace=fix /solution/gold.patch",
        "patch --batch --fuzz=5 -p1 -i /solution/gold.patch",
    ),
}

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

# QA-local image preparation: the LOCAL backend executes digest-pinned prepared
# images and cannot build. This adapter keeps the landed image layer for building
# backends and installs the identical layer commands at setup time for this QA,
# so the trials run the original digest-pinned source image unchanged.
LAYER_SETUP = (
    "useradd --create-home --uid 10001 --user-group taskcompendium-agent 2>/dev/null || true\n"
    "useradd --create-home --uid 10002 --user-group taskcompendium-grade 2>/dev/null || true\n"
    "mkdir -p /home/taskcompendium-agent /opt/taskcompendium-shared-bootstrap\n"
    "cp /setup_files/taskcompendium-shared/control.py"
    " /setup_files/taskcompendium-shared/shared_pytest.py"
    " /opt/taskcompendium-shared-bootstrap/\n"
    "chown 10001:10001 /home/taskcompendium-agent\n"
    "chmod -R go-w /opt/taskcompendium-shared-bootstrap\n"
)


def adapt_for_local(files: dict[str, bytes], workdir: str, pinned_reference: str) -> None:
    """Move the landed image layer into the pre-agent setup and declare the source image."""
    dockerfile = files["environment/Dockerfile"].decode()
    files["environment/Dockerfile"] = dockerfile.split("\nUSER 0:0\n")[0].encode()
    files["setup_files/taskcompendium-shared/control.py"] = files.pop("environment/taskcompendium-shared/control.py")
    files["setup_files/taskcompendium-shared/shared_pytest.py"] = files.pop(
        "environment/taskcompendium-shared/shared_pytest.py"
    )
    files["environment/shellbox-images.json"] = (
        json.dumps({"main": {"image": pinned_reference, "workdir": workdir}}, indent=2).encode() + b"\n"
    )
    task = tomlkit.parse(files["task.toml"].decode())
    command = str(task["environment"]["setup"]["command"])
    head, marker, rest = command.partition("\n")
    if marker != "\n" or not head.startswith("set -euo pipefail"):
        raise ValueError("Unexpected hardened setup command shape")
    task["environment"]["setup"]["command"] = head + "\n" + LAYER_SETUP + rest
    files["task.toml"] = tomlkit.dumps(task).encode()


def run_checked(command: list[str], **kwargs) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr[-4000:]}")
    return result.stdout


def read_sample_map(root: Path) -> dict:
    path = root / "scale-lego-qa-sample-map-v1.json"
    copy(MAP_URI, str(path), no_clobber=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MAP_SHA256:
        raise ValueError(f"sample map hash mismatch: {digest}")
    return json.loads(path.read_text())


def package_files(row: dict, family: str) -> tuple[dict[str, bytes], str]:
    """Assemble the native package bytes exactly as the production converter does."""
    data = row["row"]
    here = Path(swe_module.__file__).parent
    if family == "scale":
        workdir = data["workdir"]
        config = {
            "workdir": workdir,
            "base_commit": data["parent_commit"],
            "f2p_patch": data["f2p_patch"],
            "f2p_script": data["f2p_script"],
            "test_ids": swe_module._ids(data["FAIL_TO_PASS"]) + swe_module._ids(data["PASS_TO_PASS"]),
        }
        grader = {
            "grade.py": (here / "scale_grade.py").read_bytes(),
            "scale_score.py": (here / "scale_score.py").read_bytes(),
        }
        setup = "cd " + shlex.quote(workdir) + " && " + data["pre_commands"].strip().removesuffix("\\n")
    else:
        workdir = "/testbed"
        config = {
            "base_commit": data["base_commit"],
            "test_patch": data["test_patch"],
            "test_cmd": data["test_cmd"],
            "test_ids": list(data["FAIL_TO_PASS"]) + list(data["PASS_TO_PASS"]),
        }
        grader = {"grade.py": (here / "lego_grade.py").read_bytes()}
        setup = swe_module.LINK_VENV + "; " + swe_module.INSTALL_RG
    oracle = "#!/bin/sh\nset -eu\ncd " + shlex.quote(workdir) + "\n" + " || ".join(ORACLE_PATCH_COMMANDS[family]) + "\n"
    files = {
        "instruction.md": (swe_module.SOLVER_SYSTEM_PROMPT + "\n\n" + data["problem_statement"]).encode(),
        "task.toml": (
            tomlkit.dumps(
                {
                    "environment": {
                        "workdir": workdir,
                        "cpus": 4,
                        "memory_mb": 4096,
                        "storage_mb": 10240,
                        "allow_internet": True,
                        "env": dict(swe_module.ENV),
                        "setup": {"command": setup, "timeout_sec": 1800.0},
                    },
                    "verifier": {"environment_mode": "shared"},
                }
            ).encode()
        ),
        "environment/Dockerfile": ("FROM " + row["pinned_reference"] + "\nWORKDIR " + workdir + "\n").encode(),
        "tests/test.sh": b"#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython /tests/grade.py\n",
        "tests/native_scripts.py": (here / "native_scripts.py").read_bytes(),
        "tests/config.json": json.dumps(config, sort_keys=True).encode(),
        "solution/solve.sh": oracle.encode(),
        "solution/gold.patch": data["patch"].encode(),
    }
    for name, content in grader.items():
        files[f"tests/{name}"] = content
    harden_shared_source(files, profile=PROFILES[family], dependency_state=SharedDependencyState.AGENT_MUTABLE)
    return files, workdir


def write_package(
    root: Path, name: str, files: dict[str, bytes], row: dict, family: str, workdir: str, kind: str
) -> Path:
    package = root / "packages" / name
    if package.exists():
        raise FileExistsError(package)
    package.mkdir(parents=True)
    preparation = "original landed package with its Dockerfile layer (Daytona declarative build)"
    if kind == "local":
        adapt_for_local(files, workdir, row["pinned_reference"])
        preparation = "image layer installed at setup time; source image consumed digest-pinned without a build"
    for relative, content in sorted(files.items()):
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {
        "task_id": row["instance_id"],
        "family": family,
        "source_row": row["source_row"],
        "pinned_reference": row["pinned_reference"],
        "taskset_revision": swe_module.TASKSET_REVISION,
        "local_preparation": preparation,
        "files": {
            relative: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for relative, content in sorted(files.items())
        },
    }
    (root / "packages" / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return package


async def run_trial(config_path: Path, probe_state: dict) -> dict:
    """Run one production trial in-process and return its result summary."""

    class TamperAgent(BaseAgent):
        """The QA's declared tamper probe, registered as a scripted agent.

        The agent writes the interpreter probe into ``probe_state``, the
        per-attempt record bound by the enclosing run.
        """

        SUPPORTS_WINDOWS = False

        @staticmethod
        def name() -> str:
            return "tamper"

        def version(self) -> str:
            return "qa-1"

        async def setup(self, environment) -> None:
            pass

        async def run(self, instruction, environment, context) -> None:
            # Runs with the published agent identity (the task's agent user).
            probe = await environment.exec(
                command="python -c 'import sys; print(sys.version_info[:3])'", timeout_sec=120
            )
            probe_state["testbed_python"] = probe.stdout.strip()
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


def trial_environment(kind: str, limits: dict) -> dict:
    """The declared QA executor; shared adapters need a real-root backend."""
    if kind == "local":
        return {"type": "local", "kwargs": dict(limits)}
    # Validation builds use the dedicated eval account, mapped per the Daytona
    # runbook; the RL credential is never touched in this QA job. The key
    # arrives as an Iris-delivered secret file or an environment value.
    if get_secret("DAYTONA_EVAL_API_KEY") is None:
        raise ValueError("Daytona QA requires the DAYTONA_EVAL_API_KEY secret")
    os.environ["DAYTONA_API_KEY"] = get_secret("DAYTONA_EVAL_API_KEY")
    return {"type": "daytona", "kwargs": {}}


def trial_config(package: Path, trials_dir: Path, agent: dict, name: str, kind: str, limits: dict) -> dict:
    return {
        "task": {"path": str(package)},
        "agent": agent,
        "environment": trial_environment(kind, limits),
        "trials_dir": str(trials_dir),
        "trial_name": name,
        "trial_attempt_timeout_sec": ATTEMPT_TIMEOUT,
    }


async def _run_logged(config_path: Path, log_path: Path, probe_state: dict) -> dict:
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return await run_trial(config_path, probe_state)


def qa_task(root: Path, family: str, row: dict, kind: str, limits: dict) -> dict:
    name = row["instance_id"]
    LOGGER.info("[%s] building hardened package", name)
    files, workdir = package_files(row, family)
    package = write_package(root, f"{family}-{name}", files, row, family, workdir, kind)
    record = {
        "task_id": name,
        "family": family,
        "source_row": row["source_row"],
        "pinned_reference": row["pinned_reference"],
    }
    for phase, agent in (
        ("nop", {"name": "nop"}),
        ("oracle", {"name": "oracle", "kwargs": {"resource_usage_filename": "resource-usage.json"}}),
        ("tamper", {"name": "tamper"}),
    ):
        probe_state: dict = {}
        attempt = root / "trials" / f"{family}-{name}-{phase}"
        attempt.mkdir(parents=True)
        config = trial_config(package, attempt / "jobs", agent, f"{family}-{name}-{phase}", kind, limits)
        config_path = attempt / "trial.json"
        config_path.write_text(json.dumps(config))
        summary = asyncio.run(
            asyncio.wait_for(
                _run_logged(config_path, attempt / "harbor.log", probe_state), timeout=ATTEMPT_TIMEOUT + 900
            )
        )
        if phase == "tamper":
            summary["testbed_python"] = probe_state.get("testbed_python")
        record[phase] = summary
        LOGGER.info("[%s/%s] reward=%s", name, phase, summary["rewards"].get("reward"))
    return record


def main() -> None:
    run = os.environ.get("QA_RUN_NAME", "local")
    kind = os.environ.get("QA_ENVIRONMENT", "daytona")
    if kind not in ("daytona", "local"):
        raise ValueError(f"Unsupported QA environment: {kind}")
    root = Path(os.environ["IRIS_OUTPUT_DIR"]) / "scale-lego-protected-qa-v1"
    limits = worker_limits(root) if kind == "local" else {}
    output = str(StoragePath(OUTPUT_ROOT) / run)
    root.mkdir(parents=True)
    mapping = read_sample_map(root)
    revision = os.environ.get("QA_GIT_REVISION") or "bundle"
    harbor_lock = HARBOR_PROJECT / "uv.lock"
    (root / "qa-scope.json").write_text(
        json.dumps(
            {
                "git_revision": revision,
                "run": run,
                "adapter_landing": "47c99fc4229a76e61a8c91c8f97119e478fb721",
                "taskset_revision": swe_module.TASKSET_REVISION,
                "harbor_lock_sha256": (
                    hashlib.sha256(harbor_lock.read_bytes()).hexdigest() if harbor_lock.exists() else None
                ),
                "map_uri": MAP_URI,
                "map_sha256": MAP_SHA256,
                "image_pinning": "digest-pinned references from the reviewed sample map",
                "scope": "small_real_image_QA_only_not_full_population_qualification",
            },
            indent=2,
        )
        + "\n"
    )
    if kind == "local":
        LOGGER.info("worker limits: %s", json.dumps(limits))
    selected = [(family, row) for family, section in mapping["families"].items() for row in section["rows"]]
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(lambda item: qa_task(root, *item, kind, limits), selected):
            records.append(record)
    (root / "qa-records.json").write_text(json.dumps(records, indent=2) + "\n")
    verdicts = {
        record["task_id"]: {
            "family": record["family"],
            "nop_reward_0": record["nop"]["rewards"].get("reward") == 0.0,
            "oracle_reward_1": record["oracle"]["rewards"].get("reward") == 1.0,
            "tamper_reward_0": record["tamper"]["rewards"].get("reward") == 0.0,
        }
        for record in records
    }
    (root / "qa-verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    archive = root.parent / "scale-lego-protected-qa-v1.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname=root.name)
    copy(str(archive), str(StoragePath(output) / "evidence.tar.gz"), no_clobber=True)
    LOGGER.info("verdicts: %s", json.dumps(verdicts, indent=2))
    failed = [task for task, verdict in verdicts.items() if not all(verdict.values())]
    if failed:
        raise SystemExit(f"QA failures: {failed}")
    LOGGER.info("QA PASSED for all %d tasks", len(verdicts))


if __name__ == "__main__":
    main()
