# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Six-row corrected SWE QA consuming the frozen v2026.11.16.2 TaskSpec packets.

Per the ingest lane's sealed freeze map (574e2b77..., seal 1cec1f22): download
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

MAP_URI = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.16.1/swe-v2-qa-freeze-v3/qa-map.json"
MAP_SHA256 = "574e2b77e0602a57c931b867f10c5ba6d7cd9c32394a98cb3d171dcbffc72fa9"
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_map(root: Path) -> dict:
    path = root / "qa-map.json"
    copy(MAP_URI, str(path), no_clobber=True)
    digest = sha256_bytes(path.read_bytes())
    if digest != MAP_SHA256:
        raise ValueError(f"qa map hash mismatch: {digest}")
    return json.loads(path.read_text())


def fetch_task_spec(root: Path, row: dict) -> dict:
    """Download the row's actual frozen TaskSpec and verify its exact SHA.

    The local path is keyed by the URI basename: task ids share prefixes
    within a family, so a truncated id would collide across sibling rows and
    a no-clobber copy would leave another row's bytes in place.
    """
    uri, expected = row["task_spec_uri"], row["task_spec_sha256"]
    path = root / "packets" / PurePosixPath(uri).name
    path.parent.mkdir(parents=True, exist_ok=True)
    copy(uri, str(path), no_clobber=True)
    digest = sha256_bytes(path.read_bytes())
    if digest != expected:
        raise ValueError(f"TaskSpec hash mismatch for {row['instance_id']}: {digest} != {expected}")
    return json.loads(path.read_text())


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


def trial_config(packet: Path, trials_dir: Path, agent: dict, name: str) -> dict:
    return {
        "task": {"path": str(packet)},
        "agent": agent,
        "environment": {"type": "daytona", "kwargs": {"provisioning": "declarative"}},
        "trials_dir": str(trials_dir),
        "trial_name": name,
        "trial_attempt_timeout_sec": ATTEMPT_TIMEOUT,
    }


async def _run_logged(config_path: Path, log_path: Path) -> dict:
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return await run_trial(config_path)


def qa_row(root: Path, family: str, row: dict) -> dict:
    name = row["instance_id"]
    LOGGER.info("[%s] consuming frozen packet", name)
    task_spec = fetch_task_spec(root, row)
    packet = packet_from_task_spec(root / "packets" / f"{family}-{name}", task_spec)
    record = {
        "task_id": row["task_id"],
        "instance_id": name,
        "family": family,
        "failure_class": row["failure_class"],
        "pinned_reference": row["pinned_reference"],
        "task_spec_sha256": row["task_spec_sha256"],
    }
    for phase, agent in (
        ("nop", {"name": "nop"}),
        ("oracle", {"name": "oracle", "kwargs": {"resource_usage_filename": "resource-usage.json"}}),
        ("tamper", {"name": "tamper"}),
    ):
        _probe.testbed_python = None
        attempt = root / "trials" / f"{family}-{name}-{phase}"
        attempt.mkdir(parents=True)
        config = trial_config(packet, attempt / "jobs", agent, f"{family}-{name}-{phase}")
        config_path = attempt / "trial.json"
        config_path.write_text(json.dumps(config))
        summary = asyncio.run(
            asyncio.wait_for(_run_logged(config_path, attempt / "harbor.log"), timeout=ATTEMPT_TIMEOUT + 900)
        )
        if phase == "tamper":
            summary["testbed_python"] = getattr(_probe, "testbed_python", None)
        record[phase] = summary
        LOGGER.info("[%s/%s] reward=%s", name, phase, summary["rewards"].get("reward"))
    return record


def seal(run_name: str, root: Path, records: list, blocked: list) -> None:
    """Write verdicts and upload the evidence bundle before any exit."""
    verdicts = {}
    for record in records:
        verdicts[record["instance_id"]] = {
            "failure_class": record["failure_class"],
            "nop_reward_0": record["nop"]["rewards"].get("reward") == 0.0,
            "oracle_reward_1": record["oracle"]["rewards"].get("reward") == 1.0,
            "tamper_reward_0": record["tamper"]["rewards"].get("reward") == 0.0,
            "testbed_python": record["tamper"].get("testbed_python"),
        }
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


def main() -> None:
    run_name = os.environ.get("QA_RUN_NAME", "swe-v2-packet-qa-v2")
    root = Path(os.environ["IRIS_OUTPUT_DIR"]) / run_name
    root.mkdir(parents=True)
    configure_coreweave_s3()
    mapping = read_map(root)
    revision = os.environ.get("QA_GIT_REVISION") or "bundle"
    (root / "qa-scope.json").write_text(
        json.dumps(
            {
                "git_revision": revision,
                "map_uri": MAP_URI,
                "map_sha256": MAP_SHA256,
                "converter_revision": mapping["converter_revision"],
                "converter_version": mapping["converter_version"],
                "executor": "daytona-eval declarative (target=us, eval key as secret, no snapshots)",
                "packet_source": "frozen TaskSpec packet bytes consumed verbatim (no reconstruction)",
                "peak_physical_sandboxes": 2,
                "scope": "six failure-class/PyBaMM rows only; not qualification",
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
    failed = [
        record["instance_id"]
        for record in records
        if not (
            record["nop"]["rewards"].get("reward") == 0.0
            and record["oracle"]["rewards"].get("reward") == 1.0
            and record["tamper"]["rewards"].get("reward") == 0.0
        )
    ] + [entry["instance_id"] for entry in blocked]
    if failed:
        raise SystemExit(f"QA failures: {failed}")
    LOGGER.info("QA PASSED for all %d rows", len(records))


if __name__ == "__main__":
    main()
