#!/usr/bin/env python3
"""Run the c32 preservation counterexample in fresh Daytona sandboxes.

The controller process runs on the remote cluster.  Generated task code and
artifacts execute only in fresh network-blocked Daytona sandboxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

EXPECTED_HASHES = {
    "specification.json": "ad53389146ef988deb01ecfb209ba01da7fb58a0ba71a0f02a1ea03419bd2e6b",
    "renderings.json": "00ef9ac95fe6e30daa34364555baaa0d205c7fae22e4a683347eb5ed9ca07a33",
    "evaluate.py": "7c53802c1189500f9271790cc9ee211bd608a30cc73416d7a3d0559bfe6b0579",
    "run_eval.py": "de9f8628aef24aee5bc4a6a1a4fba37c39830806a74a6cd0bea9d3247dfcd19b",
    "reference_cleaned.gpkg": "5673da41c593d26c78d5a2a1a968df84c53f18181da9d816b47bedbb666ac43a",
    "ground_truth.json": "f25c5c20c12e07b40141a997ea13303094718640b954cc61de51aba41d90b064",
    "stormwater_conduits.gpkg": "79fdebbb5ced0f4da30bbf621b8da2ef2674c9a842b519daaa426d44588566ca",
}
VERIFIER_SNAPSHOT = "cap-verifier-7db1293bfef8c2edb297"
VERIFIER_IMAGE = "sha256:52e86a463eda0c05614d313f6f7f90a3e80e79f0b44b8899ef4ac6bc9251597c"
SUPERVISOR_PYTHON = "/opt/py312/bin/python3"
PREPARE = (
    "mkdir -p /input /result /tests /snapshot /opt/runtime /audit "
    "&& chmod 700 /input /result /tests "
    "&& find / -xdev -type f \\( -perm -4000 -o -perm -2000 \\) "
    "-exec chmod a-s {} + "
    "&& rm -rf /opt/runtime/taskcompendium /opt/runtime/tasktrove_verify "
    "&& rm -rf /workspace && mkdir -p /workspace"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(inputs: Path) -> tuple[dict[str, str], dict | None]:
    contract_path = inputs / "probe-contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else None
    expected = EXPECTED_HASHES if contract is None else contract["input_hashes"]
    if set(expected) != set(EXPECTED_HASHES):
        raise ValueError("probe contract must bind exactly the seven required inputs")
    actual = {name: sha256(inputs / name) for name in expected}
    if actual != expected:
        raise ValueError("c32 input hash mismatch")
    if contract is not None:
        if contract.get("expected_state") != "counterexample_rejected":
            raise ValueError("repaired probe contract requires counterexample rejection")
        specification = json.loads((inputs / "specification.json").read_text())
        runtime = specification["steps"][0]["verifier"]["runtime"]
        if runtime["image"] != contract["private_image"] or runtime["image"] != VERIFIER_IMAGE:
            raise ValueError("private image differs from probe contract")
        if contract["snapshot"] != VERIFIER_SNAPSHOT:
            raise ValueError("probe contract requires a different private snapshot")
    return actual, contract


class Daytona:
    def __init__(self, helper: Path, log: Path) -> None:
        self.helper = helper
        self.log = log
        self.calls: list[dict[str, Any]] = []

    def call(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        started = time.time()
        command = ["bash", str(self.helper), *args]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        record = {
            "args": list(args),
            "exit": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "seconds": round(time.time() - started, 3),
        }
        self.calls.append(record)
        self.log.write_text(json.dumps(self.calls, indent=2, sort_keys=True) + "\n")
        if check and result.returncode != 0:
            raise RuntimeError(f"Daytona command failed: {args!r}: {result.stderr[-2000:]}")
        return result

    def json(self, *args: str) -> Any:
        return json.loads(self.call(*args).stdout)

    def create(self) -> str:
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                return str(
                    self.json(
                        "sandbox", "create", "--snapshot", VERIFIER_SNAPSHOT, "--no-network"
                    )["id"]
                )
            except Exception as error:  # noqa: BLE001 - bounded provider retry
                last_error = error
                if attempt == 5:
                    break
                time.sleep(20 * (attempt + 1))
        raise RuntimeError("could not provision Daytona sandbox") from last_error

    def execute(self, sandbox_id: str, command: str, timeout: int = 900) -> dict[str, Any]:
        return self.json(
            "exec", sandbox_id, "--timeout", str(timeout), "--json", "--", command
        )

    def upload(self, sandbox_id: str, local: Path, remote: str) -> None:
        self.call("upload", sandbox_id, str(local), remote)

    def download(self, sandbox_id: str, remote: str, local: Path) -> None:
        self.call("download", sandbox_id, remote, str(local))

    def delete(self, sandbox_id: str) -> dict[str, Any]:
        result = self.call("sandbox", "delete", sandbox_id, check=False)
        return {
            "sandbox_id": sandbox_id,
            "deleted": result.returncode == 0,
            "exit": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def require_ok(result: dict[str, Any], label: str) -> None:
    if result.get("exit") != 0:
        raise RuntimeError(f"{label} failed: {result!r}")


def wait_ready(daytona: Daytona, sandbox_id: str) -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        result = daytona.execute(sandbox_id, "test -f /tmp/task-ready", timeout=30)
        if result.get("exit") == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"sandbox {sandbox_id} did not become ready")


def package_path(module_name: str) -> Path:
    spec = importlib_util.find_spec(module_name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"cannot locate installed package {module_name}")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def mutate(
    daytona: Daytona,
    inputs: Path,
    mutation_program: Path,
    output: Path,
    cleanup: list[dict[str, Any]],
    providers: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    sandbox_id = daytona.create()
    providers.append(
        {
            "role": "candidate_mutation",
            "sandbox_id": sandbox_id,
            "snapshot": VERIFIER_SNAPSHOT,
            "network_block_all": True,
        }
    )
    try:
        wait_ready(daytona, sandbox_id)
        require_ok(daytona.execute(sandbox_id, "mkdir -p /audit/input /audit/output"), "mkdir")
        daytona.upload(sandbox_id, mutation_program, "/audit/input/mutate.py")
        daytona.upload(sandbox_id, inputs / "reference_cleaned.gpkg", "/audit/input/reference.gpkg")
        daytona.upload(sandbox_id, inputs / "ground_truth.json", "/audit/input/ground_truth.json")
        command = (
            "python3 /audit/input/mutate.py "
            "--reference /audit/input/reference.gpkg "
            "--ground-truth /audit/input/ground_truth.json "
            "--output /audit/output/balanced_candidate.gpkg "
            "--report /audit/output/mutation.json"
        )
        require_ok(daytona.execute(sandbox_id, command), "remote mutation")
        candidate = output / "balanced_candidate.gpkg"
        report_path = output / "mutation.json"
        daytona.download(sandbox_id, "/audit/output/balanced_candidate.gpkg", candidate)
        daytona.download(sandbox_id, "/audit/output/mutation.json", report_path)
        return candidate, json.loads(report_path.read_text())
    finally:
        cleanup.append(daytona.delete(sandbox_id))


def grade(
    *,
    label: str,
    artifact: Path,
    inputs: Path,
    output: Path,
    daytona: Daytona,
    taskcompendium: Path,
    tasktrove_verify: Path,
    cleanup: list[dict[str, Any]],
    providers: list[dict[str, Any]],
) -> dict[str, Any]:
    sandbox_id = daytona.create()
    providers.append(
        {
            "role": f"private_verifier_{label}",
            "sandbox_id": sandbox_id,
            "snapshot": VERIFIER_SNAPSHOT,
            "network_block_all": True,
        }
    )
    case_dir = output / label
    workspace = case_dir / "workspace" / "out"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact, workspace / "cleaned.gpkg")
    specification = json.loads((inputs / "specification.json").read_text())
    rendering = json.loads((inputs / "renderings.json").read_text())[0]
    payload = {
        "specification": specification,
        "protocol": rendering,
        "step_index": 0,
        "attempt": {"response": "cleaned.gpkg written to /workspace/out/cleaned.gpkg", "transcript": []},
    }
    payload_path = case_dir / "payload.json"
    payload_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    try:
        wait_ready(daytona, sandbox_id)
        require_ok(daytona.execute(sandbox_id, PREPARE), f"prepare {label}")
        daytona.upload(sandbox_id, case_dir / "workspace", "/snapshot")
        daytona.upload(sandbox_id, taskcompendium, "/opt/runtime/taskcompendium")
        daytona.upload(sandbox_id, tasktrove_verify, "/opt/runtime/tasktrove_verify")
        daytona.upload(sandbox_id, payload_path, "/input/payload.json")
        supervisor_command = (
            "chmod -R a-w /snapshot /opt/runtime && "
            f"{SUPERVISOR_PYTHON} -I "
            "/opt/runtime/taskcompendium/harbor/container_entry.py < /input/payload.json"
        )
        executed = daytona.execute(sandbox_id, supervisor_command, timeout=1200)
        (case_dir / "supervisor-execution.json").write_text(
            json.dumps(executed, indent=2, sort_keys=True) + "\n"
        )
        require_ok(executed, f"exact embedded grader {label}")
        stdout = executed.get("stdout") or ""
        graded = json.loads(stdout.strip().splitlines()[-1])
        (case_dir / "grading-result.json").write_text(
            json.dumps(graded, indent=2, sort_keys=True) + "\n"
        )

        # Retain C1-C10 evidence from the byte-identical evaluator.  This is
        # secondary instrumentation; the reward above comes only from the
        # actual embedded run_eval.py through TaskCompendium container_entry.
        daytona.upload(sandbox_id, artifact, "/audit/cleaned.gpkg")
        daytona.upload(sandbox_id, inputs / "evaluate.py", "/audit/evaluate.py")
        daytona.upload(
            sandbox_id, inputs / "stormwater_conduits.gpkg", "/audit/stormwater_conduits.gpkg"
        )
        daytona.upload(sandbox_id, inputs / "ground_truth.json", "/audit/ground_truth.json")
        direct_command = (
            "python3 /audit/evaluate.py --artifact /audit/cleaned.gpkg "
            "--input /audit/stormwater_conduits.gpkg "
            "--ground-truth /audit/ground_truth.json --out /audit/per-check.json"
        )
        direct = daytona.execute(sandbox_id, direct_command, timeout=900)
        (case_dir / "per-check-execution.json").write_text(
            json.dumps(direct, indent=2, sort_keys=True) + "\n"
        )
        per_check_path = case_dir / "per-check.json"
        daytona.download(sandbox_id, "/audit/per-check.json", per_check_path)
        per_check = json.loads(per_check_path.read_text())
        if (
            direct.get("exit") not in (0, 1)
            or per_check.get("outcome", "graded") != "graded"
            or per_check.get("verdict", {}).get("passed") is not (direct["exit"] == 0)
        ):
            raise RuntimeError(f"per-check evaluator {label} ungraded or inconsistent")
        return {
            "label": label,
            "artifact_sha256": sha256(artifact),
            "sandbox_id": sandbox_id,
            "grader": graded,
            "per_check": per_check,
        }
    finally:
        cleanup.append(daytona.delete(sandbox_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutation-program", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    actual_hashes, contract = verify_inputs(args.inputs)
    mutation_program_hash = sha256(args.mutation_program)

    tools_root = Path(os.environ["CAPABILITY_DAYTONA_TOOLS"])
    helper = tools_root / "dt.sh"
    if not helper.is_file():
        raise RuntimeError(f"missing maintained Daytona helper: {helper}")
    taskcompendium_source = Path(os.environ["TASKCOMPENDIUM_SOURCE"])
    taskcompendium = taskcompendium_source / "src" / "taskcompendium"
    tasktrove_verify = package_path("tasktrove_verify")

    cleanup: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []
    daytona = Daytona(helper, args.output / "daytona-calls.json")
    state = "incomplete"
    failure: str | None = None
    cases: list[dict[str, Any]] = []
    mutation: dict[str, Any] | None = None
    try:
        candidate, mutation = mutate(
            daytona,
            args.inputs,
            args.mutation_program,
            args.output,
            cleanup,
            providers,
        )
        baseline = grade(
            label="baseline",
            artifact=args.inputs / "reference_cleaned.gpkg",
            inputs=args.inputs,
            output=args.output,
            daytona=daytona,
            taskcompendium=taskcompendium,
            tasktrove_verify=tasktrove_verify,
            cleanup=cleanup,
            providers=providers,
        )
        cases.append(baseline)
        attack = grade(
            label="balanced-deletion-duplication",
            artifact=candidate,
            inputs=args.inputs,
            output=args.output,
            daytona=daytona,
            taskcompendium=taskcompendium,
            tasktrove_verify=tasktrove_verify,
            cleanup=cleanup,
            providers=providers,
        )
        cases.append(attack)
        baseline_pass = (
            baseline["grader"].get("status") == "graded"
            and baseline["grader"].get("reward") == 1.0
            and baseline["per_check"].get("verdict", {}).get("passed") is True
        )
        attack_pass = (
            attack["grader"].get("status") == "graded"
            and attack["grader"].get("reward") == 1.0
            and attack["per_check"].get("verdict", {}).get("passed") is True
        )
        if baseline_pass and attack_pass:
            state = "false_positive_confirmed"
        elif not baseline_pass:
            state = "inconclusive_baseline_failed"
        elif (
            attack["grader"].get("status") == "graded"
            and isinstance(attack["grader"].get("reward"), (int, float))
            and 0 <= attack["grader"]["reward"] < 1.0
            and attack["per_check"].get("verdict", {}).get("passed") is False
        ):
            state = "counterexample_rejected"
        else:
            state = "inconclusive_counterexample_ungraded_or_inconsistent"
    except Exception as error:  # noqa: BLE001 - retain partial remote evidence
        failure = f"{type(error).__name__}: {error}"
        state = "infrastructure_or_probe_error"
    finally:
        evidence = {
            "schema": "c32-semantic-probe-evidence-v1",
            "state": state,
            "failure": failure,
            "source": {
                "proposal_hash": "75adb802f475a87007fcaba979b78e8d6671c393270ad54b358f7dc30e42cf33",
                "frozen_static_audit_sha256": "3b39295e812ad64ecfebbc2e46f8f12dff0a17f99bf5ba6f671e2a038cf0eaf3",
                "input_hashes": actual_hashes,
                "mutation_program_sha256": mutation_program_hash,
                "probe_contract": contract,
                "probe_contract_sha256": (
                    sha256(args.inputs / "probe-contract.json") if contract else None
                ),
            },
            "policy": {
                "snapshot": VERIFIER_SNAPSHOT,
                "network_block_all": True,
                "baseline_required_before_attack_interpretation": True,
                "reward_path": "TaskCompendium container_entry -> embedded run_eval.py",
                "per_check_path": "byte-identical embedded evaluate.py in same private image",
            },
            "mutation": mutation,
            "providers": providers,
            "cases": cases,
            "cleanup": cleanup,
        }
        (args.output / "evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
        (args.output / "cleanup.json").write_text(
            json.dumps(cleanup, indent=2, sort_keys=True) + "\n"
        )
    if state == "infrastructure_or_probe_error":
        return 1
    if contract is not None and state != contract["expected_state"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
