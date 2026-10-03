# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attest unchanged verifier and executed native code across registry revisions."""

import hashlib
import json
import subprocess
from pathlib import Path

HARBOR_VERIFIER_PATH = "src/harbor/verifier"

NATIVE_BINDINGS = {
    "infra/rl_data/sources.py",
    "skyrl-gym/skyrl_gym",
    "skyrl-train/skyrl_train/trajectory_runners/skyrl_gym_contracts.py",
    "skyrl-train/skyrl_train/trajectory_runners/harbor/contracts.py",
    "marinskyrl/packed_tasks.py",
}


def revision_attestation(root: Path, registry_revision: str, verifier_path: str) -> dict:
    """Prove the registry revision preserves native code actually used by this run."""
    identity = json.loads((root / "run.json").read_text())
    checkout = Path(identity["config"]["runtime"]["marinskyrl_checkout"]).resolve()
    original = identity["marinskyrl"]["commit"]
    if identity["marinskyrl"]["dirty"]:
        raise ValueError("Cannot attest a dirty native checkout")
    paths = {*NATIVE_BINDINGS, verifier_path}
    captured = []
    for path in root.glob("tasks/*/execution-*/native-code-index.json"):
        for item in json.loads(path.read_text()):
            origin = Path(item["origin"]).resolve()
            if origin.is_relative_to(checkout):
                paths.add(str(origin.relative_to(checkout)))
                captured.append((str(origin.relative_to(checkout)), path.parent / item["path"], item["sha256"]))
    for relative, snapshot, expected in captured:
        source = subprocess.check_output(["git", "show", f"{original}:{relative}"], cwd=checkout)
        if (
            hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected
            or hashlib.sha256(source).hexdigest() != expected
        ):
            raise ValueError(f"Captured native code differs from its recorded commit at {relative}")
    objects = []
    for path in sorted(paths):
        before = subprocess.check_output(["git", "rev-parse", f"{original}:{path}"], cwd=checkout, text=True).strip()
        after = subprocess.check_output(
            ["git", "rev-parse", f"{registry_revision}:{path}"], cwd=checkout, text=True
        ).strip()
        if before != after:
            raise ValueError(f"Native code changed at {path}; review must execute the new revision")
        objects.append({"path": path, "reviewed_object": before, "registry_object": after})
    return {
        "reviewed_commit": original,
        "registry_commit": registry_revision,
        "objects": objects,
        "scope": "Full Gym library, native bindings, source registry and captured MSkyRL modules are unchanged.",
    }


def validate_harbor_verifier_revision(root: Path, executed_commit: str | None, verifier_revision: str) -> None:
    """Reject a review unless its executed Harbor verifier tree is current and clean."""
    identity = json.loads((root / "run.json").read_text())
    harbor = identity.get("harbor")
    if not harbor or harbor["dirty"] or executed_commit is None or harbor["commit"] != executed_commit:
        raise ValueError("Cannot attest missing, dirty, or inconsistent Harbor execution provenance")
    checkout = Path(identity["config"]["runtime"]["harbor_checkout"])
    path = HARBOR_VERIFIER_PATH
    before = subprocess.check_output(["git", "rev-parse", f"{executed_commit}:{path}"], cwd=checkout, text=True).strip()
    after = subprocess.check_output(["git", "rev-parse", f"{verifier_revision}:{path}"], cwd=checkout, text=True).strip()
    if before != after:
        raise ValueError("Executed Harbor verifier differs from the current Atlas verifier")
