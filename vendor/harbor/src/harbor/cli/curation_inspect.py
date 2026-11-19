"""Read curation evidence using Harbor's own models in its locked environment."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from taskcompendium.harbor.adapter import TASK_SPEC_FILE
from taskcompendium.identity import canonical_sha256

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import resolve_verifier_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("results", "tasks", "native-grades", "native-rewards"))
    args = parser.parse_args()
    paths = json.load(sys.stdin)
    inspected = {}
    for name in paths:
        path = Path(name)
        if args.kind == "tasks":
            if Task.is_valid_dir(path):
                task = Task(path)
                inspected[name] = {"task_name": task.name, "task_checksum": task.checksum}
            continue
        if args.kind == "native-rewards":
            grade = resolve_verifier_rewards(TrialPaths(path), [])
            inspected[name] = {
                "rewards": VerifierResult(rewards=grade.rewards).rewards,
                "artifact_name": grade.artifact.name,
                "artifact_sha256": hashlib.sha256(grade.artifact.read_bytes()).hexdigest(),
            }
            continue
        document = json.loads(path.read_text())
        if isinstance(document, dict) and "trial_name" in document:
            result = TrialResult.model_validate(document)
            if args.kind == "results":
                inspected[name] = True
                continue
            inspected[name] = None
            if result.exception_info is not None or result.finished_at is None or result.verifier_result is None:
                continue
            native = Path(result.config.task.path)
            trial_paths = TrialPaths(path.parent)
            if (native / TASK_SPEC_FILE).is_file() or (trial_paths.verifier_dir / "verdict.json").exists():
                continue
            if not trial_paths.reward_json_path.exists() and not trial_paths.reward_text_path.exists():
                continue
            task = Task(native)
            grade = resolve_verifier_rewards(trial_paths, [task.paths.tests_dir])
            rewards = VerifierResult(rewards=grade.rewards).rewards
            if canonical_sha256(rewards) != canonical_sha256(result.verifier_result.rewards):
                raise ValueError(f"Native reward artifact differs from the production result: {name}")
            with grade.artifact.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            inspected[name] = {
                "rewards": rewards,
                "artifact_path": str(grade.artifact),
                "artifact_sha256": digest,
                "result_sha256": canonical_sha256(document),
                "task_checksum": result.task_checksum,
            }
    json.dump(inspected, sys.stdout)


if __name__ == "__main__":
    main()
