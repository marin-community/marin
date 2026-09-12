# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute exported task packages through the pinned Harbor trial lifecycle."""

import argparse
import asyncio
import json
from importlib.metadata import distribution
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from taskcompendium.models import HARBOR_REVISION
from taskcompendium.serialization import from_json, protocol_from_json, specification_hash


async def run_trial(
    task_dir: Path,
    execution: dict,
    trials_dir: Path,
    trial_name: str,
):
    """Run an exported task with native Harbor agent/environment/verifier config."""
    metadata = distribution("harbor").read_text("direct_url.json")
    source = json.loads(metadata) if metadata else {}
    if source.get("vcs_info", {}).get("commit_id") != HARBOR_REVISION:
        raise RuntimeError(f"Harbor runtime must be installed from pinned revision {HARBOR_REVISION}")
    manifest = json.loads((task_dir / "manifest.json").read_text())
    if manifest["harbor_revision"] != HARBOR_REVISION:
        raise ValueError("Export manifest requires a different Harbor revision")
    specification = from_json((task_dir / "specification.json").read_bytes())
    if manifest["specification_sha256"] != specification_hash(specification):
        raise ValueError("Export specification does not match its manifest hash")
    protocol = protocol_from_json((task_dir / "protocol.json").read_bytes())
    if protocol_from_json(json.dumps(manifest["protocol"])) != protocol:
        raise ValueError("Export protocol does not match its manifest")
    config = TrialConfig.model_validate(
        {
            **execution,
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
        }
    )
    trial = await Trial.create(config)
    return await trial.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--trial-name", required=True)
    args = parser.parse_args()
    execution_path = args.execution or args.task / "execution.json"
    execution = json.loads(execution_path.read_text())
    result = asyncio.run(run_trial(args.task, execution, args.trials_dir, args.trial_name))
    print(result.model_dump_json(indent=2))
    if result.verifier_result is None or result.exception_info is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
