# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run nightly control cohorts through each model's regular launcher."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from finestore.rl.rl_progress import AttemptSummary, completed_controls
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.provenance import Provenance

from experiments.post_training import rl_compose, rl_gate, rl_index, rl_publication
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_presets import ModelPresets, Preset
from experiments.post_training.snowball_rl import SNOWBALL

_MODELS = {
    SNOWBALL.model: (SNOWBALL, "experiments.post_training.snowball_rl"),
    MOCHI.model: (MOCHI, "experiments.post_training.mochi_rl"),
}


def _read_control_index(jobs: list[tuple[str, ModelPresets, Preset]]) -> list[AttemptSummary]:
    """Recover terminal receipts left by a previous workflow before counting controls."""
    records = rl_index.list_attempts()
    recovered = False
    for _, table, preset in jobs:
        recovered |= rl_publication.recover_control_outputs(preset, table)
    return rl_index.list_attempts() if recovered else records


def _run_control(module: str, table: ModelPresets, preset: Preset, batch: str, records: list[AttemptSummary]) -> int:
    launch = rl_gate.compose_preset(preset)
    version = rl_gate.config_version(launch)
    cohort = completed_controls(records, model=table.model, preset=preset.name, version=version)
    count = max(1, rl_gate.CONTROL_COUNT - len(cohort))
    command = [
        sys.executable,
        "-m",
        module,
        "--preset",
        preset.name,
        "--control",
        batch,
        "--replicates",
        str(count),
        "--version",
        rl_compose.RUN_ARTIFACT_VERSION,
        "--run",
        "--wait",
    ]
    print(json.dumps({"model": table.model, "preset": preset.name, "version": version, "requested": count}), flush=True)
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(_MODELS))
    parser.add_argument("--preset")
    parser.add_argument("--batch", required=True, help="Unique schedule or dispatch identifier.")
    args = parser.parse_args()
    provenance = Provenance.capture(rl_compose.REPO_ROOT)
    if not provenance.branch or provenance.dirty:
        parser.error("controls require a clean named branch")
    proof = provenance.branch != "main"
    if proof and os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        parser.error("a control run outside main requires workflow_dispatch")
    jobs = [
        (module, table, preset)
        for model, (table, module) in _MODELS.items()
        if args.model is None or model == args.model
        for preset in table.presets
        if preset.gated and (args.preset is None or preset.name == args.preset)
    ]
    if not jobs:
        parser.error("the filters must select at least one gated preset")
    configure_coreweave_s3()
    records = [] if proof else _read_control_index(jobs)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        submitted = [
            (table.model, preset.name, pool.submit(_run_control, module, table, preset, args.batch, records))
            for module, table, preset in jobs
        ]
        failed = False
        for model, preset, future in submitted:
            try:
                code = future.result()
            except (ValueError, OSError, RuntimeError) as error:
                print(f"{model}/{preset}: {error}", file=sys.stderr)
                failed = True
            else:
                failed |= code != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
