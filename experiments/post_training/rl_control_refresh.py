# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan control refreshes from the current source checkpoint."""

import argparse
import json
from collections.abc import Sequence

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.provenance import Provenance

from experiments.post_training import rl_compose, rl_gate
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_measurements import RunInputs
from experiments.post_training.rl_presets import ModelPresets
from experiments.post_training.snowball_rl import SNOWBALL


def plan_refresh(table: ModelPresets, preset_name: str | None = None) -> tuple[str, list[str]]:
    """Return the clean source commit and control cohorts the gate would refuse."""
    provenance = Provenance.capture(rl_compose.REPO_ROOT)
    if provenance.dirty or not provenance.base_commit:
        raise ValueError("control refresh needs a clean source commit")
    gated = [preset for preset in table.presets if preset.control_preset is not None]
    if preset_name is not None:
        gated = [preset for preset in gated if preset.name == preset_name]
        if not gated:
            raise ValueError(f"choose a gated preset, got {preset_name!r}")

    planned = []
    for preset in gated:
        owner = preset.control_preset
        assert owner is not None
        if owner in planned:
            continue
        try:
            rl_gate.validate_candidate(preset, RunInputs(preset_name=preset.name), table=table)
        except rl_gate.StaleControls:
            planned.append(owner)
    return provenance.base_commit, planned


def main(argv: Sequence[str] | None = None) -> None:
    tables = {table.model: table for table in (SNOWBALL, MOCHI)}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(tables), help="Inspect a single model table")
    parser.add_argument("--preset", help="Plan the control cohort for one gated preset in the named model")
    args = parser.parse_args(argv)
    if args.preset is not None and args.model is None:
        parser.error("--preset requires --model")
    configure_coreweave_s3()
    selected = {args.model: tables[args.model]} if args.model is not None else tables
    plans = {model: plan_refresh(table, args.preset) for model, table in selected.items()}
    commit = next(iter(plans.values()))[0]
    print(json.dumps({"commit": commit, "refresh": {model: cohorts for model, (_, cohorts) in plans.items()}}))


if __name__ == "__main__":
    main()
