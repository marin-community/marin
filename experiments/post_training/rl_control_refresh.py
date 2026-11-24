# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan control refreshes from the current source checkpoint."""

import argparse
import json
from collections.abc import Sequence

from marin.rl.skyrl_records import RecordedLaunch, RecordedSource, read_stored_skyrl_attempt
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.provenance import Provenance

from experiments.post_training import rl_compose, rl_gate
from experiments.post_training.rl_measurements import RunInputs, read_recorded_launch, read_task_set
from experiments.post_training.rl_presets import ControlPin, ModelPresets, Preset
from experiments.post_training.snowball_rl import SNOWBALL


def plan_refresh(table: ModelPresets, preset_name: str | None = None) -> tuple[str, list[str]]:
    """Return the source commit and control presets with changed revisions or launch training sets."""
    provenance = Provenance.capture(rl_compose.REPO_ROOT)
    if provenance.dirty or not provenance.base_commit:
        raise ValueError("control refresh needs a clean source commit")

    presets = table.by_name
    pins = table.control_pins
    if preset_name is not None:
        selected = presets.get(preset_name)
        name = selected.control_preset if selected is not None else None
        if name is None or name not in pins:
            raise ValueError(f"choose a gated preset with controls, got {preset_name!r}")
        pins = {name: pins[name]}

    launch_selected = {preset.control_preset for preset in presets.values() if preset.tasks_input != "forbidden"}
    digests: dict[tuple[tuple[str, str], ...], str] = {}

    def digest(sources: Sequence[RecordedSource]) -> str:
        paths = tuple(sorted((source.uri, source.relative_path) for source in sources))
        if paths not in digests:
            digests[paths] = read_task_set(sources).task_set_sha256
        return digests[paths]

    def training_changed(preset: Preset, previous: ControlPin) -> bool:
        (config,) = rl_gate._compose_plans(preset, (RunInputs(preset_name=preset.name),))
        expected = RecordedLaunch.from_document(config, attempt=config["run"]["attempt_id"])
        current_tasks = digest(expected.train_data)
        for index in range(previous.replicates):
            output = rl_gate.control_path(preset, previous, index, table=table)
            receipt = read_stored_skyrl_attempt(output)
            if receipt is None or receipt.native.result.state != "succeeded":
                raise ValueError(f"expected a successful pinned control receipt at {output}")
            if digest(read_recorded_launch(receipt).train_data) != current_tasks:
                return True
        return False

    planned = [
        name
        for name, pin in pins.items()
        if pin.revision != presets[name].revision or (name in launch_selected and training_changed(presets[name], pin))
    ]
    return provenance.base_commit, planned


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", help="Plan the control cohort for one gated preset")
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    commit, planned = plan_refresh(SNOWBALL, args.preset)
    print(json.dumps({"commit": commit, "refresh": planned}))


if __name__ == "__main__":
    main()
