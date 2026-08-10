# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the FSDP hero directly inside a reserved multi-node dev-GPU session."""

import click
from marin.execution.lazy import materialized_config
from rigging.filesystem import marin_prefix

from experiments.grug.moe_hero_fsdp.launch import build_hero_run
from experiments.grug.moe_hero_fsdp.train import _apply_hero_fsdp_runtime_defaults, _run_grug_local


@click.command()
@click.option("--run-id", required=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=25, show_default=True)
def main(run_id: str, num_steps: int) -> None:
    """Materialize the normal one-rack FSDP config and execute it in the current Iris task."""

    step = build_hero_run(
        run_id=run_id,
        dp_racks=1,
        num_steps=num_steps,
        save_checkpoints=False,
        version="dev",
    )
    config = materialized_config(step, marin_prefix())
    _apply_hero_fsdp_runtime_defaults()
    _run_grug_local(config)


if __name__ == "__main__":
    main()
