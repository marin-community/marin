"""Compat entrypoint for the block diffusion learning experiment.

After PR #3054, grug experiments live under `experiments/grug/*` templates.
This module keeps the older import path working and forwards to the template
launcher.

Preferred entrypoint:
  uv run python -m experiments.grug.block_diffusion.launch
"""

from __future__ import annotations

from marin.execution.executor import executor_main

from experiments.grug.block_diffusion.launch import grug_block_diffusion_trial


def main() -> None:
    executor_main(
        steps=[grug_block_diffusion_trial],
        description="Template grug block diffusion trial run (learning experiment).",
    )


if __name__ == "__main__":
    main()
