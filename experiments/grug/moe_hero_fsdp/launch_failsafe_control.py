# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the FSDP MoE hero with XLA failsafes but no supervisor parent."""

from experiments.grug.moe_hero_fsdp.launch import build_failsafe_control_hero_run, hero_launch_command

main = hero_launch_command(build_failsafe_control_hero_run)


if __name__ == "__main__":
    main()
