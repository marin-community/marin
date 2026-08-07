# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the FSDP MoE hero with no recovery instrumentation and no retries."""

from experiments.grug.moe_hero_fsdp.launch import build_stock_control_hero_run, hero_launch_command

main = hero_launch_command(build_stock_control_hero_run)


if __name__ == "__main__":
    main()
