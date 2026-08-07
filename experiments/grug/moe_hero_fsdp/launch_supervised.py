# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the rack-scaled FSDP MoE hero under the GPU hang supervisor."""

from experiments.grug.moe_hero_fsdp.launch import build_supervised_hero_run, hero_launch_command

main = hero_launch_command(build_supervised_hero_run)


if __name__ == "__main__":
    main()
