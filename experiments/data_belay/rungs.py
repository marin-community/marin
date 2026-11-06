# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Data Belay's ladder: the Belay `vanilla` Track's dense rungs and their token budgets."""

from experiments.grug.fast_track import belay_tracks, launch

# d512 screens a candidate; d768 and d1024 confirm a d512 result.
DATA_BELAY_RUNGS = ("d512", "d768", "d1024")


def rung_tokens(rung: str) -> int:
    """Training tokens of the `vanilla` Track at ``rung``: its steps times its batch of 4,096-token sequences."""
    _, vanilla_steps, _ = belay_tracks.RUNGS[rung]
    return vanilla_steps * launch._h100_ladder_rung(rung).baseline_batch * launch.SEQ_LEN


def ladder_tokens() -> dict[str, int]:
    return {rung: rung_tokens(rung) for rung in DATA_BELAY_RUNGS}
