# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import (
    launch_starcoder_wsd80_dense_support_empirical_optimum_confirmation as launcher,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_dense_support_empirical_optimum_confirmation_20260811 as designer,
)


def test_confirmation_design_regenerates() -> None:
    payload = designer.build_payload()

    assert payload["design_sha256"] == launcher.EXPECTED_DESIGN_SHA256
    assert payload["expected_run_count"] == 280
    assert len(payload["selected_policies"]) == 56
    assert len(payload["runs"]) == 280
    assert not any(row["discovery_is_alias"] for row in payload["selected_policies"])


def test_confirmation_launcher_has_complete_pairs() -> None:
    _, requests = launcher.load_design()

    blocks = {(request.cell_id, request.support_id) for request in requests}
    assert len(blocks) == 28
    assert len({request.data_seed for request in requests}) == 5
    for block in blocks:
        block_rows = [request for request in requests if (request.cell_id, request.support_id) == block]
        assert len(block_rows) == 10


def test_confirmation_wandb_tags_are_valid() -> None:
    _, requests = launcher.load_design()
    metadata = {row["run_name"]: row for row in launcher._load_payload()["runs"]}

    for request in requests:
        policy_class = str(metadata[request.run_name]["policy_class"])
        tags = launcher._wandb_tags(request, policy_class)
        assert all(1 <= len(tag) <= launcher.MAX_WANDB_TAG_LENGTH for tag in tags)
