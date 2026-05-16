# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.midtraining.identity import (
    RunIdentity,
    attempt_group_manifest_uri,
    build_run_identity,
    expected_run_env,
    output_region,
)


def test_run_identity_derives_all_namespaces():
    identity = build_run_identity(
        logical_cell_id="delphi-1e21-p33m67-k0p20-lr0p5",
        attempt=1,
        output_region_name="us-east5",
        wandb_project="delphi-midtraining",
    )
    assert identity.run_id == "delphi-1e21-p33m67-k0p20-lr0p5-a001"
    assert identity.output_region == "us-east5"
    assert identity.permanent_checkpoints_uri.endswith("/checkpoints")
    assert identity.manifest_uri.endswith("/midtrain_manifest.json")
    env = expected_run_env(identity)
    assert env["RUN_ID"] == identity.run_id
    assert env["WANDB_RUN_ID"] == identity.run_id
    assert env["WANDB_PROJECT"] == "delphi-midtraining"


def test_attempt_suffix_must_match_field():
    with pytest.raises(ValueError, match="encodes attempt"):
        RunIdentity(
            logical_cell_id="abc",
            attempt=2,
            output_path="gs://marin-us-east5/checkpoints/abc-a001",
            wandb_project="delphi-midtraining",
        )


def test_output_path_must_be_run_root_not_checkpoint():
    with pytest.raises(ValueError, match="must be a run root"):
        RunIdentity(
            logical_cell_id="abc",
            attempt=1,
            output_path="gs://marin-us-east5/checkpoints/abc-a001/checkpoints/step-100",
            wandb_project="delphi-midtraining",
        )


def test_output_region_extracts_from_marin_bucket():
    assert output_region("gs://marin-us-central1/checkpoints/foo") == "us-central1"
    assert output_region("gs://marin-us-east5/midtrain-manifests/data/abc.json") == "us-east5"


def test_output_region_rejects_non_marin_bucket():
    with pytest.raises(ValueError, match="Cannot resolve region"):
        output_region("gs://other-bucket/foo")


def test_attempt_group_manifest_uri_per_region():
    uri = attempt_group_manifest_uri(logical_cell_id="cell-x", region="us-east5")
    assert uri == "gs://marin-us-east5/midtrain-manifests/runs/cell-x.json"


def test_w_and_b_name_length_enforced():
    long_id = "x" * 70
    with pytest.raises(ValueError, match="exceeds W&B safe length"):
        build_run_identity(
            logical_cell_id=long_id,
            attempt=1,
            output_region_name="us-east5",
            wandb_project="delphi-midtraining",
        )


def test_build_refuses_existing_attempt_suffix_in_cell_id():
    with pytest.raises(ValueError, match="already contains an attempt suffix"):
        build_run_identity(
            logical_cell_id="cell-a001",
            attempt=1,
            output_region_name="us-east5",
            wandb_project="delphi-midtraining",
        )


def test_fresh_attempt_has_distinct_run_id():
    a1 = build_run_identity(
        logical_cell_id="cell-x",
        attempt=1,
        output_region_name="us-east5",
        wandb_project="delphi-midtraining",
    )
    a2 = build_run_identity(
        logical_cell_id="cell-x",
        attempt=2,
        output_region_name="us-east5",
        wandb_project="delphi-midtraining",
    )
    assert a1.run_id != a2.run_id
    assert a1.logical_cell_id == a2.logical_cell_id
