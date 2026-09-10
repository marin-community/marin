# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import click
import pytest
import yaml

from experiments.post_training.async_snowball import Scale, training_config
from experiments.post_training.async_snowball_bucket_timing import prepare_timing_request


@pytest.fixture
def source():
    return {
        "run_id": "prior",
        "model": {"uri": "s3://marin-us-east-02a/frozen/model", "identity": "model@revision"},
        "train_data": [{"uri": "s3://marin-us-east-02a/frozen/data", "relative_path": "train.parquet"}],
        "validation_data": [{"uri": "s3://marin-us-east-02a/frozen/data", "relative_path": "validation.parquet"}],
        "output": {"terminal_manifest_uri": "s3://marin-us-east-02a/prior/terminal.json"},
        "config_yaml": training_config(
            Scale.CADENCE_GATE,
            response_tokens=4096,
            eval_response_tokens=4096,
            context_tokens=8192,
            epoch_seeded_shuffle=True,
            dataloader_workers=0,
            publication_stage_timing=True,
        ),
    }


def make_request(source, mode):
    return prepare_timing_request(
        source,
        runtime_commit="a" * 40,
        mode=mode,
        run_id=f"timing-{mode}",
        output_prefix=f"s3://marin-us-east-02a/new/{mode}",
    )


def test_matched_modes_preserve_inputs_and_differ_only_in_transfer_and_outputs(source):
    original = copy.deepcopy(source)
    arms = [make_request(source, mode) for mode in ("reference", "bucket")]
    assert source == original
    configs = []
    for arm in arms:
        request = arm["request"]
        assert all(request[key] == source[key] for key in ("model", "train_data", "validation_data"))
        cfg = yaml.safe_load(request["config_yaml"])
        assert cfg["trainer"]["max_steps"] == cfg["trainer"]["eval_interval"] == 20
        assert cfg["trainer"]["fully_async"]["first_token_admission"]
        assert not cfg["trainer"]["algorithm"]["batch_invariant"]
        assert not cfg["generator"]["inference_engine_serial_startup"]
        cfg["generator"].pop("weight_sync_timing_mode")
        cfg["trainer"].pop("weight_sync_readback_output")
        configs.append(cfg)
    assert configs[0] == configs[1]
    assert arms[0]["request_hash"] != arms[1]["request_hash"]
    assert arms[0]["execution"] == arms[1]["execution"]


@pytest.mark.parametrize("bad", ["region", "overlap", "topology"])
def test_timing_request_rejects_wrong_region_existing_outputs_or_geometry(source, bad):
    if bad == "region":
        source["model"]["uri"] = "s3://other-region/model"
    elif bad == "overlap":
        source["output"]["terminal_manifest_uri"] = "s3://marin-us-east-02a/new/reference/terminal.json"
    else:
        cfg = yaml.safe_load(source["config_yaml"])
        cfg["generator"]["inference_engine_data_parallel_size"] = 4
        source["config_yaml"] = yaml.safe_dump(cfg)
    with pytest.raises(ValueError):
        make_request(source, "reference")


def test_timing_rno_execution_requires_opt_in_and_preserves_scientific_request(source):
    arguments = dict(
        runtime_commit="a" * 40, mode="bucket", run_id="timing", output_prefix="s3://marin-us-east-02a/new/bucket"
    )
    east = prepare_timing_request(source, **arguments)
    with pytest.raises(click.ClickException, match="not local"):
        prepare_timing_request(source, **arguments, cluster="cw-rno2a")
    rno = prepare_timing_request(source, **arguments, cluster="cw-rno2a", allow_cross_region_io=True)
    assert east["request"] == rno["request"]
    assert east["request_hash"] == rno["request_hash"]
    assert rno["execution"]["cluster"] == "cw-rno2a"
    assert rno["execution"]["timeout_seconds"] == 4800
    assert rno["execution"]["max_retries"] == 0
    assert rno["coordinator_timeout_seconds"] > rno["coordinator_wait_seconds"] > 4800
