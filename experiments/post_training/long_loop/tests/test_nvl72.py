# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context

from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.nvl72 import PROFILES, NVL72RunSpec, run_from_spec, validate_model_geometry


@pytest.fixture
def run_spec(tmp_path: Path, monkeypatch) -> NVL72RunSpec:
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-08a/marin/users/test/mochi")
    data = tmp_path / "train.parquet"
    data.write_bytes(b"pinned mixed task input")
    return NVL72RunSpec(
        format="nvl72_rl",
        marin_revision="a" * 40,
        output_prefix="s3://marin-us-east-08a/marin/users/test/mochi",
        profile="snowball-one-rack",
        model_uri=str(tmp_path / "model"),
        tokenizer_revision="b" * 40,
        data_uri=(str(tmp_path),),
        data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        input_version="2026.11.17.0",
        max_steps=2,
        max_new_tokens=1024,
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("resume", [False, True])
def test_profile_build_preserves_native_resume_and_nvlink_roles(run_spec, tmp_path, profile, resume):
    output = tmp_path / "launch.yaml"
    native = str(tmp_path / "native/global_step_13") if resume else None
    spec = replace(run_spec, profile=profile, resume_uri=native, max_steps=15 if resume else 2)
    with build_context(BuildContext(versions=VersionCodex(default="2026.11.17.1"))):
        step = run_from_spec(spec, smoke=True, config_output=output)
    launch = yaml.safe_load(output.read_text())
    trainer = launch["skyrl"]["trainer"]
    assert launch["iris"]["role_topology"] == "nvlink.domain"
    assert launch["iris"]["allocation"]["gpu_variant"] == "GB200"
    assert trainer["max_steps"] == (15 if resume else 2)
    assert trainer["restore_dataloader_state"] and not trainer["reset_global_step_on_resume"]
    assert launch["skyrl"]["generator"]["engine_init_kwargs"]["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert not launch["skyrl"]["generator"]["enforce_eager"]
    assert {c["type"] for c in trainer["callbacks"]} == {"checkpoint", "wall_time_budget"}
    if native is not None:
        assert launch["inputs"]["starting_checkpoint"]["uri"] == native
    assert step.version == "2026.11.17.1"
    assert step.runtime_args["skyrl_execution"].coordinator_name == f"mochi-{profile}-2026.11.17.1"


def test_normal_run_binds_distinct_training_and_heldout_data(run_spec, tmp_path):
    output = tmp_path / "launch.yaml"
    spec = replace(run_spec, profile="one-rack", validation_data_uri=(str(tmp_path / "heldout"),))
    with build_context(BuildContext(versions=VersionCodex(default="2026.11.17.2"))):
        run_from_spec(spec, smoke=False, config_output=output)
    launch = yaml.safe_load(output.read_text())
    assert launch["inputs"]["validation_data"][0]["uri"] == str(tmp_path / "heldout")
    assert launch["inputs"]["validation_data"][0]["local_path"] != launch["inputs"]["train_data"][0]["local_path"]


@pytest.mark.parametrize("profile", ["snowball-one-rack", "snowball-three-rack"])
def test_snowball_profiles_use_ib_ports_on_mixed_link_hosts(run_spec, tmp_path, profile):
    output = tmp_path / "launch.yaml"
    spec = replace(run_spec, profile=profile)
    with build_context(BuildContext(versions=VersionCodex(default="2026.11.18.62"))):
        run_from_spec(spec, smoke=True, config_output=output)
    environment = yaml.safe_load(output.read_text())["skyrl"]["extra_env"]
    assert environment["NCCL_BUFFSIZE"] == "4194304"
    # Port inventory recorded on both hosts of the .63 diagnostic.
    links = {f"ibp{index}:1": "InfiniBand" for index in range(4)}
    links.update({f"mlx5_{index}:1": "Ethernet" for index in range(9)})
    mask = environment["NCCL_IB_HCA"]
    assert mask.startswith("=")
    selected = set(mask[1:].split(","))
    assert selected == {port for port, layer in links.items() if layer == "InfiniBand"}
    assert "mlx5_2:1" not in selected
    assert environment["NCCL_NET"] == "IB" and environment["NCCL_IB_DISABLE"] == "0"


def test_snowball_geometry_rejects_mochi_profile_and_build_rejects_changed_data(run_spec, tmp_path):
    model = {"num_hidden_layers": 26, "num_attention_heads": 20, "num_experts": 256}
    for profile in ("snowball-one-rack", "snowball-three-rack"):
        validate_model_geometry(model, PROFILES[profile])
    with pytest.raises(ValueError, match="Cannot partition 26 layers"):
        validate_model_geometry(model, PROFILES["one-rack"])
    (tmp_path / "train.parquet").write_bytes(b"changed mixed input")
    with build_context(BuildContext(versions=VersionCodex(default="2026.11.17.3"))):
        with pytest.raises(ValueError, match="pinned SHA-256"):
            run_from_spec(run_spec, smoke=True, config_output=tmp_path / "launch.yaml")


def test_snowball_smoke_satisfies_runtime_batch_geometry(run_spec, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    output = tmp_path / "launch.yaml"
    with build_context(BuildContext(versions=VersionCodex(default="2026.11.20.201"))):
        run_from_spec(run_spec, smoke=True, config_output=output)
    # The prior two-group default failed SkyRL validation for Snowball's policy DP4.
    validate_launch(output.read_text(), tmp_path / "compiled", model_config=b"{}")
    trainer = yaml.safe_load((tmp_path / "compiled/compiled.yaml").read_text())["skyrl"]["trainer"]
    assert sum(trainer["rollout_buffer"]["consumed_route_quotas"].values()) == trainer["train_batch_size"]
