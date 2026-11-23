# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trainer-only qualification uses the same loss normalization as the RL driver."""

import hashlib
from argparse import Namespace

import pytest
import yaml


@pytest.mark.parametrize("variant, optimizer", [("chunked", "AdamW"), ("offload", "AdamW"), ("chunked", "MuonH")])
def test_probe_derives_native_loss_context_before_worker_start(tmp_path, variant, optimizer):
    pytest.importorskip("hydra", reason="Probe admission requires the native SkyRL CPU environment")
    from cloud.iris.launch_config import load_launch_config  # noqa: PLC0415 - optional SkyRL runtime
    from cloud.iris.tests.launch_config_fixtures import raw_launch_config  # noqa: PLC0415 - optional SkyRL runtime
    from omegaconf import OmegaConf  # noqa: PLC0415 - optional SkyRL runtime
    from skyrl_train.config.driver_validation import validate_cfg  # noqa: PLC0415 - optional SkyRL runtime

    from experiments.post_training.long_loop.trainer_memory_probe_config import (  # noqa: PLC0415 - optional SkyRL runtime
        compiled_config,
    )

    source = tmp_path / "launch.yaml"
    document = raw_launch_config()
    document["skyrl"]["trainer"]["policy"] = {
        "optimizer_config": {"optimizer": optimizer, "weight_decay": 0.0, "max_grad_norm": 0.5}
    }
    source.write_text(yaml.safe_dump(document))
    reference = load_launch_config(source).skyrl
    assert OmegaConf.select(reference, "trainer.algorithm.max_seq_len") is None
    validate_cfg(reference, check_runtime_resources=False)
    config = compiled_config(
        Namespace(
            config_uri=str(source),
            config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_prefix=str(tmp_path / variant),
            variant=variant,
            max_sequence_length=131072,
        )
    )
    assert config.trainer.algorithm.max_seq_len == reference.trainer.algorithm.max_seq_len == 1024
    assert config.trainer.algorithm.resolved_group_advantage == reference.trainer.algorithm.resolved_group_advantage
    assert config.trainer.policy.megatron_config.memory_profile.max_sequence_length == 131072
    assert list(config.trainer.policy.megatron_config.memory_profile.valid_token_counts) == [85557]


def test_cpu_driver_publishes_complete_receipts_after_training_and_warm_profile(tmp_path, monkeypatch):
    pytest.importorskip("hydra", reason="Probe admission requires the native SkyRL CPU environment")
    import json  # noqa: PLC0415 - optional SkyRL CPU test
    import pickle  # noqa: PLC0415 - optional SkyRL CPU test
    from types import SimpleNamespace  # noqa: PLC0415 - optional SkyRL CPU test

    import torch  # noqa: PLC0415 - optional SkyRL CPU test
    from cloud.iris.tests.launch_config_fixtures import raw_launch_config  # noqa: PLC0415 - optional SkyRL runtime
    from skyrl_train.io import io  # noqa: PLC0415 - optional SkyRL runtime
    from skyrl_train.trainer_memory_profile import (  # noqa: PLC0415 - optional SkyRL runtime
        preserve_training_state,
        warm_state_profile,
    )

    from experiments.post_training.long_loop import (  # noqa: PLC0415 - optional SkyRL runtime
        trainer_memory_probe as probe,
    )

    source = tmp_path / "launch.yaml"
    source.write_text(yaml.safe_dump(raw_launch_config()))
    batch_path = tmp_path / "batch.pkl"
    batch_path.write_bytes(pickle.dumps(SimpleNamespace(metadata={}, inputs=torch.ones(5, 3))))
    output = tmp_path / "output"
    calls = []

    class CPUActors:
        """Replace physical Ray/GPU allocation; execute Torch updates and receipt I/O."""

        def __init__(self, config, *_args, **_kwargs):
            self.model = torch.nn.Linear(3, 2)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        def async_init_model(self, *_args, **_kwargs):
            calls.append("init")
            return [{"rank": 0, "gpu_uuid": "cpu-test-no-gpu"}]

        def step(self, tokens):
            self.optimizer.zero_grad()
            self.model(torch.ones(tokens, 3)).square().mean().backward()
            self.optimizer.step()

        def async_run_ray_method(self, group, method, *args):
            calls.append(method)
            if method == "qualify_real_batch":
                assert args[0].metadata["global_step"] == 0
                result = probe.compare_projection(
                    torch.randn(19, 3),
                    torch.randn(11, 3),
                    torch.arange(19) % 11,
                    torch.nn.functional.linear,
                    temperature=0.8,
                    fused=False,
                )
                return [{"rank": 0, "pipeline_last": True, "projection": result}]
            if method == "ppo_train":
                self.step(args[0].inputs.shape[0])
                return [
                    SimpleNamespace(
                        metadata={
                            "train_status": {
                                "policy_update_steps": 1,
                                "skipped_steps": 0,
                                "policy_loss": 0.25,
                                "raw_grad_norm": 0.5,
                            },
                            "qualification": {
                                "rank": 0,
                                "learner_seconds": 0.01,
                                "dp_weight_checksum_mismatch": 0,
                                "clipping": [{"post_clip_norm": 0.5, "norm_check_seconds": 0.001}],
                            },
                        }
                    )
                ]
            assert method == "profile_train_memory" and group == "pass_through"
            before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
            measurements = []

            def measure(tokens, run, warm):
                if warm:
                    assert self.optimizer.state
                run()
                measurements.append({"valid_tokens": tokens, "warm_optimizer": warm})
                # Use the same native atomic writer as the worker's publishing path.
                io.write_bytes_atomic(str(output / "trainer-sizing/rank-0.json"), json.dumps(measurements).encode())
                return measurements[-1]

            with preserve_training_state(
                [self.model, self.optimizer], optimizer_parameters=list(self.model.parameters()), cuda_devices=[]
            ):
                warm_state_profile([5, 9], step=self.step, measure=measure)
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            return [{"rank": 0, "measurements": measurements, "device": "CPU test only"}]

    monkeypatch.setattr(probe, "initialize_ray", lambda _config: calls.append("initialize_ray"))
    monkeypatch.setattr(probe, "PPORayActorGroup", CPUActors)
    monkeypatch.setattr(probe.ray, "remote", lambda **_kwargs: lambda worker: worker)
    monkeypatch.setattr(probe.ray, "get", lambda value: value)
    monkeypatch.setattr(probe.ray, "shutdown", lambda: calls.append("shutdown"))
    probe.driver(
        Namespace(
            config_uri=str(source),
            config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_prefix=str(output),
            batch_uri=str(batch_path),
            variant="chunked",
            max_sequence_length=131072,
        ),
        CPUActors,
    )
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["completed"] and receipt["qualification_passed"]
    assert receipt["numerical"][0]["projection"]["passed"]
    assert [item["warm_optimizer"] for item in receipt["timings"]] == [False, True]
    sizing = json.loads((output / "trainer-sizing/rank-0.json").read_bytes())
    assert [item["valid_tokens"] for item in sizing] == [5, 5, 9]
    assert [item["warm_optimizer"] for item in sizing] == [False, True, True]
    assert calls == [
        "initialize_ray",
        "init",
        "qualify_real_batch",
        "ppo_train",
        "ppo_train",
        "profile_train_memory",
        "shutdown",
    ]
    assert not list(output.rglob("*.tmp"))


@pytest.mark.parametrize("change", [{"skipped_steps": 1}, {"raw_grad_norm": float("nan")}])
def test_driver_rejects_skipped_or_nonfinite_native_updates(change):
    pytest.importorskip("hydra", reason="Probe admission requires the native SkyRL CPU environment")
    from types import SimpleNamespace  # noqa: PLC0415 - optional SkyRL CPU test

    from experiments.post_training.long_loop.trainer_memory_probe import applied_update_metrics  # noqa: PLC0415

    status = {"policy_update_steps": 1, "skipped_steps": 0, "policy_loss": 0.25, "raw_grad_norm": 0.5, **change}
    with pytest.raises(ValueError):
        applied_update_metrics([SimpleNamespace(metadata={"train_status": status})])
