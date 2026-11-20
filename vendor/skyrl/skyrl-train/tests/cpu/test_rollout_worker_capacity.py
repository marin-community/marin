"""Reject the observed pinned-worker overbooking before model initialization."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.entrypoints.main_base import BasePPOExp


class TokenizerReached(RuntimeError):
    pass


class AdmissionProbeExp(BasePPOExp):
    def get_tokenizer(self):
        raise TokenizerReached("capacity check completed before tokenizer/model initialization")


@pytest.mark.parametrize("cpus,static_preferences", [(8, False), (2, False), (8, True)])
def test_pinned_worker_admission_uses_the_driver_node_before_initialization(monkeypatch, cpus, static_preferences):
    cfg = OmegaConf.create(
        {
            "environment": {"env_class": "preference_pair" if static_preferences else "taskcompendium"},
            "trainer": {"evaluation_runner": "training"},
            "trajectory_runner": {
                "rollout_workers": {
                    "num_workers": 32,
                    "cpus_per_worker": cpus,
                    "executor_threads": 256,
                    "progress_timeout_seconds": 21600,
                    "start_interval_seconds": 2,
                }
            },
        }
    )
    monkeypatch.setattr("ray.get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "driver-node"))
    monkeypatch.setattr(
        "ray.nodes",
        lambda: [
            {"NodeID": "driver-node", "Alive": True, "Resources": {"CPU": 144}},
            {"NodeID": "other-nodes", "Alive": True, "Resources": {"CPU": 6768}},
        ],
    )
    if cpus == 8 and not static_preferences:
        with pytest.raises(ValueError):
            AdmissionProbeExp(cfg)
    else:
        with pytest.raises(TokenizerReached):
            AdmissionProbeExp(cfg)
