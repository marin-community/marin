from dataclasses import replace

from skyrl_train.trajectory_runners.trajectory_retention import _orphan_reasons
from skyrl_train.trajectory_runners.trajectory_retention_config import parse_trajectory_retention_config


def test_fraction_sampling_is_independent_of_outcome_and_eval_keeps_everything():
    config = parse_trajectory_retention_config({
        "enabled": True, "output_path": "/tmp/traces", "run_id": "retention-test",
        "phases": ["train", "eval"], "sample_count_per_step": 0,
        "sample_fraction": 0.1, "eval_sample_fraction": 1.0,
        "always_retain_failures": False, "always_retain_non_terminating": False,
        "always_retain_loops": False, "max_bytes_per_step": None, "max_bytes_per_run": None,
    })
    retained = []
    for i in range(1000):
        value = {"run_id": "retention-test", "phase": "train", "global_step": 7,
                 "trajectory": {"instance_id": str(i), "repetition_id": 0},
                 "reward": {"outcome": 1}, "response": {"stop_reason": "stop"}}
        first = _orphan_reasons(value, config)
        value["reward"]["outcome"] = 0
        value["response"] = {"stop_reason": "length", "loop_spans": [1]}
        assert _orphan_reasons(value, config) == first
        retained.append(bool(first))
        value["phase"] = "eval"
        assert _orphan_reasons(value, config)
        assert not _orphan_reasons(value, replace(config, eval_sample_fraction=0))
    assert any(retained) and not all(retained)
