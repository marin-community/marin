# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.speedrun.adamh_qwen3_scaling.materialize_submission import SweepRun, select_best_runs
from experiments.speedrun.adamh_qwen3_scaling.adamh_sweep import build_config
from experiments.speedrun.adamh_qwen3_scaling.submission_support import default_speedrun


def test_select_best_runs_chooses_lowest_bpb_per_size_and_ignores_non_finished():
    runs = [
        SweepRun(
            run_name="qwen3_130m_adamh_4096_lrx0_5-aaa111",
            size="130m",
            state="finished",
            learning_rate=0.01,
            c4_en_bpb=1.20,
            c4_en_loss=3.90,
            run_info={"run_name": "a"},
        ),
        SweepRun(
            run_name="qwen3_130m_adamh_4096_lrx1-bbb222",
            size="130m",
            state="finished",
            learning_rate=0.02,
            c4_en_bpb=1.18,
            c4_en_loss=3.83,
            run_info={"run_name": "b"},
        ),
        SweepRun(
            run_name="qwen3_130m_adamh_4096_lrx1_25-ccc333",
            size="130m",
            state="crashed",
            learning_rate=0.025,
            c4_en_bpb=1.10,
            c4_en_loss=3.70,
            run_info={"run_name": "c"},
        ),
        SweepRun(
            run_name="qwen3_300m_adamh_4096_lrx0_75-ddd444",
            size="300m",
            state="finished",
            learning_rate=0.015,
            c4_en_bpb=1.06,
            c4_en_loss=3.46,
            run_info={"run_name": "d"},
        ),
        SweepRun(
            run_name="qwen3_300m_adamh_4096_lrx1-eee555",
            size="300m",
            state="finished",
            learning_rate=0.02,
            c4_en_bpb=1.08,
            c4_en_loss=3.50,
            run_info={"run_name": "e"},
        ),
    ]

    selected = select_best_runs(runs)

    assert set(selected) == {"130m", "300m"}
    assert selected["130m"].run_name == "qwen3_130m_adamh_4096_lrx1-bbb222"
    assert selected["130m"].learning_rate == 0.02
    assert selected["300m"].run_name == "qwen3_300m_adamh_4096_lrx0_75-ddd444"
    assert selected["300m"].c4_en_bpb == 1.06


def test_select_best_runs_skips_finished_runs_without_metric():
    runs = [
        SweepRun(
            run_name="qwen3_130m_adamh_4096_lrx0_5-aaa111",
            size="130m",
            state="finished",
            learning_rate=0.01,
            c4_en_bpb=None,
            c4_en_loss=None,
            run_info={"run_name": "missing-metric"},
        ),
        SweepRun(
            run_name="qwen3_130m_adamh_4096_lrx1-bbb222",
            size="130m",
            state="finished",
            learning_rate=0.02,
            c4_en_bpb=1.18,
            c4_en_loss=3.83,
            run_info={"run_name": "best"},
        ),
    ]
    selected = select_best_runs(runs)
    assert selected["130m"].run_name == "qwen3_130m_adamh_4096_lrx1-bbb222"


def test_default_speedrun_accepts_archived_tokenized_dataset():
    _, config = build_config("130m")
    steps = default_speedrun("adamh-qwen3-130m-test", config)
    assert len(steps) == 2
