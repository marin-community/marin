# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import materialized_config

import experiments.speedrun.prism_berkeley_qwen3_scaling.submission_support as submission_support
from experiments.datasets.paloma import paloma_dataset
from experiments.datasets.uncheatable import uncheatable_dataset
from experiments.llama import llama3_tokenizer
from experiments.marin_tokenizer import marin_tokenizer
from experiments.speedrun.prism_berkeley_qwen3_scaling.materialize_submission import SweepRun, select_best_runs
from experiments.speedrun.prism_berkeley_qwen3_scaling.prism_berkeley_sweep import build_config
from experiments.speedrun.prism_berkeley_qwen3_scaling.submission_support import default_speedrun


def test_select_best_runs_chooses_lowest_bpb_per_size_and_ignores_non_finished():
    runs = [
        SweepRun(
            run_name="qwen3_130m_prism_berkeley_4096_lrx0_5-aaa111",
            size="130m",
            state="finished",
            learning_rate=0.01,
            c4_en_bpb=1.20,
            c4_en_loss=3.90,
            run_info={"run_name": "a"},
        ),
        SweepRun(
            run_name="qwen3_130m_prism_berkeley_4096_lrx1-bbb222",
            size="130m",
            state="finished",
            learning_rate=0.02,
            c4_en_bpb=1.18,
            c4_en_loss=3.83,
            run_info={"run_name": "b"},
        ),
        SweepRun(
            run_name="qwen3_130m_prism_berkeley_4096_lrx1_25-ccc333",
            size="130m",
            state="crashed",
            learning_rate=0.025,
            c4_en_bpb=1.10,
            c4_en_loss=3.70,
            run_info={"run_name": "c"},
        ),
        SweepRun(
            run_name="qwen3_300m_prism_berkeley_4096_lrx0_75-ddd444",
            size="300m",
            state="finished",
            learning_rate=0.015,
            c4_en_bpb=1.06,
            c4_en_loss=3.46,
            run_info={"run_name": "d"},
        ),
        SweepRun(
            run_name="qwen3_300m_prism_berkeley_4096_lrx1-eee555",
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
    assert selected["130m"].run_name == "qwen3_130m_prism_berkeley_4096_lrx1-bbb222"
    assert selected["130m"].learning_rate == 0.02
    assert selected["300m"].run_name == "qwen3_300m_prism_berkeley_4096_lrx0_75-ddd444"
    assert selected["300m"].c4_en_bpb == 1.06


def test_select_best_runs_skips_finished_runs_without_metric():
    runs = [
        SweepRun(
            run_name="qwen3_130m_prism_berkeley_4096_lrx0_5-aaa111",
            size="130m",
            state="finished",
            learning_rate=0.01,
            c4_en_bpb=None,
            c4_en_loss=None,
            run_info={"run_name": "missing-metric"},
        ),
        SweepRun(
            run_name="qwen3_130m_prism_berkeley_4096_lrx1-bbb222",
            size="130m",
            state="finished",
            learning_rate=0.02,
            c4_en_bpb=1.18,
            c4_en_loss=3.83,
            run_info={"run_name": "best"},
        ),
    ]
    selected = select_best_runs(runs)
    assert selected["130m"].run_name == "qwen3_130m_prism_berkeley_4096_lrx1-bbb222"


def test_default_speedrun_accepts_archived_tokenized_dataset(monkeypatch):
    def _unexpected_api_call(*args, **kwargs):
        raise AssertionError("default_speedrun should not require W&B API access during graph construction")

    monkeypatch.setattr(submission_support.wandb, "Api", _unexpected_api_call)
    _, config = build_config("130m")
    train_step, result_step = default_speedrun("prism-berkeley-qwen3-130m-test", config, version="2026.07.11")
    results_config = materialized_config(result_step, "gs://test-prefix")
    train_dataset = next(dep for dep in train_step.deps if dep.name == "fineweb-edu-10B")

    assert result_step.deps == (train_step,)
    assert train_dataset.override_path == config.tokenized_dataset
    assert results_config.wandb_entity is None
    assert results_config.output_path == f"{train_step.path('gs://test-prefix')}/speedrun_results.json"


def test_default_speedrun_matches_validation_tokenizer_to_archived_train_cache():
    _, config = build_config("130m")
    train_step, _ = default_speedrun("prism-berkeley-qwen3-130m-test", config, version="2026.07.11")
    validation_steps = [step for step in train_step.deps if step.name.startswith(("paloma/", "uncheatable_eval/"))]

    assert len(validation_steps) == 23
    assert {step.version for step in validation_steps} != {"2026.06.28"}
    assert {materialized_config(step, "gs://test-prefix").tokenizer for step in validation_steps} == {llama3_tokenizer}


def test_validation_cache_identity_distinguishes_tokenizers():
    llama3_caches = [
        paloma_dataset("c4_en", tokenizer=llama3_tokenizer),
        uncheatable_dataset("bbc_news", tokenizer=llama3_tokenizer),
    ]
    marin_caches = [
        paloma_dataset("c4_en", tokenizer=marin_tokenizer),
        uncheatable_dataset("bbc_news", tokenizer=marin_tokenizer),
    ]

    llama3_identities = {(cache.name, cache.version) for cache in llama3_caches}
    marin_identities = {(cache.name, cache.version) for cache in marin_caches}

    assert llama3_identities.isdisjoint(marin_identities)
