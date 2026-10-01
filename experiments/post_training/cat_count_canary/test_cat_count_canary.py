# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from pathlib import Path

import click
import duckdb
import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.post_training.cat_count_canary.data import (
    DEFAULT_TRAIN_NS,
    EXTRAPOLATION_NS,
    HELDOUT_NS,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    CatCountDataConfig,
    cat_count_rows,
    write_cat_count_parquet,
)
from experiments.post_training.cat_count_canary.launcher import MODELS, build_run, training_config


def test_procedural_rows_balance_and_holdout_exclusion():
    config = CatCountDataConfig("/tmp/unused", DEFAULT_TRAIN_NS, 64 * 60, 17)
    train, validation = cat_count_rows(config)

    counts = Counter(row["extra_info"]["n"] for row in train)
    assert set(counts) == set(DEFAULT_TRAIN_NS)
    assert set(counts.values()) == {240}
    for start in range(0, len(train), 64):
        assert {row["extra_info"]["n"] for row in train[start : start + 64]} == set(DEFAULT_TRAIN_NS)
    assert {row["extra_info"]["n"] for row in validation} == set(DEFAULT_TRAIN_NS) | set(HELDOUT_NS) | set(
        EXTRAPOLATION_NS
    )
    assert all(row["data_source"] == f"cat_count_n{row['extra_info']['n']}" for row in train + validation)
    assert all(row["env_class"] == "cat_count" for row in train + validation)


def test_parquet_writes_typed_rows(tmp_path):
    write_cat_count_parquet(CatCountDataConfig(str(tmp_path), DEFAULT_TRAIN_NS, 32, 17))
    train = duckdb.read_parquet(str(tmp_path / TRAIN_FILENAME))
    validation = duckdb.read_parquet(str(tmp_path / VALIDATION_FILENAME))
    assert train.count("*").fetchone() == (32,)
    assert validation.count("*").fetchone() == (22,)
    assert train.columns == ["data_source", "prompt", "env_class", "reward_spec", "extra_info"]
    for source, prompt, env_class, reward_spec, extra_info in train.fetchall():
        n = extra_info["n"]
        assert source == f"cat_count_n{n}"
        assert prompt == [
            {
                "content": f"Reply with the word cat exactly {n} times, separated by single spaces. Nothing else.",
                "role": "user",
            }
        ]
        assert env_class == "cat_count"
        assert reward_spec == {"ground_truth": n, "method": "rule"}
        assert n not in (*HELDOUT_NS, *EXTRAPOLATION_NS)


def test_both_lanes_render_megatron_launch_with_complete_custom_eval_mix():
    train_ns = (*DEFAULT_TRAIN_NS, 32)
    launches = {}
    for lane in ("async", "sync"):
        run = build_run(version="2026.09.26", preset="gate", lane=lane, train_ns=train_ns)
        launch_config = run.build_config(StepContext.for_fingerprint(run.runtime_args, run.deps))
        launches[lane] = yaml.safe_load(launch_config.launch_config_yaml)
        data_step = next(dep for dep in run.deps if dep.name.startswith("documents/cat-count-canary/"))
        data_config = data_step.build_config(StepContext.for_fingerprint(data_step.runtime_args, data_step.deps))
        train, validation = cat_count_rows(data_config)
        assert len(validation) == 23
        assert len(train) >= 60 * 64 + (128 if lane == "async" else 0)

        launch = launches[lane]
        assert launch["skyrl"]["trainer"]["eval_batch_size"] == len(validation)

    assert (
        launches["async"]["skyrl"]["generator"]["chat_template"]
        == launches["sync"]["skyrl"]["generator"]["chat_template"]
    )


@pytest.mark.parametrize(
    "setting",
    (
        "trainer.policy={megatron_config: {tensor_model_parallel_size: 2}}",
        "trainer.seed=23",
        "trainer.max_ckpts_to_keep=2",
        "generator.require_exact_chat_transport=false",
    ),
)
def test_owned_settings_reject_parent_and_canonical_overrides(setting):
    with pytest.raises(click.BadParameter):
        training_config(settings=(setting,))


def test_model_pins_and_distinct_artifact_identities():
    for model, choice in MODELS.items():
        download = choice.step.build_config(StepContext.for_fingerprint(choice.step.runtime_args, choice.step.deps))
        run = build_run(version="2026.09.26", preset="dry", model=model)
        config = run.build_config(StepContext.for_fingerprint(run.runtime_args, run.deps))
        assert download.revision == config.model.tokenizer_revision
    async_run = build_run(version="2026.09.26", preset="dry", lane="async")
    sync_run = build_run(version="2026.09.26", preset="dry", lane="sync")
    changed_run = build_run(version="2026.09.26", preset="dry", settings=("trainer.policy.optimizer_config.lr=5e-6",))
    assert len({async_run.name, sync_run.name, changed_run.name}) == 3
    assert any(dep.name == MODELS["qwen2.5-0.5b-instruct"].step.name for dep in async_run.deps)


def test_downloaded_model_root_resolves_as_the_hf_snapshot(tmp_path: Path, monkeypatch):
    model_step = MODELS["qwen2.5-0.5b-instruct"].step
    model_root = Path(model_step.path(str(tmp_path)))
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text("{}")
    (model_root / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr("marin.rl.skyrl.skyrl_temporary_run_path", lambda *_args, **_kwargs: str(tmp_path / "scratch"))

    run = build_run(version="2026.09.26", preset="dry", job_timeout_seconds=1800)
    config = run.build_config(
        StepContext.for_run(
            output_path=str(tmp_path / "output"),
            prefix=str(tmp_path),
            runtime_args=run.runtime_args,
            deps=run.deps,
        )
    )

    assert config.model.uri == str(model_root)
