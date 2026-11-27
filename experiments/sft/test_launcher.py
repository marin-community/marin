# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import draccus
import numpy as np
import pytest
from fray.current_client import set_current_client
from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, dataset_for_component
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from levanter.utils.thread_utils import blocking_wait
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.step_status import StatusFile
from marin.training.training import DEFAULT_HF_CHECKPOINTS_PATH
from rigging.filesystem.step_status import STATUS_SUCCESS
from rigging.timing import Duration

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.post_training.snowball_sft import (
    EXP7_262K_32NODE,
    EXP7_TOKENIZER,
    HF_MODEL,
    HF_REVISION,
    MIXTURE_262K_32NODE,
    MIXTURE_SPEC,
    MIXTURE_TOKENIZER,
    OPENTHOUGHTS_32K_4NODE,
    OPENTHOUGHTS_SPEC,
    SNOWBALL_SFT,
)
from experiments.sft import launcher
from experiments.sft.launcher import PreparedData, PreparedModel


def invoke_sft(monkeypatch, presets, *args):
    monkeypatch.setattr(sys, "argv", ["snowball_sft", *args])
    with pytest.raises(SystemExit) as exited:
        launcher.main(presets)
    return exited.value.code


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (None, "Build a Hugging Face export for the GrugModel trainer"),
        ({"xla_flags": ("--flag",)}, "xla_flags"),
        ({"per_device_parallelism": 1}, "per_device_parallelism"),
        ({"timeout": Duration.from_minutes(12)}, "timeout"),
    ],
)
def test_sft_refuses_incompatible_grug_settings(overrides, error):
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    spec = replace(OPENTHOUGHTS_SPEC, model=grug, **(overrides or {}))
    with pytest.raises(ValueError, match=error):
        if overrides is None:
            replace(OPENTHOUGHTS_32K_4NODE, recipe=spec).model()
        else:
            launcher.sft_step(spec, OPENTHOUGHTS_32K_4NODE.resources).fingerprint()


def test_sft_resolves_an_export_from_a_structural_model_source(tmp_path):
    base = OPENTHOUGHTS_SPEC.model
    model = cast(
        launcher.ModelSource,
        SimpleNamespace(
            tokenizer_cache_key=base.tokenizer_cache_key,
            resolve_tokenizer=base.resolve_tokenizer,
            run=base.run,
            init_deps=base.init_deps,
            build_train_config=base.build_train_config,
            hf_export_path=lambda _: "published/final",
        ),
    )
    resources = OPENTHOUGHTS_32K_4NODE.resources
    exported = replace(OPENTHOUGHTS_SPEC, model=model).trained(resources)
    prefix = str(tmp_path)
    ctx = StepContext.for_run(str(tmp_path / "eval"), prefix, deps=exported.deps())
    assert exported.resolve(ctx).uri == f"{exported.step.path(prefix)}/published/final"


def test_prepared_sft_submits_the_recipe_timeout(tmp_path):
    weights = ArtifactStep.adopt("tests/weights", "2026.11.20", str(tmp_path / "weights"), kind=Artifact)
    timeout = Duration.from_minutes(12)
    spec = replace(OPENTHOUGHTS_SPEC, model=PreparedModel(weights), timeout=timeout)
    step = spec.trained(OPENTHOUGHTS_32K_4NODE.resources).step
    requests = []

    def submit(request):
        requests.append(request)
        return SimpleNamespace(wait=lambda **_kwargs: None)

    with set_current_client(SimpleNamespace(submit=submit)):
        step.run(materialized_config(step, str(tmp_path)))

    [request] = requests
    assert request.timeout == timeout


def test_prepared_full_token_loss_ignores_assistant_weights_and_preserves_padding(tmp_path):
    source = tmp_path / "release"
    source.mkdir()
    original = LmDataConfig(
        tokenizer=MIXTURE_TOKENIZER,
        auto_build_caches=False,
        components={
            "sft": DatasetComponent(
                cache_dir="/regional/cache",
                flat_cache=True,
                format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
            )
        },
        num_validation_sequences={"sft": 8},
    )
    document = draccus.encode(original)
    document["components"]["sft"]["format"].pop("loss_weight_transform")
    (source / "data.json").write_text(json.dumps(document))
    prepared = PreparedData(
        ArtifactStep.adopt("tests/release", "2026.11.25", str(source)), "data.json", MIXTURE_TOKENIZER
    )
    spec = replace(
        MIXTURE_SPEC,
        prepared_data=prepared,
        prepared_data_format=TextLmDatasetFormat(),
        pack=256,
        pack_lookahead_window=128,
    )
    step = spec.trained(MIXTURE_262K_32NODE.resources).step
    data = materialized_config(step, str(tmp_path)).train_config.data
    [component] = data.components.values()
    rows = [
        {
            "input_ids": np.array([11, 12, 13, 14, 15], dtype=np.int32),
            "loss_weights": np.array([0, 0, 1, 1, 0], dtype=np.float32),
        },
        {"input_ids": np.array([21, 22, 23], dtype=np.int32), "loss_weights": np.array([0, 1, 0], dtype=np.float32)},
    ]
    with SerialCacheWriter(str(tmp_path / "cache"), rows[0]) as writer:
        writer.write_batch(rows)
    dataset = dataset_for_component(
        component,
        Axis("position", 12),
        writer.result(),
        eos_id=None,
        block_cross_document_attention=True,
        remainder="pad",
    )
    [example] = blocking_wait(dataset.get_batch([0]))
    np.testing.assert_array_equal(example.tokens, [11, 12, 13, 14, 15, 21, 22, 23, 0, 0, 0, 0])
    np.testing.assert_array_equal(example.loss_weight, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert data.num_validation_sequences == original.num_validation_sequences
    assert json.loads((source / "data.json").read_text()) == document
    assert (
        step.fingerprint()
        != replace(spec, prepared_data_format=None).trained(MIXTURE_262K_32NODE.resources).step.fingerprint()
    )


def test_exp7_resolves_exp6_sft_controls_at_the_native_boundary(tmp_path):
    reference = json.loads(
        (Path(__file__).parents[1] / "post_training/snowball_sft_data/exp6-sft-controls.json").read_text()
    )
    data = LmDataConfig(tokenizer=EXP7_TOKENIZER, auto_build_caches=False)
    (tmp_path / "data.json").write_text(json.dumps(draccus.encode(data)))
    binding = PreparedData(ArtifactStep.adopt("tests/exp7", "2026.11.25", str(tmp_path)), "data.json", EXP7_TOKENIZER)
    spec = EXP7_262K_32NODE.recipe.with_data_mix(binding).with_steps(1893, version="2026.11.25.7")
    step = spec.trained(EXP7_262K_32NODE.resources).step
    config = materialized_config(step, str(tmp_path)).train_config
    actual = {
        key: draccus.encode(getattr(config, key))
        for key in reference
        if not key.startswith("source_") and key != "trainer"
    }
    actual["trainer"] = {
        key: (
            json.loads(canonical_json(getattr(config.trainer, key)))
            if key in {"mp", "progress_watchdog"}
            else draccus.encode(getattr(config.trainer, key))
        )
        for key in reference["trainer"]
    }
    actual["snowball_sft"]["special_token_lr_multiplier"] = actual["snowball_sft"].pop("special_token_update_multiplier")
    for key in ("model", "optimizer"):
        actual[key].pop("type", None)
    for key, value in reference.items():
        if key.startswith("source_"):
            continue
        if key == "optimizer":
            value = {**value, "decay": reference["source_cooldown"]["fraction"], "lr_schedule": "linear"}
        if key == "trainer":
            assert {name: actual[key][name] for name in value} == value
        else:
            assert actual[key] == value, key
    for updates in (1893, 1679):
        adjusted = spec.with_steps(updates, version="2026.11.25.7")
        native = materialized_config(adjusted.trained(EXP7_262K_32NODE.resources).step, str(tmp_path)).train_config
        cooldown = reference["source_cooldown"]
        tail = int(cooldown["fraction"] * updates)
        begin = updates - tail
        points = (0, 15, 30, begin, begin + tail // 2, updates)
        scales = [
            min(point / 30, 1)
            * max(cooldown["min_lr_ratio"], 1 - max(point - begin, 0) / tail * (1 - cooldown["min_lr_ratio"]))
            for point in points
        ]
        for group in ("learning_rate", "adam_lr"):
            peak = reference["optimizer"][group]
            curve = native.optimizer.lr_scheduler(native.trainer.num_train_steps, override_lr=peak)
            np.testing.assert_allclose([curve(point) for point in points], np.asarray(scales) * peak, rtol=1e-6)
    requests = []
    with set_current_client(
        SimpleNamespace(
            submit=lambda request: (requests.append(request) or SimpleNamespace(wait=lambda **_kwargs: None))
        )
    ):
        step.run(materialized_config(step, str(tmp_path)))
    [request] = requests
    # The child leaves the band unset; the coordinator flow resolves it to the parent's band.
    assert request.priority == 0
    assert request.max_retries_failure == 2 and request.max_task_failures == 64
    for name, expected in reference["source_runtime_env"].items():
        observed = request.environment.env_vars[name]
        if name == "XLA_FLAGS":
            observed_flags = {flag.partition("=")[0]: flag.partition("=")[2] for flag in observed.split()}
            expected_flags = {flag.partition("=")[0]: flag.partition("=")[2] for flag in expected.split()}
            assert observed_flags == expected_flags
        else:
            assert observed == expected, name
    assert request.resources.target_cluster == "cw-rno2a"
    assert request.resources.replicas == 32 and request.resources.device.count == 8
    assert request.timeout is None


def test_openthoughts_keeps_one_conversation_per_context_window(tmp_path):
    training = materialized_config(OPENTHOUGHTS_32K_4NODE.model().step, str(tmp_path)).train_config
    assert training.data.tokenizer == training.initialize_from_hf
    [component] = training.data.components.values()
    conversations = (
        {"input_ids": np.arange(10, 22, dtype=np.int32), "assistant_masks": np.arange(12) >= 2},
        {"input_ids": np.array([31, 32, 33], dtype=np.int32), "assistant_masks": np.array([False, True, True])},
        {"input_ids": np.array([41, 42, 43], dtype=np.int32), "assistant_masks": np.array([False, True, True])},
    )
    with SerialCacheWriter(str(tmp_path / "cache"), conversations[0]) as writer:
        writer.write_batch(conversations)
    dataset = dataset_for_component(
        component,
        Axis("position", 8),
        writer.result(),
        eos_id=None,
        block_cross_document_attention=True,
        remainder="drop",
    )
    assert blocking_wait(dataset.async_len()) == 3
    long, first_short, second_short = blocking_wait(dataset.get_batch([0, 1, 2]))
    np.testing.assert_array_equal(long.tokens, np.arange(10, 18))
    np.testing.assert_array_equal(np.flatnonzero(long.loss_weight), np.arange(1, 7))
    np.testing.assert_array_equal(first_short.tokens, [31, 32, 33, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(second_short.tokens, [41, 42, 43, 0, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"resources": OPENTHOUGHTS_32K_4NODE.resources}, "mesh does not fit SFT resources"),
        ({"recipe": replace(MIXTURE_SPEC, batch_size=7)}, "must be divisible by the mesh microbatch"),
    ],
    ids=["mesh", "batch"],
)
def test_sft_plan_refuses_invalid_gpu_geometry(monkeypatch, capsys, changes, message):
    preset = replace(MIXTURE_262K_32NODE, **changes)
    assert invoke_sft(monkeypatch, (preset,), "--preset", preset.name) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        ("declared", r"prepared_data\.tokenizer must match"),
        ("manifest", "file tokenizer does not match"),
        ("resolved", "resolved SFT model tokenizer"),
        ("epochs", "prepared_data requires num_train_steps"),
    ],
)
def test_prepared_data_requires_matching_tokenizers_and_fixed_steps(tmp_path, scenario, error):
    artifact = ArtifactStep.adopt(name="mixture", version="2026.11.19", source=str(tmp_path), kind=Artifact)
    model = OPENTHOUGHTS_SPEC.model
    source = PreparedData(artifact, "mix.json", model.tokenizer_cache_key())
    if scenario == "resolved":
        weights = ArtifactStep.adopt(
            name="weights", version="2026.11.19", source=str(tmp_path / "weights"), kind=Artifact
        )
        model = PreparedModel(weights)
        source = replace(source, tokenizer=model.tokenizer_cache_key())
    if scenario == "declared":
        source = replace(source, tokenizer="different-tokenizer")
    tokenizer = "different-tokenizer" if scenario == "manifest" else source.tokenizer
    steps = {"num_train_steps": None, "num_train_epochs": 1} if scenario == "epochs" else {}
    (tmp_path / source.file).write_text(json.dumps(draccus.encode(LmDataConfig(tokenizer=tokenizer))))
    with pytest.raises(ValueError, match=error):
        spec = replace(OPENTHOUGHTS_SPEC, model=model, datasets=(), prepared_data=source, **steps)
        materialized_config(replace(OPENTHOUGHTS_32K_4NODE, recipe=spec).model().step, str(tmp_path))


def test_snowball_262k_mixture_binds_the_published_mixture(tmp_path, monkeypatch):
    prefix = str(tmp_path)
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    model = MIXTURE_262K_32NODE.model()
    manifest, mirror = model.step.deps
    requests = []
    configs = []

    def submit(request):
        requests.append(request)
        request.entrypoint.callable_entrypoint.callable()
        return SimpleNamespace(wait=lambda **kwargs: None)

    monkeypatch.setattr("marin.experiment.train.run_levanter_train_lm", configs.append)
    args = ("--preset", MIXTURE_262K_32NODE.name)
    with set_current_client(SimpleNamespace(submit=submit)):
        assert invoke_sft(monkeypatch, SNOWBALL_SFT, *args) == 0
        assert not list(tmp_path.iterdir())
        assert not requests

        mirror_path = mirror.path(prefix)
        mirror_config = materialized_config(mirror, prefix)
        assert (mirror_config.hf_dataset_id, mirror_config.revision) == (HF_MODEL, HF_REVISION)
        StatusFile(mirror_path, worker_id="test-mirror").write_status(STATUS_SUCCESS)
        assert invoke_sft(monkeypatch, SNOWBALL_SFT, *args, "--run") == 0

    with pytest.raises(ValueError, match="SHA-256 pin"):
        manifest.run({**materialized_config(manifest, prefix), "sha256": "0" * 64})
    [pod_config] = configs
    config = pod_config.train_config
    assert config.initialize_from_hf == mirror_path
    assert config.data.tokenizer == MIXTURE_TOKENIZER
    assert model.relative_path == f"{DEFAULT_HF_CHECKPOINTS_PATH}/step-203"

    [request] = requests
    assert request.timeout == Duration.from_seconds(72_000)
    flags = request.environment.env_vars["XLA_FLAGS"].split()
    assert "--xla_gpu_shard_autotuning=false" in flags
