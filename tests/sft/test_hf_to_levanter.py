# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the HF->Levanter conversion step and its wiring into ``sft_step``.

Coverage splits three ways: the ``LevanterCheckpointModel`` wiring (a conversion handle becomes a
dependency; native weights-only init, the carried architecture, and the padded tokenizer all resolve
to the conversion output; a raw path/step still needs its arch + tokenizer); the conversion step's own
config + identity (the pinned HF inputs and the resolved arch bear identity, resources do not); and the
draccus round-trip that carries the ``LmConfig`` from construction to the worker and the downstream.
"""
from __future__ import annotations

import dataclasses

import draccus
import pytest
from fray.types import ResourceConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.experiment.checkpoints import (
    HfToLevanterCheckpoint,
    HfToLevanterConfig,
    hf_to_levanter,
    resolve_lm_config,
)
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.sft.launcher import (
    ConvertedCheckpointModel,
    DatasetSpec,
    LevanterCheckpointModel,
    SFTSpec,
    sft_step,
)

_PREFIX = "gs://test-prefix"
_DATASET = DatasetSpec(
    slug="ds",
    hf_dataset_id="some-org/some-dataset",
    revision="main",
    adapter_kwargs=dict(conversation_column="messages"),
    weight=1.0,
)


def _tiny_arch() -> LmConfig:
    """A small concrete architecture that stands in for a resolved HF config (no network)."""
    return LmConfig.get_choice_class("llama")(num_layers=2, num_heads=2, num_kv_heads=2, hidden_dim=32, max_seq_len=64)


def _fake_conversion(arch: LmConfig | None = None) -> HfToLevanterCheckpoint:
    """An ``HfToLevanterCheckpoint`` whose step is a bare handle — enough to exercise the wiring
    without running the conversion (only the step's *path* is referenced downstream)."""
    step: ArtifactStep[LevanterCheckpoint] = ArtifactStep(
        name="checkpoints/hf-to-levanter/qwen-qwen3-0.6b-qwen3",
        version="2026.07.17",
        artifact_type=LevanterCheckpoint,
        run=lambda config: None,
        build_config=lambda ctx: {},
    )
    return HfToLevanterCheckpoint(step=step, model=arch or _tiny_arch())


def _spec(model) -> SFTSpec:
    return SFTSpec(
        name="checkpoints/test-sft",
        version="2026.07.17-dev",
        model=model,
        chat_template="{% for m in messages %}{% generation %}{{ m['content'] }}{% endgeneration %}{% endfor %}",
        datasets=[_DATASET],
        optimizer=AdamConfig(learning_rate=1e-5),
        num_train_steps=1,
    )


def test_conversion_handle_is_a_dependency_and_wires_native_weights_only_init():
    """A conversion handle makes the step a dep; init is weights-only, arch + tokenizer come from it."""
    conv = _fake_conversion()
    step = sft_step(_spec(ConvertedCheckpointModel(conversion=conv)), ResourceConfig.with_cpu())

    assert conv.step in step.deps
    train_config = materialized_config(step, _PREFIX).train_config

    conv_out = conv.step.path(_PREFIX)
    # Native weights-only init points at the checkpoint root (the model is a `model/` subtree there,
    # loaded with subpath="model"); the HF/full-state fields are off.
    assert train_config.initialize_model_from_checkpoint_path == conv_out
    assert train_config.initialize_from_hf is False
    assert train_config.initialize_from_checkpoint_path is None
    assert train_config.use_hf_model_config is False
    # The carried architecture builds the pytree; the padded tokenizer is the conversion output root.
    assert train_config.model == conv.model
    assert train_config.data.tokenizer == conv_out


def test_raw_path_requires_explicit_arch_and_tokenizer():
    """A static checkpoint dir carries no arch/tokenizer, so both must be supplied."""
    arch = _tiny_arch()
    model = LevanterCheckpointModel(init_from="gs://staged/ckpt", model=arch, tokenizer_path="gs://tok")
    train_config = materialized_config(sft_step(_spec(model), ResourceConfig.with_cpu()), _PREFIX).train_config
    assert train_config.initialize_model_from_checkpoint_path == "gs://staged/ckpt"
    assert train_config.data.tokenizer == "gs://tok"
    assert train_config.model == arch


def test_epoch_chat_tokenize_declares_the_conversion_dependency():
    """The epoch path tokenizes off-pod, resolving the tokenizer through the conversion step, so the
    chat-tokenize step must declare it as a dep (else StepContext rejects the undeclared reference)."""
    conv = _fake_conversion()
    spec = dataclasses.replace(
        _spec(ConvertedCheckpointModel(conversion=conv)), num_train_steps=None, num_train_epochs=1
    )
    step = sft_step(spec, ResourceConfig.with_cpu())

    chat_caches = [dep for dep in step.deps if dep.artifact_type is TokenizedCache]
    assert chat_caches, "epoch path should materialize a chat_tokenize cache"
    # A prefix free of the forbidden train-url tokens ("test") the tokenize config rejects.
    prefix = "gs://sft-prefix"
    for cache in chat_caches:
        assert conv.step in cache.deps
        # Materializing resolves the tokenizer through the (now declared) conversion step.
        assert materialized_config(cache, prefix).tokenizer == conv.step.path(prefix)


def test_lm_config_draccus_round_trip_selects_subclass_by_model_type():
    """The carried arch survives the encode/decode used to ship it to the worker + downstream."""
    arch = _tiny_arch()
    config = HfToLevanterConfig(
        hf_id="Qwen/Qwen3-0.6B",
        hf_revision="deadbeef",
        model_type="llama",
        tokenizer="Qwen/Qwen3-0.6B",
        tokenizer_revision="deadbeef",
        compute_dtype="bfloat16",
        model_config_json=draccus.encode(arch),
        output_path="gs://out",
        resources=ResourceConfig.with_cpu(),
    )
    decoded = draccus.decode(LmConfig.get_choice_class(config.model_type), config.model_config_json)
    assert decoded == arch


@pytest.mark.slow
def test_hf_to_levanter_resolves_real_arch_and_is_identity_stable():
    """Against a real small model: the arch is read from the HF config (not the class default) and the
    conversion step's identity tracks the pinned inputs but not where it runs."""
    conv = hf_to_levanter("Qwen/Qwen3-0.6B", model_type="qwen3", hf_revision="main", version="2026.07.17")
    # 0.6B differs from the Qwen3Config class default, so a real resolution must have happened.
    assert conv.model != LmConfig.get_choice_class("qwen3")()
    assert conv.model == resolve_lm_config("qwen3", "Qwen/Qwen3-0.6B", "main")

    config = materialized_config(conv.step, _PREFIX)
    assert isinstance(config, HfToLevanterConfig)
    assert config.hf_id == "Qwen/Qwen3-0.6B"
    assert config.output_path == conv.step.path(_PREFIX)

    base = conv.step.fingerprint()
    big = hf_to_levanter(
        "Qwen/Qwen3-0.6B",
        model_type="qwen3",
        hf_revision="main",
        version="2026.07.17",
        resources=ResourceConfig.with_cpu(cpu=32, ram="256g"),
    )
    assert big.step.fingerprint() == base  # resources are a runtime choice
