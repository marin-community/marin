# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import pytest
from transformers import AutoTokenizer

import haliax as hax
from haliax.quantization import apply_updates, partition_for_grad_overwrite
from haliax.partitioning import pspec_for
from haliax.jax_utils import is_jax_array_like

from levanter.data.text import DpoExample, PreferenceChatProcessor, PreferencePairDataset
from levanter.main.train_dpo import (
    DpoModel,
    _bool_tree_like,
    _logp_sum,
    _policy_model_for_hf_save,
    dpo_loss_from_logps,
)
from levanter.metrics import Metric
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmExample
from levanter.optim import AdamConfig
from levanter.optim.model_averaging import ModelAveraging
from levanter.store.cache import SerialCacheWriter
from levanter.trainer_state import TrainerState, trainables_only
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode


MODEL_NAME = "stanford-crfm/marin-tokenizer"


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return Path(tokenizer.name_or_path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {exc}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def _load_tokenizer(tokenizer_path: Path):
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def _tiny_gpt2_config() -> Gpt2Config:
    return Gpt2Config(
        max_seq_len=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )


def _namedarray_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree, is_leaf=lambda x: isinstance(x, hax.NamedArray))
        if isinstance(leaf, hax.NamedArray)
    ]


def _preference_exemplar() -> dict[str, np.ndarray]:
    return {
        "chosen_input_ids": np.zeros((0,), dtype=np.int32),
        "chosen_assistant_masks": np.zeros((0,), dtype=np.int32),
        "rejected_input_ids": np.zeros((0,), dtype=np.int32),
        "rejected_assistant_masks": np.zeros((0,), dtype=np.int32),
    }


def _make_preference_row(chosen_ids: list[int], rejected_ids: list[int]) -> dict[str, np.ndarray]:
    return {
        "chosen_input_ids": np.array(chosen_ids, dtype=np.int32),
        "chosen_assistant_masks": np.ones((len(chosen_ids),), dtype=np.int32),
        "rejected_input_ids": np.array(rejected_ids, dtype=np.int32),
        "rejected_assistant_masks": np.ones((len(rejected_ids),), dtype=np.int32),
    }


def _write_preference_cache(tmpdir: str, rows: list[dict[str, np.ndarray]]):
    with SerialCacheWriter(tmpdir, _preference_exemplar()) as writer:
        writer.write_batch(rows)
    return writer.result()


def _build_policy_reference(config: Gpt2Config):
    Vocab = hax.Axis("vocab", 32)
    policy = config.build(Vocab, key=jrandom.PRNGKey(0))
    reference = config.build(Vocab, key=jrandom.PRNGKey(1))
    return Vocab, policy, reference


def test_dpo_trainable_filter_excludes_reference_params():
    config = _tiny_gpt2_config()
    _, policy, reference = _build_policy_reference(config)
    model = DpoModel(policy=policy, reference=reference)

    trainable_filter = DpoModel(
        policy=_bool_tree_like(policy, True),
        reference=_bool_tree_like(reference, False),
    )
    trainable = trainables_only(model, trainable_filter)

    assert parameter_count(trainable) == parameter_count(policy)


def test_dpo_trainable_filter_has_no_namedarray_nones():
    config = _tiny_gpt2_config()
    _, policy, reference = _build_policy_reference(config)
    model = DpoModel(policy=policy, reference=reference)

    trainable_filter = DpoModel(
        policy=_bool_tree_like(policy, True),
        reference=_bool_tree_like(reference, False),
    )
    trainable = trainables_only(model, trainable_filter)

    for leaf in _namedarray_leaves(trainable):
        assert leaf.array is not None


def test_policy_model_for_hf_save_unwraps_dpo_model():
    config = _tiny_gpt2_config()
    _, policy, reference = _build_policy_reference(config)
    model = DpoModel(policy=policy, reference=reference)

    assert _policy_model_for_hf_save(model) is policy
    assert _policy_model_for_hf_save(policy) is policy


def test_trainer_state_init_with_dpo_model():
    config = _tiny_gpt2_config()
    _, policy, reference = _build_policy_reference(config)
    model = DpoModel(policy=policy, reference=reference)

    trainable_filter = DpoModel(
        policy=_bool_tree_like(policy, True),
        reference=_bool_tree_like(reference, False),
    )

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(0),
        is_trainable=trainable_filter,
        mp=jmp.get_policy("f32"),
    )

    assert isinstance(state.model, DpoModel)


def test_preference_chat_processor_skips_invalid_rows(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {"chosen": [], "rejected": []},
        {
            "chosen": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "rejected": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "No."},
            ],
        },
    ]

    result = processor(batch)
    assert len(result) == 1


def test_dpo_loss_decreases_with_margin():
    Batch = hax.Axis("batch", 2)
    delta_ref = hax.zeros(Batch)

    delta_small = hax.named(jnp.array([0.1, 0.2], dtype=jnp.float32), Batch)
    delta_large = hax.named(jnp.array([1.0, 1.2], dtype=jnp.float32), Batch)

    loss_small, metrics_small = dpo_loss_from_logps(delta_small, delta_ref, beta=1.0)
    loss_large, metrics_large = dpo_loss_from_logps(delta_large, delta_ref, beta=1.0)

    assert float(loss_large) < float(loss_small)
    assert float(metrics_large["dpo_accuracy"]) >= float(metrics_small["dpo_accuracy"])


def test_dpo_metrics_are_explicit_metrics():
    Batch = hax.Axis("batch", 2)
    delta_ref = hax.zeros(Batch)
    delta_pi = hax.named(jnp.array([0.4, 0.8], dtype=jnp.float32), Batch)

    _, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=1.0)

    assert isinstance(metrics["dpo_loss"], Metric)
    assert isinstance(metrics["dpo_margin_policy"], Metric)
    assert isinstance(metrics["dpo_margin_ref"], Metric)
    assert isinstance(metrics["dpo_accuracy"], Metric)


def test_logp_sum_passes_key_for_dropout():
    config = Gpt2Config(max_seq_len=8, hidden_dim=16, num_layers=1, num_heads=2, embed_pdrop=0.1)
    Vocab = hax.Axis("vocab", 32)
    model = config.build(Vocab, key=jrandom.PRNGKey(0))
    model = inference_mode(model, False)

    Pos = hax.Axis("position", 8)
    tokens = hax.named(jnp.arange(Pos.size, dtype=jnp.int32) % Vocab.size, Pos)
    example = LmExample.causal(tokens)

    out = _logp_sum(model, example, key=jrandom.PRNGKey(1))
    assert isinstance(out, hax.NamedArray)


def test_preference_chat_processor_outputs_masks(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {
            "chosen": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "No."},
            ],
        }
    ]

    result = processor(batch)
    assert len(result) == 1

    row = result[0]
    assert row["chosen_input_ids"].shape == row["chosen_assistant_masks"].shape
    assert row["rejected_input_ids"].shape == row["rejected_assistant_masks"].shape
    assert row["chosen_assistant_masks"].sum() > 0
    assert row["rejected_assistant_masks"].sum() > 0


def test_preference_pair_dataset_builds_example(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {
            "chosen": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Nope."},
            ],
        }
    ]

    processed = processor(batch)
    with tempfile.TemporaryDirectory() as tmpdir:
        with SerialCacheWriter(tmpdir, processor.output_exemplar) as writer:
            writer.write_batch(processed)

        cache = writer.result()
        Pos = hax.Axis("position", 128)
        dataset = PreferencePairDataset(cache, Pos, max_segments_per_example=1, slice_strategy="raise")
        example = dataset.as_sync_dataset()[0]

        assert isinstance(example, DpoExample)
        assert example.chosen.tokens.axes == (Pos,)
        assert example.rejected.tokens.axes == (Pos,)
        assert example.chosen.loss_weight.array.sum() > 0
        assert example.rejected.loss_weight.array.sum() > 0


def _preference_rows_for_slicing():
    return [
        _make_preference_row([1, 2], [6]),
        _make_preference_row([10, 11, 12, 13, 14], [20, 21, 22, 23, 24]),
        _make_preference_row([3, 4, 5], [7, 8, 9]),
    ]


def test_preference_pair_dataset_slice_strategy_raise():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _write_preference_cache(tmpdir, _preference_rows_for_slicing())
        Pos = hax.Axis("position", 4)
        with pytest.raises(ValueError, match="exceeds"):
            PreferencePairDataset(cache, Pos, max_segments_per_example=1, slice_strategy="raise")


def test_preference_pair_dataset_slice_strategy_drop():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _write_preference_cache(tmpdir, _preference_rows_for_slicing())
        Pos = hax.Axis("position", 4)
        dataset = PreferencePairDataset(cache, Pos, max_segments_per_example=1, slice_strategy="drop")
        examples = dataset.as_sync_dataset()

        assert len(examples) == 2
        first = np.array(examples[0].chosen.tokens.array).tolist()
        second = np.array(examples[1].chosen.tokens.array).tolist()
        assert first == [1, 2, 0, 0]
        assert second == [3, 4, 5, 0]


@pytest.mark.parametrize(
    ("slice_strategy", "expected_middle"),
    [
        ("left", [10, 11, 12, 13]),
        ("right", [11, 12, 13, 14]),
    ],
)
def test_preference_pair_dataset_slice_strategy_truncation(slice_strategy: str, expected_middle: list[int]):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _write_preference_cache(tmpdir, _preference_rows_for_slicing())
        Pos = hax.Axis("position", 4)
        dataset = PreferencePairDataset(cache, Pos, max_segments_per_example=1, slice_strategy=slice_strategy)
        examples = dataset.as_sync_dataset()

        assert len(examples) == 3
        first = np.array(examples[0].chosen.tokens.array).tolist()
        middle = np.array(examples[1].chosen.tokens.array).tolist()
        last = np.array(examples[2].chosen.tokens.array).tolist()

        assert first == [1, 2, 0, 0]
        assert middle == expected_middle
        assert last == [3, 4, 5, 0]


def test_apply_updates_handles_none_updates_for_namedarrays():
    Axis = hax.Axis("axis", 4)
    model = hax.ones(Axis)

    updated = apply_updates(model, None, None)

    assert hax.all(updated == model).scalar()


def test_partition_for_grad_overwrite_preserves_namedarrays():
    Axis = hax.Axis("axis", 4)
    model = hax.ones(Axis)

    overwrites, _ = partition_for_grad_overwrite(model)
    updated = apply_updates(model, None, overwrites)

    assert is_jax_array_like(updated.array)
    assert hax.all(updated == model).scalar()


def test_pspec_for_handles_filtered_namedarray():
    Layers = hax.Axis("layers", 2)
    Embed = hax.Axis("embed", 3)
    arr = hax.zeros((Layers, Embed))
    filtered = eqx.filter(arr, lambda _: False)

    pspec = pspec_for(filtered, resource_mapping={"layers": "data", "embed": "model"})

    assert pspec is None


class _DummyModelAveraging(ModelAveraging[DpoModel]):
    model: DpoModel

    def update(self, model, step):
        return self

    @property
    def model_params(self) -> DpoModel:
        return self.model


def test_eval_model_fills_missing_namedarrays_from_model():
    config = _tiny_gpt2_config()
    _, policy, reference = _build_policy_reference(config)
    model = DpoModel(policy=policy, reference=reference)
    trainable_filter = DpoModel(
        policy=_bool_tree_like(policy, True),
        reference=_bool_tree_like(reference, False),
    )

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(0),
        is_trainable=trainable_filter,
        mp=jmp.get_policy("f32"),
    )

    missing = eqx.filter(model, lambda _: False)
    state = dataclasses.replace(state, model_averaging=_DummyModelAveraging(model=missing))

    eval_model = state.eval_model
    weight = eval_model.policy.embeddings.token_embeddings.weight

    assert is_jax_array_like(weight.array)


def test_vmapped_init_with_sharding_handles_layer_axis():
    script = textwrap.dedent(
        """
        import numpy as np
        import jax
        import haliax as hax
        from haliax import Axis
        from haliax.partitioning import set_mesh
        from jax.sharding import Mesh

        Embed = Axis("embed", 32)
        Qkv = Axis("qkv", 3)
        Heads = Axis("heads", 4)
        HeadSize = Axis("head_size", 8)
        Layers = Axis("layers", 5)

        def init_fn(key):
            return hax.random.truncated_normal(key, (Embed, Qkv, Heads, HeadSize), -3, 3)

        devices = jax.devices()
        assert len(devices) == 4, f"Expected 4 devices, got {len(devices)}"
        mesh = Mesh(np.array(devices).reshape((4, 1)), ("data", "model"))

        with set_mesh(mesh), hax.axis_mapping({"embed": "data", "heads": "model"}):
            keys = jax.random.split(jax.random.PRNGKey(0), Layers.size)
            out = hax.vmap(init_fn, Layers)(keys)
            assert out.axes[0].name == "layers"
        """
    )
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(
        [
            str(repo_root / "lib" / "haliax" / "src"),
            str(repo_root / "lib" / "levanter" / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
