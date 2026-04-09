# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import tempfile
from pathlib import Path
from types import SimpleNamespace

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import pytest
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, set_mesh
from haliax.quantization import apply_updates, partition_for_grad_overwrite
from haliax.jax_utils import is_jax_array_like
from jax.sharding import Mesh

from levanter.data.loader import DataLoader
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text import (
    DatasetComponent,
    DpoExample,
    PreferenceChatLmDatasetFormat,
    PreferenceChatProcessor,
    PreferenceLmDataConfig,
    PreferencePairDataset,
    TextLmDatasetFormat,
)
from levanter.adaptation import LoraAdaptationConfig, NoAdaptationConfig
from levanter.dpo import (
    CachedDpoExample,
    DpoModel,
    ReferenceEvalCacheConfig,
    ValidationDatasetSpec,
    _logp_sum,
    build_or_load_reference_eval_cache,
    dpo_loss,
    dpo_loss_from_logps,
    load_reference_eval_cache,
    reference_eval_cache_metadata,
    reference_eval_cache_path,
)
from levanter.main.lora_dpo import LoraDpoConfig, _translate_legacy_lora_dpo_config
from levanter.main.train_dpo import (
    AdapterBaseReferenceConfig,
    SeparateReferenceConfig,
    TrainDpoConfig,
    _build_dpo_dataset,
    _derive_training_keys,
    _periodic_eval_callback,
    _restore_policy_model_from_partial_checkpoint,
    _scheduled_eval_callback,
    _validate_dpo_config,
)
from levanter.metrics import Metric
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.lm_model import LmExample
from levanter.optim import AdamConfig
from levanter.optim.model_averaging import ModelAveraging
from levanter.store.cache import SerialCacheWriter
from levanter.tokenizers import MarinTokenizer, load_tokenizer as load_marin_tokenizer
from levanter.trainer_state import TrainerState, saveable_training_mask, trainables_only
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.tree_utils import inference_mode


MODEL_NAME = "stanford-crfm/marin-tokenizer"


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    try:
        tok = load_marin_tokenizer(MODEL_NAME)
        return Path(tok.name_or_path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {exc}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def _load_tokenizer(tokenizer_path: Path) -> MarinTokenizer:
    return load_marin_tokenizer(str(tokenizer_path))


def _tiny_gpt2_config() -> Gpt2Config:
    return Gpt2Config(
        max_seq_len=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )


def _build_policy_model(config: Gpt2Config):
    Vocab = hax.Axis("vocab", 32)
    return config.build(Vocab, key=jrandom.PRNGKey(0))


def _make_dpo_example(*, seq_len: int = 8, vocab_size: int = 32, key=jrandom.PRNGKey(0)) -> DpoExample:
    Pos = hax.Axis("position", seq_len)
    chosen_key, rejected_key = jrandom.split(key)
    chosen = LmExample.causal(hax.random.randint(chosen_key, Pos, 0, vocab_size))
    rejected = LmExample.causal(hax.random.randint(rejected_key, Pos, 0, vocab_size))
    return DpoExample(chosen=chosen, rejected=rejected)


def test_trainer_state_init_with_policy_model():
    config = _tiny_gpt2_config()
    model = _build_policy_model(config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(0),
        is_trainable=True,
        mp=jmp.get_policy("f32"),
    )

    assert isinstance(state.model, Gpt2LMHeadModel)


def test_separate_reference_state_marks_reference_as_non_saveable():
    config = _tiny_gpt2_config()
    policy_model = _build_policy_model(config)
    reference_model = _build_policy_model(config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        DpoModel(policy=policy_model, reference=reference_model),
        key=jrandom.PRNGKey(0),
        is_trainable=DpoModel(policy=True, reference=False),
        mp=jmp.get_policy("f32"),
    )

    assert isinstance(state.model, DpoModel)
    assert isinstance(state.model.reference, Gpt2LMHeadModel)
    mask = saveable_training_mask(state, DpoModel(policy=True, reference=False))
    assert mask.model.reference is False


def test_adapter_base_reference_requires_non_none_adapter():
    config = TrainDpoConfig(
        model=_tiny_gpt2_config(),
        reference=AdapterBaseReferenceConfig(),
        adapter=NoAdaptationConfig(),
    )

    with pytest.raises(ValueError, match="requires a non-none adapter"):
        _validate_dpo_config(config)


def test_lora_adapter_base_reference_requires_zero_init_b():
    config = TrainDpoConfig(
        model=_tiny_gpt2_config(),
        reference=AdapterBaseReferenceConfig(),
        adapter=LoraAdaptationConfig(zero_init_b=False),
    )

    with pytest.raises(ValueError, match="requires zero_init_b=true"):
        _validate_dpo_config(config)


def test_training_keys_preserve_legacy_full_dpo_model_key_path():
    seed = 123
    legacy_data_key, _legacy_loader_key, legacy_model_key, legacy_training_key = jrandom.split(
        jrandom.PRNGKey(seed), 4
    )
    legacy_policy_key, _legacy_reference_key = jrandom.split(legacy_model_key)

    data_key, model_key, policy_key, _adapter_key, training_key = _derive_training_keys(seed)

    assert np.array_equal(np.asarray(data_key), np.asarray(legacy_data_key))
    assert np.array_equal(np.asarray(model_key), np.asarray(legacy_model_key))
    assert np.array_equal(np.asarray(policy_key), np.asarray(legacy_policy_key))
    assert np.array_equal(np.asarray(training_key), np.asarray(legacy_training_key))


def test_periodic_eval_callback_dedupes_same_step():
    calls: list[int] = []
    callback = _periodic_eval_callback(lambda info: calls.append(info.step))

    callback(SimpleNamespace(step=0), force=True)
    callback(SimpleNamespace(step=0))
    callback(SimpleNamespace(step=1))
    callback(SimpleNamespace(step=1), force=True)

    assert calls == [0, 1]


def test_scheduled_eval_callback_runs_only_on_scheduled_steps():
    calls: list[int] = []
    callback = _scheduled_eval_callback(lambda info: calls.append(info.step), {5, 10})

    callback(SimpleNamespace(step=0), force=True)
    callback(SimpleNamespace(step=4))
    callback(SimpleNamespace(step=5))
    callback(SimpleNamespace(step=5), force=True)
    callback(SimpleNamespace(step=10))

    assert calls == [0, 5, 10]


def test_canonical_dpo_config_parses_from_yaml(tmp_path: Path):
    config_path = tmp_path / "canonical_dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: gpt2",
                "adapter:",
                "  type: none",
                "reference:",
                "  type: separate",
                "  model_path: meta-llama/Llama-3.1-8B-Instruct",
                "  is_hf: true",
                "reference_eval_cache:",
                "  mode: build_or_load",
                "  cache_dir: gs://example-bucket/reference-eval",
            ]
        )
    )

    config = draccus.parse(TrainDpoConfig, str(config_path), args=[])

    assert isinstance(config.adapter, NoAdaptationConfig)
    assert isinstance(config.reference, SeparateReferenceConfig)
    assert config.reference_eval_cache == ReferenceEvalCacheConfig(
        mode="build_or_load",
        cache_dir="gs://example-bucket/reference-eval",
    )


def test_canonical_lora_dpo_config_parses_from_yaml(tmp_path: Path):
    config_path = tmp_path / "canonical_lora_dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: gpt2",
                "adapter:",
                "  type: lora",
                "  r: 8",
                "  alpha: 8.0",
                "  zero_init_b: true",
                "reference:",
                "  type: adapter_base",
            ]
        )
    )

    config = draccus.parse(TrainDpoConfig, str(config_path), args=[])

    assert isinstance(config.adapter, LoraAdaptationConfig)
    assert isinstance(config.reference, AdapterBaseReferenceConfig)


def test_legacy_lora_dpo_config_translates_to_canonical(tmp_path: Path):
    config_path = tmp_path / "legacy_lora_dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "initialize_from_hf: meta-llama/Llama-3.1-8B-Instruct",
                "lora:",
                "  r: 8",
                "  alpha: 8.0",
                "  zero_init_b: true",
                "hf_save_steps: 100",
            ]
        )
    )

    legacy_config = draccus.parse(LoraDpoConfig, str(config_path), args=[])

    translated = _translate_legacy_lora_dpo_config(legacy_config)

    assert isinstance(translated.adapter, LoraAdaptationConfig)
    assert isinstance(translated.reference, AdapterBaseReferenceConfig)


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


def test_preference_lm_data_config_rejects_non_preference_format():
    with pytest.raises(ValueError, match="requires preference_chat"):
        PreferenceLmDataConfig(components={"train": DatasetComponent(format=TextLmDatasetFormat())})


def test_preference_lm_data_config_rejects_packed_preference_format():
    with pytest.raises(ValueError, match="Packed preference_chat datasets are not supported"):
        PreferenceLmDataConfig(
            components={
                "train": DatasetComponent(format=PreferenceChatLmDatasetFormat(pack=True)),
            }
        )


def test_preference_lm_data_config_rejects_non_raise_slice_strategy():
    with pytest.raises(ValueError, match="slice_strategy must be 'raise'"):
        PreferenceLmDataConfig(
            components={
                "train": DatasetComponent(format=PreferenceChatLmDatasetFormat(slice_strategy="left")),
            }
        )


def test_build_dpo_dataset_with_strict_preference_config(monkeypatch):
    config = PreferenceLmDataConfig(
        components={
            "pref": DatasetComponent(format=PreferenceChatLmDatasetFormat()),
        },
        shuffle=False,
    )

    monkeypatch.setattr(
        PreferenceLmDataConfig,
        "build_caches",
        lambda self, split: {"pref": object()},
    )

    monkeypatch.setattr(
        "levanter.main.train_dpo.dataset_for_preference_format",
        lambda format, Pos, cache: ListAsyncDataset([0]),
    )

    dataset = _build_dpo_dataset(config, hax.Axis("position", 8), key=jrandom.PRNGKey(0))
    assert isinstance(dataset, MixtureDataset)
    assert set(dataset.datasets.keys()) == {"pref"}


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


def test_dpo_loss_matches_cached_reference_values():
    config = _tiny_gpt2_config()
    policy_model = _build_policy_model(config)

    Vocab = hax.Axis("vocab", 32)
    reference_model = config.build(Vocab, key=jrandom.PRNGKey(1))
    reference_model = inference_mode(reference_model, True)

    example = _make_dpo_example(key=jrandom.PRNGKey(2))

    logp_ref_chosen = _logp_sum(reference_model, example.chosen, key=None)
    logp_ref_rejected = _logp_sum(reference_model, example.rejected, key=None)
    cached_example = CachedDpoExample(
        chosen=example.chosen,
        rejected=example.rejected,
        logp_ref_chosen=float(np.asarray(logp_ref_chosen.array)),
        logp_ref_rejected=float(np.asarray(logp_ref_rejected.array)),
    )

    uncached_loss, uncached_metrics = dpo_loss(policy_model, reference_model, example, beta=0.1)
    cached_loss, cached_metrics = dpo_loss(policy_model, None, cached_example, beta=0.1)

    assert np.isclose(float(uncached_loss), float(cached_loss))
    assert np.isclose(float(uncached_metrics["dpo_loss"]), float(cached_metrics["dpo_loss"]))
    assert np.isclose(float(uncached_metrics["dpo_accuracy"]), float(cached_metrics["dpo_accuracy"]))


def test_reference_eval_cache_path_changes_with_slice_bounds():
    dataset = ListAsyncDataset([_make_dpo_example()])
    base_spec = ValidationDatasetSpec(
        name="val",
        dataset=dataset,
        source_cache_path="gs://bucket/tokenized/run/validation",
        source_split="validation",
    )
    split_spec = dataclasses.replace(base_spec, source_split="train", slice_start=10, slice_end=20)
    reference_identity = {"reference_type": "separate", "reference": {"model_path": "hf/model"}}

    base_path = reference_eval_cache_path(base_spec, reference_identity=reference_identity, seq_len=32, cache_dir=None)
    split_path = reference_eval_cache_path(
        split_spec, reference_identity=reference_identity, seq_len=32, cache_dir=None
    )

    assert base_path != split_path


def test_load_reference_eval_cache_rejects_metadata_mismatch(tmp_path: Path):
    cache_dir = tmp_path / "reference_eval_cache"
    spec = ValidationDatasetSpec(
        name="val",
        dataset=ListAsyncDataset([_make_dpo_example()]),
        source_cache_path=str(tmp_path / "validation"),
        source_split="validation",
    )
    metadata = reference_eval_cache_metadata(
        spec,
        reference_identity={"reference_type": "separate", "reference": {"model_path": "hf/model-a"}},
        seq_len=8,
    )

    with SerialCacheWriter(
        str(cache_dir),
        {"logp_ref_chosen": np.zeros((), dtype=np.float32), "logp_ref_rejected": np.zeros((), dtype=np.float32)},
        metadata=metadata,
    ) as writer:
        writer.write_batch(
            {
                "logp_ref_chosen": np.array([1.0], dtype=np.float32),
                "logp_ref_rejected": np.array([2.0], dtype=np.float32),
            }
        )

    mismatched_metadata = reference_eval_cache_metadata(
        spec,
        reference_identity={"reference_type": "separate", "reference": {"model_path": "hf/model-b"}},
        seq_len=8,
    )

    with pytest.raises(FileNotFoundError, match="metadata mismatch"):
        load_reference_eval_cache(str(cache_dir), metadata=mismatched_metadata)


def test_build_or_load_reference_eval_cache_matches_direct_logps(tmp_path: Path):
    config = _tiny_gpt2_config()
    reference_model = inference_mode(_build_policy_model(config), True)

    examples = [_make_dpo_example(key=jrandom.PRNGKey(i)) for i in range(3)]
    dataset = ListAsyncDataset(examples)
    spec = ValidationDatasetSpec(
        name="val",
        dataset=dataset,
        source_cache_path=str(tmp_path / "validation"),
        source_split="validation",
    )
    reference_identity = {"reference_type": "separate", "reference": {"model_path": "hf/model"}}
    cache_dir = reference_eval_cache_path(
        spec, reference_identity=reference_identity, seq_len=8, cache_dir=str(tmp_path / "cache")
    )
    metadata = reference_eval_cache_metadata(spec, reference_identity=reference_identity, seq_len=8)

    with local_cpu_mesh() as mesh, hax.axis_mapping({"batch": ResourceAxis.DATA}):
        loader = DataLoader(
            dataset,
            batch_size=1,
            mesh=mesh,
            axis_resources=None,
            max_buffered_batches=1,
            prefetch_size=1,
            allow_nondivisible_batch_size=True,
        )
        chosen, rejected = build_or_load_reference_eval_cache(
            reference_model=reference_model,
            dataset=dataset,
            eval_loader=loader,
            compute_axis_mapping=None,
            mp=jmp.get_policy("f32"),
            cache_dir=cache_dir,
            metadata=metadata,
        )

    expected_chosen = np.asarray(
        [float(np.asarray(_logp_sum(reference_model, ex.chosen, key=None).array)) for ex in examples], dtype=np.float32
    )
    expected_rejected = np.asarray(
        [float(np.asarray(_logp_sum(reference_model, ex.rejected, key=None).array)) for ex in examples],
        dtype=np.float32,
    )

    assert np.allclose(chosen, expected_chosen)
    assert np.allclose(rejected, expected_rejected)

    loaded_chosen, loaded_rejected = load_reference_eval_cache(cache_dir, metadata=metadata)
    assert np.allclose(loaded_chosen, expected_chosen)
    assert np.allclose(loaded_rejected, expected_rejected)


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


class _DummyModelAveraging(ModelAveraging[Gpt2LMHeadModel]):
    model: Gpt2LMHeadModel

    def update(self, model, step):
        return self

    @property
    def model_params(self) -> Gpt2LMHeadModel:
        return self.model


def test_eval_model_fills_missing_namedarrays_from_model():
    config = _tiny_gpt2_config()
    model = _build_policy_model(config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(0),
        is_trainable=True,
        mp=jmp.get_policy("f32"),
    )

    missing = eqx.filter(model, lambda _: False)
    state = dataclasses.replace(state, model_averaging=_DummyModelAveraging(model=missing))

    eval_model = state.eval_model
    weight = eval_model.embeddings.token_embeddings.weight

    assert is_jax_array_like(weight.array)


def test_restore_policy_model_from_partial_checkpoint_recovers_base_model():
    config = _tiny_gpt2_config()
    Vocab = Axis("vocab", 32)
    base_key, wrong_base_key, adapter_key, wrong_adapter_key, example_key = jrandom.split(jrandom.PRNGKey(0), 5)

    adapter = LoraAdaptationConfig(r=4, zero_init_b=False)
    trained_policy = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    wrong_resume_skeleton = adapter.apply(config.build(Vocab, key=wrong_base_key), key=wrong_adapter_key)
    correct_source_skeleton = adapter.apply(config.build(Vocab, key=base_key), key=wrong_adapter_key)
    trainable_filter = adapter.trainable_filter(trained_policy)

    checkpointed_trainables = trainables_only(trained_policy, trainable_filter)
    wrong_resumed_policy = eqx.combine(checkpointed_trainables, wrong_resume_skeleton)
    restored_policy = _restore_policy_model_from_partial_checkpoint(
        wrong_resumed_policy,
        correct_source_skeleton,
        trainable_filter,
    )

    example = _make_dpo_example(seq_len=8, vocab_size=Vocab.size, key=example_key)
    trained_logp = _logp_sum(inference_mode(trained_policy, True), example.chosen).scalar()
    wrong_logp = _logp_sum(inference_mode(wrong_resumed_policy, True), example.chosen).scalar()
    restored_logp = _logp_sum(inference_mode(restored_policy, True), example.chosen).scalar()

    assert wrong_logp != pytest.approx(trained_logp)
    assert restored_logp == pytest.approx(trained_logp)


def test_vmapped_init_with_sharding_handles_layer_axis():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip(f"requires at least 4 JAX devices; found {len(devices)}")

    Embed = Axis("embed", 32)
    Qkv = Axis("qkv", 3)
    Heads = Axis("heads", 4)
    HeadSize = Axis("head_size", 8)
    Layers = Axis("layers", 5)

    def init_fn(key):
        return hax.random.truncated_normal(key, (Embed, Qkv, Heads, HeadSize), -3, 3)

    mesh = Mesh(np.array(devices[:4]).reshape((4, 1)), ("data", "model"))
    with set_mesh(mesh), hax.axis_mapping({"embed": "data", "heads": "model"}):
        keys = jax.random.split(jax.random.PRNGKey(0), Layers.size)
        out = hax.vmap(init_fn, Layers)(keys)
        assert out.axes[0].name == "layers"
