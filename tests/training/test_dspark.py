# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets.exceptions import DatasetGenerationError
from levanter.models.snowball import GRUG_MOE_ARCHITECTURE
from marin.datakit.chat_normalize import CHAT_SCHEMA
from marin.training.drafting import activation_store
from marin.training.drafting.dspark_data import (
    EMBEDDING,
    FINAL_NORM,
    GATE_DOWN,
    GATE_UP,
    GRUG_MASK_TOKEN,
    LM_HEAD,
    DraftDataConfig,
    GrugShape,
    inspect_teacher,
    prepare_data,
)
from marin.training.drafting.dspark_distillation import DraftRecipe, extraction_config, validate_draft
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from safetensors.numpy import save_file
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def recipe() -> DraftRecipe:
    return DraftRecipe(
        draft_layers=3,
        data=DraftDataConfig(shape=GrugShape(16, 4, 2, 1, 8, 8), mask_token_id=7, max_sequence_length=16),
        feature_taps=(1, 3),
        max_anchors=8,
        max_steps=1,
        trainer_ranks=1,
        teacher_replicas=1,
        cross_entropy_weight=0.1,
        total_variation_weight=0.9,
        optimizer="muon",
        lr_schedule="linear",
        draft_vocab_size=3,
        markov_rank=4,
    )


@pytest.fixture
def teacher(tmp_path: Path) -> Path:
    root = tmp_path / "teacher"
    root.mkdir()
    vocab = {
        word: index
        for index, word in enumerate(("[UNK]", "system", "prompt", "answer", "held", "done", "tag", GRUG_MASK_TOKEN))
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens([AddedToken(GRUG_MASK_TOKEN, special=True)])
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    hf.chat_template = (
        "{% for m in messages %}{% if m.role == 'assistant' %}"
        "{% generation %}{{ m.content }} {% endgeneration %}"
        "{% else %}{{ m.content }} {% endif %}{% endfor %}"
    )
    hf.save_pretrained(root)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "grug_moe",
                "architectures": [GRUG_MOE_ARCHITECTURE],
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 8,
                "max_position_embeddings": 32,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-5,
            }
        )
    )
    weights = {
        EMBEDDING: np.ones((8, 16), dtype=np.float32),
        LM_HEAD: np.arange(128, dtype=np.float32).reshape(8, 16) / 128,
        FINAL_NORM: np.ones((16,), dtype=np.float32),
        GATE_DOWN: np.full((2, 16), 0.25, dtype=np.float32),
        GATE_UP: np.full((16, 2), 0.5, dtype=np.float32),
        **{f"model.layers.{i}.input_layernorm.weight": np.ones((16,), dtype=np.float32) for i in range(4)},
    }
    save_file(weights, root / "model.safetensors")
    return root


def _mix(tmp_path: Path, *, overlapping: bool, harmony: bool) -> Path:
    def chat(prompt: str, answer: str) -> dict:
        messages = [("system", "system"), ("user", prompt), ("assistant", answer)]
        if not harmony:
            return {"messages": [{"role": role, "content": content} for role, content in messages]}
        return {
            "id": prompt,
            "messages": [
                {
                    "role": role,
                    "channel": "final" if role == "assistant" else None,
                    "content": [{"type": "text", "text": content}],
                }
                for role, content in messages
            ],
        }

    train = tmp_path / "train" / "rows.parquet"
    val = tmp_path / "validation" / "rows.parquet"
    train.parent.mkdir()
    val.parent.mkdir()
    schema = CHAT_SCHEMA if harmony else None
    pq.write_table(pa.Table.from_pylist([chat("prompt", "answer done")], schema=schema), train)
    rows = [chat("held", "tag"), chat("prompt", "answer done" if overlapping else "done")]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), val)
    return tmp_path


@pytest.mark.parametrize("overlapping", [False, True])
@pytest.mark.parametrize("harmony", [False, True], ids=["openai", "datakit"])
def test_prepare_data_masks_and_holds_out_rendered_rows(
    tmp_path: Path, teacher: Path, recipe: DraftRecipe, overlapping: bool, harmony: bool
) -> None:
    storage = StoragePath(f"memory://{tmp_path.name}")
    output = storage / "output"
    mix = _mix(tmp_path, overlapping=overlapping, harmony=harmony)
    for file in teacher.iterdir():
        directory = "tokenizer" if harmony and file.name not in {"config.json", "model.safetensors"} else "teacher"
        (storage / f"{directory}/{file.name}").upload_from(str(file))
    for name in ("train/rows.parquet", "validation/rows.parquet"):
        (storage / f"mix/{name}").upload_from(str(mix / name))
    kwargs = dict(
        output=str(output),
        teacher=str(storage / "teacher"),
        teacher_subdirectory="",
        mix=str(storage / "mix"),
        config=recipe.data,
        tokenizer_source=str(storage / "tokenizer") if harmony else None,
    )
    if overlapping:
        with pytest.raises(DatasetGenerationError):
            prepare_data(**kwargs)
        assert not (output / "prepared").exists()
        return

    prepare_data(**kwargs)
    prepared = tmp_path / "staged-data"
    (output / "prepared").download_to(str(prepared), recursive=True)
    rows = datasets.load_from_disk(str(prepared / "dataset"))
    split = json.loads((prepared / "splits.json").read_text())
    assert int(len(rows) * split["train_ratio"]) == 1
    assert torch.equal(rows[0]["input_ids"], torch.tensor([1, 2, 3, 5]))
    assert torch.equal(rows[0]["loss_mask"], torch.tensor([False, False, True, True]))
    assert rows[1]["input_ids"].tolist() == [1, 4, 6]
    assert json.loads((prepared / "token_frequencies.json").read_text()) == {"3": 1, "5": 1}


def test_teacher_preflight_rejects_an_incomplete_sharded_export(teacher: Path, recipe: DraftRecipe) -> None:
    (teacher / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {EMBEDDING: "missing.safetensors"}}))
    with pytest.raises(ValueError, match=r"missing\.safetensors"):
        inspect_teacher(
            StoragePath(str(teacher)),
            recipe.data,
            layers=recipe.draft_layers,
            intermediate_size=recipe.intermediate_size,
        )


def test_draft_config_preserves_feature_taps_and_vocab_mapping(
    tmp_path: Path, teacher: Path, recipe: DraftRecipe
) -> None:
    decoder = inspect_teacher(
        StoragePath(str(teacher)), recipe.data, layers=recipe.draft_layers, intermediate_size=recipe.intermediate_size
    )
    speculative, _ = extraction_config(recipe, tmp_path / "hidden_states")
    assert speculative["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 3, 4]
    ckpt = tmp_path / "draft"
    ckpt.mkdir()
    config = {
        "speculators_model_type": "dspark",
        "layer_type": "qwen3",
        "sample_from_anchor": True,
        "block_size": 8,
        "aux_hidden_state_layer_ids": [1, 3],
        "mask_token_id": 7,
        "draft_vocab_size": 3,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "transformer_layer_config": decoder,
        "speculators_config": {"proposal_methods": [{"speculative_tokens": 8}]},
    }
    (ckpt / "config.json").write_text(json.dumps(config))
    weights = {
        "fc.weight": np.zeros((16, 32), dtype=np.float32),
        LM_HEAD: np.zeros((3, 16), dtype=np.float32),
        "markov_head.markov_w1.weight": np.zeros((8, 4), dtype=np.float32),
        "markov_head.markov_w2.weight": np.zeros((3, 4), dtype=np.float32),
        "confidence_head.proj.weight": np.zeros((1, 20), dtype=np.float32),
        "d2t": np.array([0, 2, 5], dtype=np.int64),
    }
    weight_file = ckpt / "model.safetensors"
    save_file(weights, weight_file)
    validate_draft(ckpt, recipe, np.array([0, 2, 5]))
    save_file({**weights, "d2t": np.array([0, 3, 5])}, weight_file)
    with pytest.raises(ValueError, match="vocabulary mapping"):
        validate_draft(ckpt, recipe, np.array([0, 2, 5]))


def test_activation_cache_reuses_completed_rows_until_the_reuse_window_ends(tmp_path, monkeypatch):
    now = 0.0
    monkeypatch.setattr(activation_store, "time", lambda: now)
    root = StoragePath(str(tmp_path / "cached"))
    reusable_for = Duration.from_hours(48)

    def populate(path):
        activation_store.activation_path(path, 0).write_bytes(b"activation")

    cache = activation_store.get_or_build_activation_cache(root, reusable_for=reusable_for, populate=populate)
    assert activation_store.read_activation_cache(root, reusable_for=reusable_for) == cache
    reused = activation_store.get_or_build_activation_cache(
        root, reusable_for=reusable_for, populate=lambda _: pytest.fail("recomputed existing activations")
    )
    assert reused == cache
    assert activation_store.activation_path(reused, 0).read_bytes() == b"activation"

    now += reusable_for.to_seconds()
    assert activation_store.read_activation_cache(root, reusable_for=reusable_for) is None
    replacement = activation_store.get_or_build_activation_cache(root, reusable_for=reusable_for, populate=populate)
    assert replacement != cache
    assert activation_store.activation_path(replacement, 0).read_bytes() == b"activation"
    assert activation_store.activation_path(cache, 0).read_bytes() == b"activation"


def test_activation_cache_retries_without_exposing_partial_rows(tmp_path):
    root = StoragePath(str(tmp_path / "cached"))
    reusable_for = Duration.from_hours(48)

    def fail(path):
        activation_store.activation_path(path, 0).write_bytes(b"partial")
        raise RuntimeError("teacher stopped")

    with pytest.raises(RuntimeError):
        activation_store.get_or_build_activation_cache(root, reusable_for=reusable_for, populate=fail)
    assert activation_store.read_activation_cache(root, reusable_for=reusable_for) is None

    def populate(path):
        activation_store.activation_path(path, 0).write_bytes(b"complete")
        activation_store.activation_path(path, 1).write_bytes(b"complete")

    cache = activation_store.get_or_build_activation_cache(root, reusable_for=reusable_for, populate=populate)
    assert [activation_store.activation_path(cache, row).read_bytes() for row in range(2)] == [b"complete"] * 2
