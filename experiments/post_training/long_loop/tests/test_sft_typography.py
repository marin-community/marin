# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typography store transform (mix V8)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheLedger, CacheMetadata, write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from experiments.post_training.long_loop.sft_typography import (
    ASSISTANT_HEADER,
    NORMALIZED_MARKER,
    UNCHANGED_MARKER,
    CensusMaskRule,
    Rewriter,
    ShardTask,
    SourceLossPin,
    StoreSummary,
    TeacherMaskRule,
    _sft_stores,
    assemble_catalog,
    assistant_spans,
    normalize_catalog_part,
    normalize_shard,
    normalize_store,
    normalize_text,
    supervision_of,
)
from experiments.post_training.long_loop.source_loss import SourceLossMetadata, SourceLossMode


def _shard_task(source, output, tokenizer_path, header_ids, teacher_only):
    return ShardTask(
        source_shard=source,
        output_shard=output,
        tokenizer_path=tokenizer_path,
        header_ids=header_ids,
        mask_rule=TeacherMaskRule() if teacher_only else CensusMaskRule(),
    )


VOCAB_SIZE = 128000  # leaves room for special tokens at ids >= 128000


def _test_tokenizer() -> Tokenizer:
    """A byte-level BPE tokenizer with the real special-id layout and tiny merges.

    Single-byte tokens cover every byte, so text encodes character by character;
    a few merges give the whole-target and split-target token shapes the rewriter
    must handle. Special tokens land at ids >= 128000, as in the release tokenizer.
    """
    vocab = {ch: i for i, ch in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    merges = [("â", "Ģ"), ("âĢ", "¯"), ("âĢ", "ģ"), ("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e")]
    next_id = len(vocab)
    for token in ("âĢ", "âĢ¯", "âĢģ", "Ġt", "Ġth", "Ġthe"):
        vocab[token] = next_id
        next_id += 1
    while next_id < VOCAB_SIZE:
        vocab[f"unused{next_id:06d}"] = next_id
        next_id += 1
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    return tokenizer


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_text(_test_tokenizer().to_str())
    return str(path)


@pytest.fixture(scope="module")
def header_ids(tokenizer_path) -> np.ndarray:
    tokenizer = Tokenizer.from_str(Path(tokenizer_path).read_text())
    return np.asarray(tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False).ids)


def _tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_str(Path(tokenizer_path).read_text())


def _enc(tokenizer: Tokenizer, text: str) -> np.ndarray:
    return np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32)


def _assistant_row(tokenizer: Tokenizer, header_ids: np.ndarray, content: str, trailing: str = "") -> np.ndarray:
    """Tokens for one canonical assistant turn: header, newline, content, optional trailing text."""
    start, end = int(header_ids[0]), int(header_ids[-2])
    pieces = [np.asarray([start]), _enc(tokenizer, "assistant"), np.asarray([end]), _enc(tokenizer, "\n")]
    if content:
        pieces.append(_enc(tokenizer, content))
    if trailing:
        pieces.append(_enc(tokenizer, trailing))
    return np.concatenate(pieces)


def _weighted_row(tokenizer: Tokenizer, header_ids: np.ndarray, content: str, trailing: str = "") -> dict:
    """A canonical assistant turn with census-conforming binary weights."""
    row = _assistant_row(tokenizer, header_ids, content, trailing)
    weights = np.zeros(len(row), dtype=np.float32)
    _, expected = assistant_spans(row, header_ids)
    weights[expected] = 1.0
    return {"input_ids": row, "loss_weights": weights}


def _write_store(root: Path, name: str, rows: list[dict]) -> CacheLedger:
    """Write one sharded store under root/name and return its component ledger."""
    store = str(root / name)
    write_levanter_cache(iter(rows), store, metadata={}, batch_size=2, max_pending_batches=4)
    return _component_ledger(name, rows)


def _component_ledger(name: str, rows: list[dict]) -> CacheLedger:
    tokens = sum(len(r["input_ids"]) for r in rows)
    return CacheLedger(
        total_num_rows=len(rows),
        shard_rows={name: len(rows)},
        is_finished=True,
        finished_shards=[name],
        field_counts={"input_ids": tokens, "loss_weights": tokens},
        field_counts_by_shard={name: {"input_ids": tokens, "loss_weights": tokens}},
        layout=CACHE_LAYOUT_SHARDED,
        metadata=CacheMetadata(preprocessor_metadata={"loss_mode": "chat_assistant_only", "source": name}),
    )


def _read_store(store: str) -> list[dict]:
    ledger = CacheLedger.from_json((Path(store) / "shard_ledger.json").read_text())
    records = []
    for shard in ledger.finished_shards:
        tokens = JaggedArrayStore.open(str(Path(store) / shard / "input_ids"), mode="r", dtype=np.int32)
        weights = JaggedArrayStore.open(str(Path(store) / shard / "loss_weights"), mode="r", dtype=np.float32)
        assert len(tokens) == len(weights)
        indices = list(range(len(tokens)))
        records.extend(
            {"input_ids": np.asarray(t), "loss_weights": np.asarray(w)}
            for t, w in zip(tokens.get_batch_sync(indices), weights.get_batch_sync(indices), strict=True)
        )
    return records


def test_rewriter_replaces_and_verifies_a_whole_target_token(tokenizer_path, header_ids):
    rewriter = Rewriter(_tokenizer(tokenizer_path))
    record = _weighted_row(_tokenizer(tokenizer_path), header_ids, " the\u202f the", trailing="<|eot_id|>")
    row, weights = record["input_ids"], record["loss_weights"]
    CensusMaskRule().verify_row(row, weights, header_ids)

    found = rewriter.occurrences(row)
    assert len(found) == 1
    spans, _ = assistant_spans(row, header_ids)
    new_tokens, new_weights = rewriter.rewrite(row, weights, found)
    assert rewriter.decode(new_tokens) == normalize_text(rewriter.decode(row))
    assert rewriter.occurrences(new_tokens) == []
    assert new_tokens[-1] == row[-1]  # the eot special is kept
    CensusMaskRule().verify_rewritten(
        new_tokens, new_weights, spans, [float(weights[left]) for left, _ in spans], header_ids
    )
    assert new_weights.sum() == weights.sum()  # the repaired space stays supervised


def test_rewriter_handles_a_target_split_across_tokens(tokenizer_path, header_ids):
    tokenizer = _tokenizer(tokenizer_path)
    rewriter = Rewriter(tokenizer)
    # U+202F is E2 80 AF; force the byte tokens apart so the target spans three tokens.
    split_target = [
        tokenizer.token_to_id(t) for t in ("Ġthe", "â", "Ģ", "¯", "Ġthe")  # the whole-target token, forced apart
    ]
    row = np.concatenate(
        [
            _assistant_row(tokenizer, header_ids, ""),
            np.asarray(split_target, dtype=np.int32),
            np.asarray([tokenizer.token_to_id("<|eot_id|>")]),
        ]
    )
    weights = np.zeros(len(row), dtype=np.float32)
    _, expected = assistant_spans(row, header_ids)
    weights[expected] = 1.0
    found = rewriter.occurrences(row)
    assert len(found) == 1
    new_tokens, new_weights = rewriter.rewrite(row, weights, found)
    assert rewriter.decode(new_tokens) == normalize_text(rewriter.decode(row))
    assert rewriter.occurrences(new_tokens) == []
    assert len(new_tokens) < len(row)  # fresh tokenization merges the split bytes
    # the supervised region follows the fresh tokenization (verified in full above)
    assert new_weights.sum() == assistant_spans(new_tokens, header_ids)[1].sum()


def test_rewriter_keeps_user_turn_weights_at_zero(tokenizer_path, header_ids):
    tokenizer = _tokenizer(tokenizer_path)
    rewriter = Rewriter(tokenizer)
    start, end = int(header_ids[0]), int(header_ids[-2])
    eot = tokenizer.token_to_id("<|eot_id|>")
    user_header = np.concatenate(
        [np.asarray([start]), _enc(tokenizer, "user"), np.asarray([end]), _enc(tokenizer, "\n")]
    )
    assistant = _weighted_row(tokenizer, header_ids, " ok", trailing="<|eot_id|>")
    row = np.concatenate([assistant["input_ids"], user_header, _enc(tokenizer, "a\u2011b"), np.asarray([eot])])
    weights = np.zeros(len(row), dtype=np.float32)
    spans, expected = assistant_spans(row, header_ids)
    assert len(spans) == 1  # only the assistant header opens a span
    weights[expected] = 1.0

    found = rewriter.occurrences(row)
    assert len(found) == 1
    new_tokens, new_weights = rewriter.rewrite(row, weights, found)
    assert rewriter.decode(new_tokens) == normalize_text(rewriter.decode(row))
    CensusMaskRule().verify_rewritten(
        new_tokens, new_weights, spans, [float(weights[left]) for left, _ in spans], header_ids
    )
    assert new_weights.sum() == weights.sum()  # the rewritten user-turn token stays unsupervised


def test_verify_rejects_broken_masks(tokenizer_path, header_ids):
    tokenizer = _tokenizer(tokenizer_path)
    record = _weighted_row(tokenizer, header_ids, " ok", trailing="<|eot_id|>")
    row, weights = record["input_ids"], record["loss_weights"]
    _, expected = assistant_spans(row, header_ids)

    bad = weights.copy()
    bad[int(np.flatnonzero(expected)[0])] = 0.0
    rule = CensusMaskRule()
    with pytest.raises(ValueError, match="census assistant-span rule"):
        rule.verify_row(row, bad, header_ids)
    with pytest.raises(ValueError, match="nonbinary"):
        rule.verify_row(row, weights + 2.0, header_ids)
    with pytest.raises(ValueError, match="length differs"):
        rule.verify_row(row, weights[:-1], header_ids)


def test_teacher_only_spans_keep_their_value(tokenizer_path, header_ids):
    tokenizer = _tokenizer(tokenizer_path)
    rewriter = Rewriter(tokenizer)
    record = _weighted_row(tokenizer, header_ids, " the\u202f the", trailing="<|eot_id|>")
    row, weighted = record["input_ids"], record["loss_weights"]
    spans, _ = assistant_spans(row, header_ids)
    found = rewriter.occurrences(row)

    # an unweighted teacher span: binary zero everywhere
    rule = TeacherMaskRule()
    zero = np.zeros(len(row), dtype=np.float32)
    rule.verify_row(row, zero, header_ids)
    new_tokens, new_weights = rewriter.rewrite(row, zero, found)
    rule.verify_rewritten(new_tokens, new_weights, spans, [0.0], header_ids)
    assert new_weights.sum() == 0.0

    # a weighted teacher span keeps its value through the rewrite
    rule.verify_row(row, weighted, header_ids)
    tokens2, weights2 = rewriter.rewrite(row, weighted, found)
    rule.verify_rewritten(tokens2, weights2, spans, [1.0], header_ids)

    # loss outside candidate spans is rejected
    stray = weighted.copy()
    stray[0] = 1.0
    with pytest.raises(ValueError, match="outside assistant spans"):
        rule.verify_row(row, stray, header_ids)


def test_normalize_shard_rewrites_a_store(tmp_path, tokenizer_path, header_ids):
    tokenizer = _tokenizer(tokenizer_path)
    rewriter = Rewriter(tokenizer)
    rows = [
        _weighted_row(tokenizer, header_ids, " the\u202f the", trailing="<|eot_id|>"),
        _weighted_row(tokenizer, header_ids, " plain", trailing="<|eot_id|>"),
        _weighted_row(tokenizer, header_ids, "a\u2011b\u202fc", trailing="<|eot_id|>"),
    ]
    source = str(tmp_path / "source")
    write_levanter_cache(iter(rows), source, metadata={}, batch_size=2, max_pending_batches=4)

    output = str(tmp_path / "output")
    stats = normalize_shard(_shard_task(source, output, tokenizer_path, header_ids, teacher_only=False))
    assert stats.rows == 3
    assert stats.changed_rows == 2
    assert stats.count_nnbsp == 2 and stats.count_nbhyphen == 1
    assert stats.token_delta == stats.out_tokens - stats.tokens

    written = _read_store(output)
    assert len(written) == 3
    for original, out in zip(rows, written, strict=True):
        assert len(out["loss_weights"]) == len(out["input_ids"])
        if rewriter.occurrences(original["input_ids"]):
            assert not rewriter.occurrences(out["input_ids"])
            assert rewriter.decode(out["input_ids"]) == normalize_text(rewriter.decode(original["input_ids"]))
        else:
            assert np.array_equal(out["input_ids"], original["input_ids"])
            assert np.array_equal(out["loss_weights"], original["loss_weights"])

    summary = StoreSummary(source, output, [stats]).to_json()
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(stats.input_tokens_sha256))
    assert summary["store_input_tokens_sha256"] == digest.hexdigest()
    assert summary["store_output_tokens_sha256"] != summary["store_input_tokens_sha256"]


def test_normalize_store_writes_marker_and_ledger(tmp_path, tokenizer_path, header_ids):
    tokenizer = _tokenizer(tokenizer_path)
    record = _weighted_row(tokenizer, header_ids, " the\u202f the", trailing="<|eot_id|>")
    source = str(tmp_path / "source")
    # a self-ledgered store: the shard lives under the store dir, with a
    # component-level sharded ledger beside it, as the release stores do
    write_levanter_cache(
        iter([record]), str(tmp_path / "source" / "shard-0"), metadata={}, batch_size=2, max_pending_batches=4
    )
    _component_ledger("shard-0", [record])._serialize_and_commit(source)

    output = str(tmp_path / "output")
    summary = normalize_store(source, output, tokenizer_path, mask_rule=CensusMaskRule(), workers=1)
    assert summary["changed_rows"] == 1
    marker = json.loads((Path(output) / NORMALIZED_MARKER).read_text())
    assert marker["source"] == source
    assert marker["transform"] == {"name": "sft-typography-normalize", "version": "v1"}
    ledger = CacheLedger.from_json((Path(output) / "shard_ledger.json").read_text())
    assert ledger.total_num_rows == 1
    assert ledger.field_counts["input_ids"] == ledger.field_counts["loss_weights"] == summary["out_tokens"]

    count_only = normalize_store(source, None, tokenizer_path, mask_rule=CensusMaskRule(), workers=1)
    assert count_only["changed_rows"] == 1
    assert count_only["store_output_tokens_sha256"] == count_only["store_input_tokens_sha256"]


def _catalog(tmp_path: Path, tokenizer: Tokenizer, header_ids: np.ndarray) -> tuple[str, str, dict]:
    """A catalog with one target-bearing SFT store, one clean SFT store, one replay store."""
    root = tmp_path / "cache_root"
    root.mkdir(parents=True)
    rows_with = [
        _weighted_row(tokenizer, header_ids, " the\u202f the", trailing="<|eot_id|>"),
        _weighted_row(tokenizer, header_ids, "a\u2011b", trailing="<|eot_id|>"),
    ]
    rows_clean = [_weighted_row(tokenizer, header_ids, " plain", trailing="<|eot_id|>")]
    rows_replay = [{"input_ids": _enc(tokenizer, " replay\u202ftext")}]

    ledgers = {
        "sft/source/with-targets": _write_store(root, "sft-a", rows_with),
        "sft/source/clean": _write_store(root, "sft-b", rows_clean),
        "pretrain/source/replay": _write_store(root, "replay-x", rows_replay),
    }
    catalog = {
        "version": 1,
        "splits": {
            "train": {
                name: {"cache_dir": str(root), "ledger": json.loads(ledger.to_json())}
                for name, ledger in ledgers.items()
            }
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))

    sidecar = {
        "version": 1,
        "sources": {
            "sft/source/with-targets": {
                "loss_mode": "chat_assistant_only",
                "loss_weights_indexing": "prediction",
                "loss_weights_key": "loss_weights",
                "teacher_only": False,
                "source_revision": "a",
            },
            "sft/source/clean": {
                "loss_mode": "chat_assistant_only",
                "loss_weights_indexing": "prediction",
                "loss_weights_key": "loss_weights",
                "teacher_only": False,
                "source_revision": "b",
            },
            "pretrain/source/replay": {
                "loss_mode": "full_token",
                "loss_weights_indexing": "prediction",
                "loss_weights_key": "loss_weights",
                "teacher_only": False,
                "source_revision": "c",
            },
        },
    }
    sidecar_path = tmp_path / "source_loss.json"
    sidecar_path.write_text(json.dumps(sidecar))
    pin = {"uri": str(sidecar_path), "sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest()}
    return str(catalog_path), str(root), pin


def test_catalog_flow_versions_every_sft_store(tmp_path, tokenizer_path, header_ids):
    catalog, root, pin = _catalog(tmp_path / "flow", _tokenizer(tokenizer_path), header_ids)
    output_root = str(tmp_path / "flow" / "stores-v1")
    normalize_catalog_part(
        catalog, output_root, tokenizer_path, workers=1, part=0, parts=1, source_loss_pin=SourceLossPin(**pin)
    )

    catalog_output = str(tmp_path / "flow" / "cache_catalog-v8.json")
    assemble_catalog(catalog, output_root, catalog_output, SourceLossPin(**pin))

    v8 = json.loads(Path(catalog_output).read_text())
    train = v8["splits"]["train"]
    # the rewritten store points at its normalized copy with the preserved metadata
    rewritten = train["sft/source/with-targets"]
    assert rewritten["cache_dir"].startswith(output_root)
    assert rewritten["ledger"]["metadata"]["preprocessor_metadata"]["loss_mode"] == "chat_assistant_only"
    assert (Path(rewritten["cache_dir"]) / NORMALIZED_MARKER).exists()
    # the clean store keeps pointing at its source (a pointer) with an unchanged marker
    clean = train["sft/source/clean"]
    assert clean["cache_dir"] == root
    assert (Path(output_root) / "sft__source__clean" / "train" / UNCHANGED_MARKER).exists()
    # the replay entry is untouched
    assert train["pretrain/source/replay"]["cache_dir"] == root

    receipt = json.loads((Path(output_root) / "typography-receipt.json").read_text())
    assert receipt["complete"] is True
    assert receipt["transform"] == {"name": "sft-typography-normalize", "version": "v1"}
    assert receipt["totals"]["stores"] == 2
    assert receipt["totals"]["changed_rows"] == 2
    assert receipt["totals"]["count_nnbsp"] == 1 and receipt["totals"]["count_nbhyphen"] == 1
    assert receipt["totals"]["token_delta"] == receipt["totals"]["out_tokens"] - receipt["totals"]["tokens"]

    # the rewritten store's rows still satisfy the supervision contract
    for out in _read_store(rewritten["cache_dir"]):
        _, expected = assistant_spans(out["input_ids"], header_ids)
        assert np.array_equal(out["loss_weights"].astype(bool), expected)


def test_catalog_part_refuses_a_broken_or_foreign_sidecar(tmp_path, tokenizer_path, header_ids):
    catalog, _root, pin = _catalog(tmp_path / "guard", _tokenizer(tokenizer_path), header_ids)
    broken = dict(pin, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        normalize_catalog_part(
            catalog,
            str(tmp_path / "guard" / "out"),
            tokenizer_path,
            workers=1,
            part=0,
            parts=1,
            source_loss_pin=SourceLossPin(**broken),
        )

    sidecar = json.loads(Path(pin["uri"]).read_text())
    sidecar["sources"]["sft/source/with-targets"]["loss_mode"] = "full_token"
    foreign_path = tmp_path / "guard" / "foreign.json"
    foreign_path.write_text(json.dumps(sidecar))
    foreign = {"uri": str(foreign_path), "sha256": hashlib.sha256(foreign_path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="chat_assistant_only"):
        normalize_catalog_part(
            catalog,
            str(tmp_path / "guard" / "out2"),
            tokenizer_path,
            workers=1,
            part=0,
            parts=1,
            source_loss_pin=SourceLossPin(**foreign),
        )


def test_output_root_may_not_overlap_sources(tmp_path, tokenizer_path, header_ids):
    catalog, _root, _pin = _catalog(tmp_path / "overlap", _tokenizer(tokenizer_path), header_ids)
    document = json.loads(Path(catalog).read_text())
    # V7-style shared cache_dir with nested shard paths: a sibling output root is fine
    document["splits"]["train"]["sft/source/nested"] = {
        "cache_dir": str(tmp_path / "overlap" / "deep-root"),
        "ledger": json.loads(
            CacheLedger(
                total_num_rows=1,
                shard_rows={"shard-x": 1},
                is_finished=True,
                finished_shards=["shard-x"],
                field_counts={"input_ids": 1, "loss_weights": 1},
                layout=CACHE_LAYOUT_SHARDED,
            ).to_json()
        ),
    }
    nested_catalog = tmp_path / "overlap" / "nested.json"
    nested_catalog.write_text(json.dumps(document))
    stores = _sft_stores(json.loads(nested_catalog.read_text()), str(tmp_path / "overlap" / "stores"))
    assert "sft/source/nested" in stores
    # but an output root under a source shard path is refused
    with pytest.raises(ValueError, match="overlaps source store"):
        _sft_stores(json.loads(nested_catalog.read_text()), str(tmp_path / "overlap" / "deep-root" / "shard-x"))


def test_supervision_of_selects_the_mask_rule_from_the_sidecar():
    sidecar = {
        "sft/source/chat": SourceLossMetadata(
            loss_mode=SourceLossMode.CHAT_ASSISTANT_ONLY,
            loss_weights_key="loss_weights",
            loss_weights_indexing="prediction",
            teacher_only=False,
            source_revision="a",
        ),
        "sft/source/relay": SourceLossMetadata(
            loss_mode=SourceLossMode.CHAT_ASSISTANT_ONLY,
            loss_weights_key="loss_weights",
            loss_weights_indexing="prediction",
            teacher_only=True,
            source_revision="r",
        ),
        "sft/source/wrong": SourceLossMetadata(
            loss_mode=SourceLossMode.FULL_TOKEN,
            loss_weights_key="loss_weights",
            loss_weights_indexing="prediction",
            teacher_only=False,
            source_revision="c",
        ),
    }
    assert type(supervision_of(sidecar, "sft/source/chat")) is CensusMaskRule
    assert type(supervision_of(sidecar, "sft/source/relay")) is TeacherMaskRule
    with pytest.raises(ValueError, match="chat_assistant_only"):
        supervision_of(sidecar, "sft/source/wrong")
    with pytest.raises(ValueError, match="no entry"):
        supervision_of(sidecar, "sft/source/missing")
