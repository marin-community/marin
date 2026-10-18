# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import itertools
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.formats import PrebuiltCacheProcessor
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig, main
from levanter.models.llama import LlamaConfig
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.plateau import row_key
from levanter.schedule import BatchSchedule
from levanter.store.cache import SerialCacheWriter
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from safetensors.numpy import load_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.agentic_sweep.arms import (
    arm_data,
    arm_epochs,
    arm_settings,
    epoch_file,
    row_context,
    verify_arm,
    verify_caches,
)

LENGTH = 32
PAD = 0
VOCAB = 64
REPLAY_ROWS = 8


def _row(seed: int) -> tuple[list[int], list[int]]:
    """A builder-style row: real tokens then padding; weight i scores token i + 1, zero from the last real token."""
    rng = np.random.default_rng(seed)
    length = int(rng.integers(LENGTH // 2, LENGTH + 1))
    ids = [int(token) for token in rng.integers(2, VOCAB, length)] + [PAD] * (LENGTH - length)
    start = int(rng.integers(1, length // 2))
    weights = [int(start <= index < length - 1) for index in range(LENGTH)]
    return ids, weights


def _line(row: tuple[list[int], list[int]]) -> str:
    return json.dumps({"input_ids": row[0], "loss_weights": row[1]}, separators=(",", ":"))


def _write_arm(root: Path, arm: str, agentic: list[list[int]], replay_seed: int = 0) -> Path:
    """Builder outputs: one file per epoch of the given agentic row seeds plus the shared replay, seeded order."""
    root.mkdir(parents=True)
    replay = [_row(10_000 + replay_seed + index) for index in range(REPLAY_ROWS)]
    splits, index, hashes = {}, [], {}
    for epoch, seeds in enumerate(agentic):
        entries = [(_row(seed), "tmax") for seed in seeds] + [(row, "replay") for row in replay]
        order = np.random.default_rng(epoch).permutation(len(entries))
        padding = -len(entries) % 16
        lines = [_line(entries[position][0]) for position in order] + [_line(([PAD] * LENGTH, [0] * LENGTH))] * padding
        text = "".join(line + "\n" for line in lines).encode()
        data = gzip.compress(text)
        (root / epoch_file(epoch)).write_bytes(data)
        hashes[epoch_file(epoch)] = hashlib.sha256(data).hexdigest()
        supervised, context = {}, {}
        for row_number, position in enumerate(order):
            (ids, weights), family = entries[position]
            supervised[family] = supervised.get(family, 0) + sum(weights)
            context[family] = context.get(family, 0) + row_context(np.asarray(ids), PAD)
            index.append({"epoch": epoch, "row": row_number, "family": family})
        splits[f"epoch-{epoch}"] = {
            "rows": len(entries),
            "padding_rows": padding,
            "padded_rows": len(entries) + padding,
            "steps": (len(entries) + padding) // 16,
            "supervised_tokens": supervised,
            "context_tokens": context,
            "supervised_total": sum(supervised.values()),
            "rows_sha256": hashlib.sha256(text).hexdigest(),
        }
    (root / "train-index.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in index))
    manifest = {
        "arm": arm,
        "batch_size": 16,
        "max_sequence_tokens": LENGTH,
        "pad_token_id": PAD,
        "splits": splits,
        "replay": {"rows_sha256": "replay-cut", "rows": REPLAY_ROWS},
        "output_hashes": hashes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def _write_caches(root: Path, epochs: int) -> list[str]:
    """What the tokenize step writes for each epoch file: one prebuilt cache, rows in file order."""
    processor = PrebuiltCacheProcessor("input_ids", "loss_weights")
    paths = []
    for epoch in range(epochs):
        path = root / "caches" / f"epoch-{epoch}"
        rows = [json.loads(line) for line in gzip.decompress((root / epoch_file(epoch)).read_bytes()).splitlines()]
        with SerialCacheWriter(str(path), processor.output_exemplar) as writer:
            writer.write_batch(processor(rows))
        paths.append(str(path))
    return paths


def _tokenizer(root: Path) -> Path:
    path = root / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"<pad>": PAD, "<eos>": 1, **{f"t{i}": i for i in range(2, VOCAB)}}, unk_token="<pad>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
    ).save_pretrained(path)
    return path


def _rs_multi(root: Path) -> tuple[Path, Path]:
    """Plain repeats its rows; RS-multi draws new rows each epoch, with epochs of 4, 5 and 5 updates."""
    plain = _write_arm(root / "plain", "plain", [list(range(56))] * 3)
    rs_multi = _write_arm(
        root / "rs-multi", "rs-multi", [list(range(1000, 1056)), list(range(2000, 2066)), list(range(3000, 3060))]
    )
    return plain, rs_multi


def _train_config(
    root: Path, arm: Path, epochs: int, name: str, model, plateau: bool = False, **trainer
) -> TrainLmConfig:
    tokenizer = _tokenizer(root)
    plan = arm_epochs(_manifest(arm), epochs)
    output = root / name
    config = TrainLmConfig(
        data=arm_data(_caches(arm, epochs), plan, str(tokenizer)),
        train_seq_len=LENGTH,
        model=model(tokenizer),
        optimizer=AdamConfig(learning_rate=3e-3, weight_decay=0, warmup=0, lr_schedule="constant", max_grad_norm=1.0),
        trainer=TrainerConfig(
            id=name,
            seed=42,
            train_batch_size=16,
            max_eval_batches=0,
            load_checkpoint=False,
            tracker=JsonLoggerConfig(),
            checkpointer=CheckpointerConfig(base_path=str(output / "checkpoints"), append_run_id_to_base_path=False),
            require_accelerator=False,
            log_jaxprs=False,
            log_xla_hlo=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            **trainer,
        ),
        hf_save_path=str(output / "hf"),
    )
    return arm_settings(config, plan, str(output), PAD, plateau)


def _caches(arm: Path, epochs: int) -> list[str]:
    if (arm / "caches").exists():
        return [str(arm / "caches" / f"epoch-{epoch}") for epoch in range(epochs)]
    return _write_caches(arm, epochs)


def _llama(tokenizer: Path) -> LlamaConfig:
    return LlamaConfig(
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=LENGTH,
        hidden_dim=32,
        intermediate_dim=64,
        attn_backend=None,
        reference_checkpoint=str(tokenizer),
    )


def _snowball(tokenizer: Path) -> SnowballConfig:
    return SnowballConfig(
        vocab_size=VOCAB,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=LENGTH,
        sliding_window=16,
        attention_implementation="reference",
        tokenizer=str(tokenizer),
    )


def _epoch_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "epoch-metrics.jsonl").read_text().splitlines()]


def _rewrite(arm: Path, epoch: int, lines: list[str], *, fix_rows_sha: bool = True) -> None:
    text = "".join(line + "\n" for line in lines).encode()
    data = gzip.compress(text)
    (arm / epoch_file(epoch)).write_bytes(data)
    manifest = _manifest(arm)
    manifest["output_hashes"][epoch_file(epoch)] = hashlib.sha256(data).hexdigest()
    if fix_rows_sha:
        manifest["splits"][f"epoch-{epoch}"]["rows_sha256"] = hashlib.sha256(text).hexdigest()
    (arm / "manifest.json").write_text(json.dumps(manifest))


def _lines(arm: Path, epoch: int) -> list[str]:
    return gzip.decompress((arm / epoch_file(epoch)).read_bytes()).decode().splitlines()


def test_verification_rejects_changed_files_replay_and_cache_order(tmp_path):
    plain, rs_multi = _rs_multi(tmp_path)
    keys = verify_arm(str(rs_multi), str(plain), _manifest(rs_multi), 3)
    assert [len(epoch) for epoch in keys] == [64, 80, 80]
    # Only the replay rows (and, between padded epochs, the padding row) recur across epochs.
    assert len(set(keys[0]) & set(keys[1])) == REPLAY_ROWS
    assert len(set(keys[1]) & set(keys[2])) == REPLAY_ROWS + 1
    verify_caches(_write_caches(rs_multi, 3), keys)

    manifest = _manifest(rs_multi)
    manifest["splits"]["epoch-1"]["steps"] = 4
    with pytest.raises(ValueError, match="whole batches"):
        arm_epochs(manifest, 2)
    with pytest.raises(ValueError, match="no epoch file"):
        arm_epochs(_manifest(rs_multi), 4)

    # Rows swapped in a cache: same rows, wrong order.
    cache_arm = _write_arm(tmp_path / "swapped", "rs-multi", [list(range(1000, 1056))])
    lines = _lines(cache_arm, 0)
    good_keys = verify_arm(str(cache_arm), str(plain), _manifest(cache_arm), 1)
    _rewrite(cache_arm, 0, [lines[1], lines[0], *lines[2:]])
    with pytest.raises(ValueError, match="cache row 0 differs"):
        verify_caches(_write_caches(cache_arm, 1), good_keys)

    # A changed file whose manifest hash was not updated.
    (rs_multi / epoch_file(2)).write_bytes(gzip.compress(b"\n".join(line.encode() for line in _lines(rs_multi, 1))))
    with pytest.raises(ValueError, match="output hash"):
        verify_arm(str(rs_multi), str(plain), _manifest(rs_multi), 3)

    # A file whose rows differ from rows_sha256.
    changed = _write_arm(tmp_path / "rows", "relay", [list(range(56))])
    _rewrite(changed, 0, list(reversed(_lines(changed, 0))), fix_rows_sha=False)
    with pytest.raises(ValueError, match="rows_sha256"):
        verify_arm(str(changed), str(plain), _manifest(changed), 1)

    # A replay row that differs from plain's, with every hash consistent.
    other_replay = _write_arm(tmp_path / "replay", "relay", [list(range(56))], replay_seed=1)
    with pytest.raises(ValueError, match="replay rows are not byte-identical"):
        verify_arm(str(other_replay), str(plain), _manifest(other_replay), 1)


@pytest.mark.timeout(600)
def test_rs_multi_epoch_k_trains_only_epoch_k_rows(tmp_path):
    plain, rs_multi = _rs_multi(tmp_path)
    keys = verify_arm(str(rs_multi), str(plain), _manifest(rs_multi), 3)
    plan = arm_epochs(_manifest(rs_multi), 3)
    config = _train_config(tmp_path, rs_multi, 3, "rs-multi-run", _llama)

    # The loader's batches: step s of epoch k holds rows 16b..16b+15 of epoch k's file and nothing else.
    dataset = config.data.train_set(
        config.model.max_Pos.resize(LENGTH), BatchSchedule(16), key=jax.random.PRNGKey(0)
    ).as_sync_dataset()
    other_epochs = [set().union(*(set(keys[j]) for j in range(3) if j != k)) - set(keys[k]) for k in range(3)]
    step = 0
    for epoch, info in enumerate(plan):
        for batch in range(info.updates):
            examples = dataset.get_batch(range(16 * step, 16 * step + 16))
            trained = [row_key(np.asarray(e.tokens.array), np.asarray(e.loss_weight.array)) for e in examples]
            assert sorted(trained) == sorted(keys[epoch][16 * batch : 16 * batch + 16])
            assert not set(trained) & other_epochs[epoch]
            step += 1

    main(config)
    output = tmp_path / "rs-multi-run"
    rows = _epoch_rows(output)
    ends = list(itertools.accumulate(info.updates for info in plan))
    assert [row["completed_updates"] for row in rows] == ends == [4, 9, 14]
    assert [row["supervised_tokens"] for row in rows] == [info.supervised_tokens for info in plan]
    assert [row["context_tokens"] for row in rows] == [info.context_tokens for info in plan]
    assert len({info.supervised_tokens for info in plan}) == 3
    assert sorted(path.name for path in (output / "checkpoints").glob("step-*")) == sorted(
        f"step-{end - 1}" for end in ends
    )
    assert sorted(path.name for path in (output / "hf").iterdir()) == sorted(f"step-{end - 1}" for end in ends[1:])


@pytest.mark.timeout(900)
@pytest.mark.parametrize("plateau", [False, True], ids=["plain-loss", "plateau"])
def test_resume_from_an_epoch_checkpoint_is_bit_exact(tmp_path, plateau):
    plain = _write_arm(tmp_path / "plain", "plain", [list(range(56))] * 3)
    config = _train_config(tmp_path, plain, 3, "full", _llama, plateau)
    main(config)
    rows = _epoch_rows(tmp_path / "full")
    if plateau:
        assert all("plateau_dropped_fraction" in row for row in rows)
        assert rows[0]["plateau_dropped_tokens"] == rows[1]["plateau_dropped_tokens"] == 0

    resumed = _train_config(tmp_path, plain, 3, "resumed", _llama, plateau)
    epoch_tracker = resumed.trainer.tracker[-1]
    resumed = replace(
        resumed,
        trainer=replace(
            resumed.trainer,
            load_checkpoint=True,
            load_checkpoint_path=str(tmp_path / "full" / "checkpoints" / "step-3"),
            tracker=(*resumed.trainer.tracker[:-1], replace(epoch_tracker, start_step=4)),
        ),
    )
    main(resumed)
    resumed_rows = _epoch_rows(tmp_path / "resumed")
    assert [row["completed_updates"] for row in resumed_rows] == [8, 12]
    assert resumed_rows == rows[1:]
    for shard in (tmp_path / "full" / "hf" / "step-11").glob("*.safetensors"):
        expected = load_file(str(shard))
        restored = load_file(str(tmp_path / "resumed" / "hf" / "step-11" / shard.name))
        assert expected.keys() == restored.keys()
        for name in expected:
            np.testing.assert_array_equal(expected[name], restored[name])


@pytest.mark.timeout(900)
def test_arm_trainer_on_a_context_parallel_snowball_mesh():
    run_on_cpu_devices(
        "import runpy; runpy.run_path('tests/experiment/test_arm_trainer.py')['_check_context_parallel_arm']()",
        device_count=8,
    )


def _check_context_parallel_arm():
    """96-row RS-multi epochs on 8 CPU devices: 2-way context and 4-way expert parallel, against no context axis."""
    meshes = {
        "context": MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "context": 2, "expert": -1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        ),
        "no-context": MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "expert": -1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "vocab": "model"},
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_arm(root / "plain", "plain", [list(range(88))] * 2)
        arm = _write_arm(root / "rs-multi", "rs-multi", [list(range(1000, 1088)), list(range(2000, 2088))])
        plan = arm_epochs(_manifest(arm), 2)
        assert [info.rows for info in plan] == [96, 96]
        verify_caches(_write_caches(arm, 2), verify_arm(str(arm), str(root / "plain"), _manifest(arm), 2))
        results = {}
        for name, mesh in meshes.items():
            main(_train_config(root, arm, 2, name, _snowball, mesh=mesh, use_explicit_mesh_axes=True))
            results[name] = _epoch_rows(root / name)
        for left, right, info in zip(results["context"], results["no-context"], plan, strict=True):
            assert left["supervised_tokens"] == right["supervised_tokens"] == info.supervised_tokens
            assert left["context_tokens"] == right["context_tokens"] == info.context_tokens
            np.testing.assert_allclose(left["train_loss"], right["train_loss"], rtol=2e-3)
        assert [row["completed_updates"] for row in results["context"]] == [6, 12]
        assert (root / "context" / "hf" / "step-11" / "config.json").exists()
