# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Plateau dropping through train_lm: drops begin at the third visit, and a resumed run reproduces the history."""

import dataclasses
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig, main
from levanter.models.llama import LlamaConfig
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.plateau import PlateauDroppingConfig, PlateauState
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.tracker.tracker import NoopTracker, TrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

ROWS = 8
LENGTH = 16
BATCH = 2
STEPS_PER_EPOCH = ROWS // BATCH
EPOCHS = 6
STEP_METRICS: dict[str, list[dict[str, float]]] = defaultdict(list)


class StepCapture(NoopTracker):
    """Keeps every step's training metrics in STEP_METRICS[run]."""

    def __init__(self, run: str):
        self.run = run

    def log(self, metrics: Mapping[str, Any], *, step: int | None, commit: bool | None = None):
        if step is not None and "train/plateau_dropped_tokens" in metrics:
            STEP_METRICS[self.run].append(
                {"step": step, **{k: float(v) for k, v in metrics.items() if k.startswith("train/")}}
            )


@TrackerConfig.register_subclass("plateau_step_capture")
@dataclasses.dataclass
class StepCaptureConfig(TrackerConfig):
    run: str = ""

    def init(self, run_id: str | None) -> StepCapture:
        return StepCapture(self.run)


def _rows(count: int, vocab: int) -> list[GrugLmExample]:
    rows = []
    for index in range(count):
        tokens = (jnp.arange(LENGTH, dtype=jnp.int32) * (index % 7 + 3) + index) % (vocab - 2) + 2
        weights = (jnp.arange(LENGTH) >= 3 + index % 4).astype(jnp.float32)
        rows.append(GrugLmExample.causal(tokens, loss_weight=weights))
    return rows


def _tokenizer(root: Path, vocab: int) -> Path:
    path = root / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, vocab)}}, unk_token="<pad>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
    ).save_pretrained(path)
    return path


def _config(root: Path, run: str, rows: list[GrugLmExample], model, *, batch: int, **trainer) -> TrainLmConfig:
    """Rows in a fixed order every epoch (one mixture block per epoch, no shuffle), a checkpoint every epoch."""
    steps_per_epoch = len(rows) // batch
    return TrainLmConfig(
        data=LmDataConfig(
            components={"train": DirectDatasetComponent(datasets={"train": ListAsyncDataset(rows)})},
            tokenizer=str(root / "tokenizer"),
            shuffle=False,
            mixture_block_size=len(rows),
        ),
        train_seq_len=LENGTH,
        model=model,
        optimizer=AdamConfig(learning_rate=3e-3, weight_decay=0, warmup=0, lr_schedule="constant", max_grad_norm=1.0),
        trainer=TrainerConfig(
            id=run,
            seed=42,
            train_batch_size=batch,
            num_train_steps=EPOCHS * steps_per_epoch,
            max_eval_batches=0,
            load_checkpoint=False,
            tracker=StepCaptureConfig(run),
            checkpointer=CheckpointerConfig(
                base_path=str(root / run / "checkpoints"),
                append_run_id_to_base_path=False,
                keep=[dict(every=steps_per_epoch)],
                save_interval=None,
            ),
            require_accelerator=False,
            log_jaxprs=False,
            log_xla_hlo=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            **trainer,
        ),
        plateau_dropping=PlateauDroppingConfig(),
    )


def _per_epoch(run: str, steps_per_epoch: int, key: str) -> list[float]:
    totals = defaultdict(float)
    for metrics in STEP_METRICS[run]:
        totals[metrics["step"] // steps_per_epoch] += metrics[f"train/{key}"]
    return [totals[epoch] for epoch in sorted(totals)]


def _resume(config: TrainLmConfig, root: Path, source: str, run: str, updates: int) -> TrainLmConfig:
    return replace(
        config,
        trainer=replace(
            config.trainer,
            id=run,
            load_checkpoint=True,
            load_checkpoint_path=str(root / source / "checkpoints" / f"step-{updates - 1}"),
            tracker=StepCaptureConfig(run),
            checkpointer=replace(config.trainer.checkpointer, base_path=str(root / run / "checkpoints")),
        ),
    )


def _assert_same_history(left: Path, right: Path) -> None:
    expected, actual = PlateauState.loads(left.read_bytes()), PlateauState.loads(right.read_bytes())
    assert expected.rows.keys() == actual.rows.keys()
    for key, row in expected.rows.items():
        assert row.visits == actual.rows[key].visits == EPOCHS
        np.testing.assert_array_equal(row.previous, actual.rows[key].previous)
        np.testing.assert_array_equal(row.last, actual.rows[key].last)


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


@pytest.mark.timeout(600)
def test_plateau_dropping_starts_at_the_third_visit_and_resume_reproduces_it(tmp_path):
    rows = _rows(ROWS, 32)
    config = _config(tmp_path, "full", rows, _llama(_tokenizer(tmp_path, 32)), batch=BATCH)
    main(config)
    dropped = _per_epoch("full", STEPS_PER_EPOCH, "plateau_dropped_tokens")
    supervised = _per_epoch("full", STEPS_PER_EPOCH, "supervised_tokens")
    kept = _per_epoch("full", STEPS_PER_EPOCH, "plateau_kept_tokens")
    assert dropped[0] == dropped[1] == 0
    assert dropped[-1] > dropped[2] > 0
    assert [k + d for k, d in zip(kept, dropped, strict=True)] == supervised

    # The every-epoch keep policy saves at step 12, after 13 updates: resume there, inside epoch 4. The remaining
    # steps and the final history are bit-identical.
    resume_at = 3 * STEPS_PER_EPOCH + 1
    main(_resume(config, tmp_path, "full", "resumed", resume_at))
    assert STEP_METRICS["resumed"] == STEP_METRICS["full"][resume_at:]
    final = f"plateau-updates-{EPOCHS * STEPS_PER_EPOCH}.npz"
    _assert_same_history(tmp_path / "full" / "checkpoints" / final, tmp_path / "resumed" / "checkpoints" / final)

    (tmp_path / "full" / "checkpoints" / f"plateau-updates-{resume_at}.npz").unlink()
    with pytest.raises(FileNotFoundError, match="plateau-updates"):
        main(_resume(config, tmp_path, "full", "missing", resume_at))


@pytest.mark.timeout(900)
def test_plateau_dropping_under_context_parallelism():
    run_on_cpu_devices(f"import runpy; runpy.run_path({__file__!r})['_check_context_parallel']()", device_count=8)


def _check_context_parallel():
    """Tiny Snowball on 8 CPU devices: 2-way context and 4-way expert parallel, against no context axis."""
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
    rows, batch, vocab = _rows(32, 64), 8, 64
    steps_per_epoch = len(rows) // batch
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokenizer = _tokenizer(root, vocab)
        model = SnowballConfig(
            vocab_size=vocab,
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
            sliding_window=8,
            attention_implementation="reference",
            tokenizer=str(tokenizer),
        )
        configs = {
            name: _config(root, name, rows, model, batch=batch, mesh=mesh, use_explicit_mesh_axes=True)
            for name, mesh in meshes.items()
        }
        for config in configs.values():
            main(config)
        parallel = _per_epoch("context", steps_per_epoch, "plateau_dropped_tokens")
        assert parallel[0] == parallel[1] == 0 and parallel[-1] > 0
        assert parallel == _per_epoch("no-context", steps_per_epoch, "plateau_dropped_tokens")
        np.testing.assert_allclose(
            _per_epoch("context", steps_per_epoch, "loss_sum"),
            _per_epoch("no-context", steps_per_epoch, "loss_sum"),
            rtol=2e-3,
        )

        # The every-epoch keep policy saves at step 12, after 13 updates: a resume in the middle of epoch 4.
        resume_at = 3 * steps_per_epoch + 1
        main(_resume(configs["context"], root, "context", "context-resumed", resume_at))
        final = f"plateau-updates-{EPOCHS * steps_per_epoch}.npz"
        _assert_same_history(
            root / "context" / "checkpoints" / final, root / "context-resumed" / "checkpoints" / final
        )
