import json
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig, main
from levanter.models.llama import LlamaConfig
from levanter.optim.config import AdamConfig
from levanter.plateau import PlateauDroppingConfig, PlateauState
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from safetensors.numpy import load_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.agentic_sweep.callbacks import EpochTracker, EpochTrackerConfig, SweepExportsConfig

STEPS_PER_EPOCH = 4
EPOCHS = 8


def _rows():
    rows = []
    for index in range(8):
        tokens = (jnp.arange(16, dtype=jnp.int32) * (index + 3) + index) % 30 + 2
        weights = (jnp.arange(16) >= 4 + index % 3).astype(jnp.float32)
        rows.append(GrugLmExample.causal(tokens, loss_weight=weights))
    return rows


def _config(tmp_path, tokenizer_path) -> TrainLmConfig:
    rows = _rows()
    supervised = int(sum(float(row.loss_weight.sum()) for row in rows))
    data = LmDataConfig(
        components={"train": DirectDatasetComponent(datasets={"train": ListAsyncDataset(rows)})},
        tokenizer=str(tokenizer_path),
        shuffle=False,
        mixture_block_size=len(rows),
    )
    validation = LmDataConfig(
        components={"tmax": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(rows[:2])})},
        tokenizer=str(tokenizer_path),
        shuffle=False,
    )
    return TrainLmConfig(
        data=data,
        train_seq_len=16,
        model=LlamaConfig(
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=16,
            hidden_dim=32,
            intermediate_dim=64,
            attn_backend=None,
            reference_checkpoint=str(tokenizer_path),
        ),
        optimizer=AdamConfig(learning_rate=3e-3, weight_decay=0, warmup=0, lr_schedule="constant", max_grad_norm=1.0),
        trainer=TrainerConfig(
            id="plateau-integration",
            seed=42,
            train_batch_size=2,
            per_device_parallelism=2,
            num_train_steps=EPOCHS * STEPS_PER_EPOCH,
            load_checkpoint=False,
            tracker=(EpochTrackerConfig(str(tmp_path), STEPS_PER_EPOCH, supervised), JsonLoggerConfig()),
            checkpointer=CheckpointerConfig(
                base_path=str(tmp_path / "checkpoints"),
                append_run_id_to_base_path=False,
                keep=[],
                save_interval=None,
            ),
            require_accelerator=False,
            log_jaxprs=False,
            log_xla_hlo=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
        adapter=SweepExportsConfig(validation, STEPS_PER_EPOCH, 16),
        hf_save_path=str(tmp_path / "hf"),
        plateau_dropping=PlateauDroppingConfig(),
    )


def _epoch_rows(path):
    return [json.loads(line) for line in (path / "epoch-metrics.jsonl").read_text().splitlines()]


def test_plateau_epoch_statistics(tmp_path):
    tracker = EpochTracker(str(tmp_path), steps_per_epoch=2, tokens_per_epoch=10)
    for step, (kept, kept_loss, dropped, dropped_loss) in enumerate([(4, 4.0, 1, 0.1), (3, 3.0, 2, 0.2)]):
        tracker.log(
            {
                "train/loss_sum": kept_loss + dropped_loss,
                "train/supervised_tokens": kept + dropped,
                "train/plateau_kept_tokens": kept,
                "train/plateau_kept_loss_sum": kept_loss,
                "train/plateau_dropped_tokens": dropped,
                "train/plateau_dropped_loss_sum": dropped_loss,
            },
            step=step,
        )
    (row,) = _epoch_rows(tmp_path)
    assert row["plateau_dropped_fraction"] == pytest.approx(0.3)
    assert row["plateau_kept_loss"] == pytest.approx(1.0)
    assert row["plateau_dropped_loss"] == pytest.approx(0.1)


@pytest.mark.timeout(600)
def test_plateau_dropping_rises_over_epochs_and_resume_reproduces_the_mask(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 32)}}, unk_token="<pad>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
    ).save_pretrained(tokenizer_path)
    run_path = tmp_path / "run"
    config = _config(run_path, tokenizer_path)
    main(config)
    rows = _epoch_rows(run_path)
    fractions = [row["plateau_dropped_fraction"] for row in rows]
    assert [row["completed_updates"] for row in rows] == [STEPS_PER_EPOCH * (e + 1) for e in range(EPOCHS)]
    assert fractions[0] == fractions[1] == 0
    assert fractions[-1] > fractions[2] > 0
    assert all(row["supervised_tokens"] == row["plateau_kept_tokens"] + row["plateau_dropped_tokens"] for row in rows)
    final_updates = EPOCHS * STEPS_PER_EPOCH
    saved = sorted(path.name for path in (run_path / "checkpoints").glob("plateau-updates-*.npz"))
    assert saved == sorted(f"plateau-updates-{STEPS_PER_EPOCH * (e + 1)}.npz" for e in range(EPOCHS))

    resume_at = 3 * STEPS_PER_EPOCH
    resumed_path = tmp_path / "resumed"
    resumed = _config(resumed_path, tokenizer_path)
    resumed = replace(
        resumed,
        trainer=replace(
            resumed.trainer,
            load_checkpoint=True,
            load_checkpoint_path=str(run_path / "checkpoints" / f"step-{resume_at - 1}"),
            tracker=(
                EpochTrackerConfig(str(resumed_path), STEPS_PER_EPOCH, rows[0]["manifest_supervised_tokens"], resume_at),
                JsonLoggerConfig(),
            ),
        ),
    )
    main(resumed)
    resumed_rows = _epoch_rows(resumed_path)
    plateau_keys = ("plateau_kept_tokens", "plateau_dropped_tokens", "train_loss")
    assert [[row[key] for key in plateau_keys] for row in resumed_rows] == [
        [row[key] for key in plateau_keys] for row in rows[3:]
    ]
    expected = PlateauState.loads((run_path / "checkpoints" / f"plateau-updates-{final_updates}.npz").read_bytes())
    actual = PlateauState.loads((resumed_path / "checkpoints" / f"plateau-updates-{final_updates}.npz").read_bytes())
    assert expected.rows.keys() == actual.rows.keys()
    for key, row in expected.rows.items():
        np.testing.assert_array_equal(row.last, actual.rows[key].last)
        assert row.visits == actual.rows[key].visits == EPOCHS
    for shard in (run_path / "hf" / f"step-{final_updates - 1}").glob("*.safetensors"):
        original = load_file(str(shard))
        restored = load_file(str(resumed_path / "hf" / f"step-{final_updates - 1}" / shard.name))
        for name in original:
            np.testing.assert_array_equal(original[name], restored[name])

    missing = replace(
        resumed,
        trainer=replace(
            resumed.trainer,
            load_checkpoint_path=str(run_path / "checkpoints" / f"step-{resume_at - 1}"),
            checkpointer=replace(resumed.trainer.checkpointer, base_path=str(tmp_path / "missing")),
        ),
    )
    (run_path / "checkpoints" / f"plateau-updates-{resume_at}.npz").unlink()
    with pytest.raises(FileNotFoundError, match="plateau-updates"):
        main(missing)
