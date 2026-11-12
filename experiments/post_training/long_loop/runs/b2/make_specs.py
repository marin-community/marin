# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write one B2 arm: 10/05 native start, loop SFT recipe, 10/05 mix replay, optional contributor insert.

    python make_specs.py SPECS_DIR T436_SPECS_DIR OUTPUT_DIR MARIN_REVISION --contrib-loss-share S

S is the contributor share of supervised (loss) tokens over the whole run; 0 is replay only.
Arms differ only in S: same recipe, seed and replay stream. Contributor rows go into evenly spaced
positions and the replay rows keep their order (sft_runtime/insert_dataset.py).
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml

STEPS = 100
BATCH = 256
POSITIONS = 262144
# Contributor packed rows: 452 per epoch, 16,137,263 supervised tokens per epoch (runs/t436 cache summary).
CONTRIB_ROWS_PER_EPOCH = 452
CONTRIB_LOSS_PER_ROW = 16137263 / CONTRIB_ROWS_PER_EPOCH
CONTRIB_EPOCHS = 3
# Replay supervised tokens per 262,144-position row, measured on the 10/05 mix (see REPLAY_LOSS_NOTE).
REPLAY_LOSS_PER_ROW = 0.9459 * POSITIONS
REPLAY_LOSS_NOTE = (
    "mean loss-weight fraction of 72 rows drawn from the first 100 steps of the 10/05 mix (2026-11-12),"
    " std 0.07 per row, standard error 0.009"
)


def insert_rows(share: float) -> int:
    """Rows of contributor data so that it holds ``share`` of the run's loss tokens."""
    total = STEPS * BATCH
    replay = REPLAY_LOSS_PER_ROW
    return round(share * total * replay / (CONTRIB_LOSS_PER_ROW + share * (replay - CONTRIB_LOSS_PER_ROW)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", type=Path)
    parser.add_argument("t436_specs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--contrib-loss-share", type=float, required=True)
    args = parser.parse_args()
    share, d = args.contrib_loss_share, args.output
    if not 0 <= share < 1:
        raise ValueError("--contrib-loss-share must be in [0, 1)")
    d.mkdir(parents=True, exist_ok=True)
    for path in [*args.specs.glob("*.yaml"), *args.specs.glob("*.json")]:
        shutil.copyfile(path, d / path.name)
    for name in ("initializer-1005.yaml", "execution.yaml", "sft-contrib-data.yaml"):
        shutil.copyfile(args.t436_specs / name, d / name)

    def load(name: str):
        return yaml.safe_load((d / name).read_text())

    def dump(name: str, value) -> None:
        (d / name).write_text(yaml.safe_dump(value, sort_keys=False))

    # Loop recipe (MuonH 5e-5, Adam 5e-5, 256 x 262k, CP8, 32 nodes), 100 steps ending in a linear decay.
    sft = load("sft.yaml")
    optimizer = sft["optimizer"]
    optimizer.update(learning_rate=5.0e-5, adam_lr=5.0e-5, lr_schedule="linear", warmup=0, decay=STEPS // 10)
    sft["stop_after_steps"] = STEPS
    t = sft["trainer"]
    t["reseed_special_token_rows"] = False  # 10/05 already trained its special-token rows.
    # MuonH moves each matrix by lr * ||W|| per step, ~5e-5 per element: below bf16 rounding (~2e-3).
    t["trainer"]["mp"] = "compute=bfloat16,params=float32,output=bfloat16"
    t["trainer"]["num_train_steps"] = STEPS
    assert t["trainer"]["train_batch_size"] == BATCH and sft["model"]["max_seq_len"] == POSITIONS
    t["trainer"]["checkpointer"]["keep"] = []  # The final save is permanent.
    arm = "replay" if share == 0 else f"contrib-{round(share * 100)}pct-loss"
    t["trainer"]["tracker"]["tags"] = ["long-loop", "b2", arm]
    t["trainer"]["tracker"]["entity"] = "marin-community"
    t["trainer"]["initialize_from"] = "UNBOUND"
    dump("sft-b2.yaml", sft)

    spec = load("round1.yaml")
    if share > 0:
        rows = insert_rows(share)
        insert = load("sft-contrib-data.yaml")
        epochs = [f"epoch-{e}" for e in range(CONTRIB_EPOCHS)]
        cycles = -(-rows // CONTRIB_ROWS_PER_EPOCH)
        # Read with batch size 1, so stages count rows; the three shuffled epochs repeat in turn.
        insert["data"]["mixture_block_size"] = 1
        insert["data"]["train_weights"] = [
            [CONTRIB_ROWS_PER_EPOCH * i, {name: float(name == epochs[i % CONTRIB_EPOCHS]) for name in epochs}]
            for i in range(cycles)
        ]
        insert["rows"] = rows
        insert["provenance"]["insert"] = {
            "loss_share": share,
            "rows": rows,
            "contributor_epochs": rows / CONTRIB_ROWS_PER_EPOCH,
            "position_share": rows / (STEPS * BATCH),
            "replay_loss_per_row": REPLAY_LOSS_PER_ROW,
            "replay_loss_note": REPLAY_LOSS_NOTE,
        }
        dump("sft-insert.yaml", insert)
        spec["datasets"]["sft_insert"] = {"path": "sft-insert.yaml"}
    (d / "sft-contrib-data.yaml").unlink()

    spec["name"] = f"long-loop/b2-{arm}"
    spec["marin_revision"] = args.revision
    spec["sft"]["path"] = "sft-b2.yaml"
    spec["initializer"]["path"] = "initializer-1005.yaml"
    spec["smoke_overrides"] = {"sft_steps": 2, "rl_steps": 2, "eval_tasks_per_benchmark": 1, "eval_replicas": 1}
    spec["storage"] = {
        "output_prefix": f"s3://marin-rno2a/tmp/ttl=14d/long-loop-b2/{arm}",
        "temporary_prefix": f"s3://marin-rno2a/tmp/ttl=3d/long-loop-b2/{arm}",
    }
    for name, priority_path in (
        ("execution.yaml", ("coordinator",)),
        ("rl.yaml", ("execution",)),
        ("evaluation.yaml", ("execution",)),
        ("endpoint.yaml", ()),
    ):
        doc = load(name)
        node = doc
        for key in priority_path:
            node = node[key]
        node["priority"] = "interactive"
        dump(name, doc)

    def pin(section):
        section["sha256"] = hashlib.sha256((d / section["path"]).read_bytes()).hexdigest()

    for section in spec["datasets"].values():
        pin(section)
    for key in ("sft", "initializer", "tokenizer", "endpoint", "execution", "archive", "rl", "evaluation"):
        pin(spec[key])
    dump("round1.yaml", spec)
    for name in ("sft.yaml", "initializer.yaml"):
        (d / name).unlink()
    print("ok", spec["name"])


if __name__ == "__main__":
    main()
