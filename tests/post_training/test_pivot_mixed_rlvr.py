# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.nemotron_pivot import VALIDATION_FILENAME
from marin.rl.pass_rates import PASS_RATES_FILENAME

from experiments.post_training.pivotrl.mixed_rlvr import ROUTE_COLUMN, domain_weights, mix_routes
from experiments.post_training.pivotrl.selection import TRAIN_FILENAME

RLVR_MESSAGE = pa.struct([("role", pa.large_string()), ("content", pa.large_string())])
PIVOT_MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()), ("reasoning_content", pa.string())])


def _write(path, filename: str, table: pa.Table) -> dict[str, str]:
    (path / filename).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path / filename)
    return {"path": str(path), "filename": filename}


def _rlvr(ids: list[str]) -> pa.Table:
    prompts = [[{"role": "user", "content": row_id}] for row_id in ids]
    return pa.table({"prompt": pa.array(prompts, pa.large_list(RLVR_MESSAGE)), "id": pa.array(ids, pa.large_string())})


def _pivots(ids: list[str], **counts: list) -> pa.Table:
    prompts = [[{"role": "assistant", "content": row_id, "reasoning_content": "think"}] for row_id in ids]
    columns = {"prompt": pa.array(prompts, pa.list_(PIVOT_MESSAGE)), "id": pa.array(ids)}
    return pa.table({**columns, **counts})


def test_arms_tag_rows_by_route_rlvr_first_and_random_turns_drop_unscored_rows(tmp_path):
    rlvr = _write(tmp_path / "rlvr", "data/train.parquet", _rlvr(["r0", "r1"]))
    swe = _write(tmp_path / "swe", TRAIN_FILENAME, _pivots(["s0"]))
    scored = _pivots(
        ["t0", "t1", "t2"],
        passed=[3, 0, 16],
        total=[16, 0, 16],
        pass_rate=[3 / 16, None, 1.0],
        exclusion=[None, "context_window", None],
        passed_exact=[1, 0, 16],
    )
    terminal = _write(tmp_path / "terminal", PASS_RATES_FILENAME, scored)
    validation = _write(tmp_path / "rlvr", "data/validation.parquet", _rlvr(["v0"]))
    out = tmp_path / "out"
    out.mkdir()

    mix_routes(
        train={"rlvr": rlvr, "swe": swe, "terminal": terminal},
        validation={"rlvr": validation, "swe": _write(tmp_path / "swe", VALIDATION_FILENAME, _pivots(["sv"]))},
        output_path=str(out),
    )

    train = pq.read_table(out / TRAIN_FILENAME)
    assert train.column("id").to_pylist() == ["r0", "r1", "s0", "t0", "t2"]
    assert train.column(ROUTE_COLUMN).to_pylist() == ["rlvr", "rlvr", "swe", "terminal", "terminal"]
    assert set(train.column_names) == {"prompt", "id", ROUTE_COLUMN}
    messages = train.column("prompt").to_pylist()
    assert messages[0][0]["reasoning_content"] is None and messages[2][0]["reasoning_content"] == "think"
    assert pq.read_table(out / VALIDATION_FILENAME).column(ROUTE_COLUMN).to_pylist() == ["rlvr", "swe"]
    assert json.loads((out / "manifest.json").read_text()) == {
        "train": {"rlvr": 2, "swe": 1, "terminal": 2},
        "validation": {"rlvr": 1, "swe": 1},
    }


def test_pivot_share_splits_evenly_between_swe_and_terminal_and_fills_the_batch():
    assert domain_weights(0.2) == {"rlvr": 410, "swe": 51, "terminal": 51}
    assert domain_weights(0.1) == {"rlvr": 461, "swe": 26, "terminal": 25}
    assert domain_weights(0.0) == {"rlvr": 512}
