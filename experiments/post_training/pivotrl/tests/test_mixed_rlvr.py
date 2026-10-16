# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.nemotron_pivot import VALIDATION_FILENAME
from marin.rl.pass_rates import PASS_RATES_FILENAME

from experiments.post_training.pivotrl.mixed_rlvr import ROUTE_COLUMN, Route, domain_weights, mix_routes
from experiments.post_training.pivotrl.selection import TRAIN_FILENAME

RLVR_MESSAGE = pa.struct([("role", pa.large_string()), ("content", pa.large_string())])
PIVOT_MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()), ("reasoning_content", pa.string())])


def _extra_info(count: int, route: str = "skyrl_gym") -> list[dict]:
    ultra = {"agent": "a", "route": route, "request_json": json.dumps({"tools": []})}
    return [{"index": index, "nemotron_ultra": ultra} for index in range(count)]


def _write(path, filename: str, table: pa.Table) -> dict[str, str]:
    (path / filename).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path / filename)
    return {"path": str(path), "filename": filename}


def _rlvr(ids: list[str]) -> pa.Table:
    prompts = [[{"role": "user", "content": row_id}] for row_id in ids]
    columns = {"prompt": pa.array(prompts, pa.large_list(RLVR_MESSAGE)), "id": pa.array(ids, pa.large_string())}
    return pa.table({**columns, "extra_info": _extra_info(len(ids))})


def _pivots(ids: list[str], **counts: list) -> pa.Table:
    prompts = [[{"role": "assistant", "content": row_id, "reasoning_content": "think"}] for row_id in ids]
    columns = {"prompt": pa.array(prompts, pa.list_(PIVOT_MESSAGE)), "id": pa.array(ids)}
    return pa.table({**columns, "extra_info": _extra_info(len(ids)), **counts})


def test_arms_tag_rows_by_route_rlvr_first_and_drop_unscored_turns_and_blank_messages(tmp_path):
    # Every row also moves to the task-session route MarinSkyRL grades Nemotron Ultra rows on, and a
    # capped pool's rows carry their reply cap in their request.
    rlvr = _write(tmp_path / "rlvr", "data/train.parquet", _rlvr(["r0", "r1", " "]))
    swe = {**_write(tmp_path / "swe", TRAIN_FILENAME, _pivots(["s0"])), "max_output_tokens": 4096}
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
    assert set(train.column_names) == {"prompt", "id", "extra_info", ROUTE_COLUMN}
    ultra = [row["nemotron_ultra"] for row in train.column("extra_info").to_pylist()]
    assert {row["route"] for row in ultra} == {"task_session"}
    caps = [json.loads(row["request_json"]).get("max_output_tokens") for row in ultra]
    assert caps == [None, None, 4096, None, None]
    messages = train.column("prompt").to_pylist()
    assert messages[0][0]["reasoning_content"] is None and messages[2][0]["reasoning_content"] == "think"
    assert pq.read_table(out / VALIDATION_FILENAME).column(ROUTE_COLUMN).to_pylist() == ["rlvr", "swe"]
    assert json.loads((out / "manifest.json").read_text()) == {
        "train": {
            "rlvr": {"rows": 2, "dropped_empty_message": 1},
            "swe": {"rows": 1, "dropped_empty_message": 0},
            "terminal": {"rows": 2, "dropped_empty_message": 0},
        },
        "validation": {"rlvr": {"rows": 1, "dropped_empty_message": 0}, "swe": {"rows": 1, "dropped_empty_message": 0}},
    }


def test_pivot_share_splits_evenly_between_swe_and_terminal_and_fills_the_batch():
    assert domain_weights(0.2) == {Route.RLVR: 410, Route.SWE: 51, Route.TERMINAL: 51}
    assert domain_weights(0.1) == {Route.RLVR: 461, Route.SWE: 26, Route.TERMINAL: 25}
    assert domain_weights(0.0) == {Route.RLVR: 512}
