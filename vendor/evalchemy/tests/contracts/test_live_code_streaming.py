"""LiveCodeBench caps apply before private-test decoding."""

import base64
import json
import pickle
import zlib

from datasets import Dataset
import pytest

from eval.chat_benchmarks.LiveCodeBench import eval_instruct as source_module
from eval.sources.common import SourceOptions
from eval.sources.programs import live_code_records


def encoded_tests(index):
    tests = [{"input": str(index), "output": str(index + 1), "testtype": "stdin"}]
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def raw_rows(count):
    return [{"question_id": f"problem/{index}", "question_content": f"Compute {index}+1.",
             "private_test_cases": encoded_tests(index), "starter_code": "",
             "public_test_cases": json.dumps([{"input": "0", "output": "1", "testtype": "stdin"}]),
             "difficulty": "easy"} for index in range(count)]


@pytest.mark.parametrize("limit,debug,count,selected", [(1, False, 8, 1), (0.25, False, 8, 2), (None, True, 12, 10)])
def test_lcb_caps_leave_later_private_tests_undecoded(monkeypatch, limit, debug, count, selected):
    rows = raw_rows(count)
    rows[selected]["private_test_cases"] = "invalid unselected private payload"
    dataset = Dataset.from_list(rows)
    monkeypatch.setattr(source_module, "load_dataset", lambda *_args, **_kwargs: dataset)
    # Keep the old eager-map regression bounded on hosts with many cores.
    monkeypatch.setattr("os.cpu_count", lambda: 1)
    source = source_module.LiveCodeBenchBenchmark(debug=debug)

    specs = list(source.iter_specs(SourceOptions("LiveCodeBench", source.SOURCE, limit=limit)))

    assert len(specs) == selected * source.n_repeat
    assert [(spec.source_id, spec.ordinal, spec.trial) for spec in specs] == [
        (f"problem/{index}", index, trial) for index in range(selected) for trial in range(source.n_repeat)
    ]
    assert all(spec.target == [{"input": str(spec.ordinal), "output": str(spec.ordinal + 1), "testtype": "stdin"}]
               for spec in specs)


def test_lcb_records_preserve_native_four_shard_order_and_values():
    rows = raw_rows(9)
    rows[1]["public_test_cases"] = json.dumps([{"testtype": "functional"}])
    records = live_code_records(Dataset.from_list(rows))

    assert len(records) == len(rows)
    assert list(records) == [{
        "prompt": row["question_content"],
        "test": [{"input": str(index), "output": str(index + 1), "testtype": "stdin"}],
        "entry_point": "", "task_id": row["question_id"], "is_stdin": index != 1,
        "public_test_cases": row["public_test_cases"], "difficulty": "easy",
    } for index, row in enumerate(rows)]
