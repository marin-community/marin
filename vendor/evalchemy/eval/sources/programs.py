"""Data transformations shared by program-evaluation sources."""

import base64
from collections.abc import Iterator
from dataclasses import dataclass
import json
import pickle
from typing import Any
import zlib

from datasets import Dataset


def difficulty(rating):
    if not rating:
        return "Easy"
    if rating < 1000 or rating > 3500:
        return "Easy"
    return "Medium" if rating < 1300 else "Hard"


def exact_tests(tests):
    normalized = []
    for test in tests:
        output = test["output"].strip()
        if output.endswith("-"):
            output = output[:output.rfind("-")].rstrip()
        normalized.append({"input": test["input"], "output": output})
    return normalized


@dataclass(frozen=True)
class LiveCodeRecords:
    """Size-aware source view that decodes private tests only on consumption."""

    rows: Dataset

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(4):
            shard = self.rows.shard(num_shards=4, index=index)
            for position in range(len(shard)):
                # Dataset iteration formats batches; indexing keeps even the
                # compressed private payloads bounded to one source row here.
                row = shard[position]
                yield _live_code_record({**row, "private_test_cases": _private_tests(row["private_test_cases"])})


def live_code_records(rows):
    return LiveCodeRecords(rows)


def _private_tests(encoded_data):
    decoded = zlib.decompress(base64.b64decode(encoded_data))
    return json.loads(pickle.loads(decoded))


def _live_code_record(row):
    public_tests = json.loads(row["public_test_cases"])
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": any(test.get("testtype") == "stdin" for test in public_tests),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
    }
