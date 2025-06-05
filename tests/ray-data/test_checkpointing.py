import json
import os

import fsspec
import pytest
import ray

from marin.generation.inference import find_all_finished_ids

TEST_CHECKPOINT_DIR = "gs://marin-us-east1/documents/checkpoint-test"
TEST_INPUT_DIR = "gs://marin-us-east1/documents/input-dir-test"
TEST_INPUT_PATH = os.path.join(TEST_INPUT_DIR, "input.jsonl.gz")
TEST_FINISHED_IDS_PATH = os.path.join(TEST_CHECKPOINT_DIR, "finished_ids.jsonl.gz")


@pytest.fixture(scope="module")
def input_path():
    lines = [
        {"id": "doc_1"},
        {"id": "doc_2"},
        {"id": "doc_3"},
        {"id": "doc_4"},
        {"id": "doc_5"},
    ]

    with fsspec.open(TEST_INPUT_PATH, "w", compression="gzip") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    yield TEST_INPUT_PATH


@pytest.fixture(scope="module")
def finished_ids_path():
    lines = [
        {"id": "doc_1"},
        {"id": "doc_2"},
        {"id": "doc_3"},
    ]

    with fsspec.open(TEST_FINISHED_IDS_PATH, "w", compression="gzip") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    yield TEST_FINISHED_IDS_PATH


def test_find_all_finished_ids(finished_ids_path):
    finished_ids = find_all_finished_ids(
        TEST_CHECKPOINT_DIR,
        "jsonl.gz",
        "id",
    )
    assert finished_ids == {"doc_1", "doc_2", "doc_3"}


def test_ray_data_resumption(input_path, finished_ids_path):
    finished_ids = find_all_finished_ids(
        TEST_CHECKPOINT_DIR,
        "jsonl.gz",
        "id",
    )
    ds = ray.data.read_json(TEST_INPUT_PATH, arrow_open_stream_args={"compression": "gzip"}, override_num_blocks=1)
    ds = ds.filter(lambda x: x["id"] not in finished_ids)
    assert ds.count() == 2

    id_list = [x["id"] for x in ds.take_all()]
    assert id_list == ["doc_4", "doc_5"]
