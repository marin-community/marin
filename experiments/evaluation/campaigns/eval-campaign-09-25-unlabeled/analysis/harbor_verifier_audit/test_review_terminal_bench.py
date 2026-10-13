# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import statistics

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor.driver_config import HarborErrorTaxonomy
from rigging.filesystem import factory

from . import review_terminal_bench as reviewer
from .review_terminal_bench import Aggregate, failed_verifier_setup, remove_rewards


def test_verifier_failures_are_removed_from_reward_mean_and_sem():
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    original = Aggregate(len(rewards), statistics.mean(rewards), statistics.stdev(rewards) / math.sqrt(len(rewards)))
    result = remove_rewards(original, [0.0, 0.0, 0.0])
    valid = rewards[:-3]
    assert result.count == len(valid)
    assert result.mean == pytest.approx(statistics.mean(valid), abs=1e-12)
    assert result.sem == pytest.approx(statistics.stdev(valid) / math.sqrt(len(valid)), abs=1e-12)


def test_verifier_failure_does_not_turn_a_genuine_test_failure_into_infrastructure():
    bootstrap_error = "404 Not Found\ncurl: command not found\nuvx: command not found"
    assert failed_verifier_setup(bootstrap_error, "")
    assert not failed_verifier_setup(bootstrap_error + "\n1 failed, 5 passed", "")
    assert not failed_verifier_setup("1 failed, 5 passed", "")


class MemoryBucket(MemoryFileSystem):
    def split_path(self, path):
        bucket, _, key = path.lstrip("/").partition("/")
        return bucket, key, None

    def call_s3(self, method, **kwargs):
        assert method == "list_objects_v2"
        prefix = kwargs["Prefix"]
        parent, _, leaf = prefix.rpartition("/")
        paths = self.ls("/" + kwargs["Bucket"] + "/" + parent, detail=True)
        return {
            "CommonPrefixes": [
                {"Prefix": entry["name"].lstrip("/").partition("/")[2] + "/"}
                for entry in paths
                if entry["type"] == "directory" and entry["name"].rsplit("/", 1)[-1].startswith(leaf)
            ]
        }


@pytest.fixture
def qemu_bucket(monkeypatch, tmp_path):
    fs = MemoryBucket(skip_instance_cache=True)
    bucket = "review-" + tmp_path.name
    uri = "s3://" + bucket + "/run/results"
    monkeypatch.setattr(reviewer, "filesystem_for", lambda url: (fs, url.removeprefix("s3://")))
    monkeypatch.setattr(
        factory,
        "open_url",
        lambda url, mode="rb", **kwargs: fsspec.core.OpenFile(fs, url.removeprefix("s3://"), mode, **kwargs),
    )
    rewards = [1.0] * 80 + [0.0] * 163
    fs.pipe_file(
        bucket + "/run/results/harbor_result.json",
        json.dumps(
            {
                "scored_trials": 243,
                "mean_reward": statistics.mean(rewards),
                "reward_stderr": statistics.stdev(rewards) / math.sqrt(243),
            }
        ).encode(),
    )
    taxonomy = HarborErrorTaxonomy(
        infrastructure=frozenset({"TmuxSessionEndedError"}),
        agent=frozenset({"AgentTimeoutError"}),
        passthrough=frozenset(),
        undecided=frozenset(),
        commit="test",
    )
    return fs, uri, taxonomy


def write_qemu_results(fs, uri, outcomes):
    for index, outcome in enumerate(outcomes):
        path = uri.removeprefix("s3://") + f"/harbor_jobs/job/qemu-alpine-ssh__{index}"
        fs.makedirs(path, exist_ok=True)
        if outcome == "pending":
            continue
        verifier = None
        if outcome != "infra":
            verifier = {
                "rewards": {"reward": 0.0},
                "stdout": (
                    "1 failed, 5 passed"
                    if outcome == "tests"
                    else "404 Not Found\ncurl: command not found\nuvx: command not found"
                ),
                "stderr": "",
            }
        exception = "TmuxSessionEndedError" if outcome.startswith("infra") else "AgentTimeoutError"
        fs.pipe_file(
            path + "/result.json",
            json.dumps(
                {
                    "trial_name": f"qemu-alpine-ssh__{index}",
                    "task_checksum": "task",
                    "verifier_result": verifier,
                    "exception_info": {"exception_type": exception},
                }
            ).encode(),
        )


@pytest.mark.parametrize(
    "outcomes,valid_count",
    [
        (["bootstrap", "bootstrap", "infra"], 241),
        (["bootstrap", "tests", "infra"], 242),
        (["bootstrap", "bootstrap", "infra_bootstrap"], 241),
        (["tests", "tests", "tests"], 243),
    ],
)
def test_review_preserves_valid_trials_with_mixed_qemu_outcomes(qemu_bucket, outcomes, valid_count):
    fs, uri, taxonomy = qemu_bucket
    write_qemu_results(fs, uri, outcomes)
    result = reviewer.review("model", uri, {}, taxonomy)
    valid_rewards = [1.0] * 80 + [0.0] * (valid_count - 80)
    assert result["corrected"]["count"] == valid_count
    assert result["corrected"]["mean"] == pytest.approx(statistics.mean(valid_rewards), abs=1e-12)
    assert result["corrected"]["sem"] == pytest.approx(
        statistics.stdev(valid_rewards) / math.sqrt(valid_count), abs=1e-12
    )
    assert len(result["reviewed_trials"]) == 3
    assert result["sealable"]


def test_review_refuses_completion_when_a_qemu_repetition_has_no_result(qemu_bucket):
    fs, uri, taxonomy = qemu_bucket
    write_qemu_results(fs, uri, ["bootstrap", "bootstrap", "pending"])
    with pytest.raises(ValueError, match="all three QEMU repetitions"):
        reviewer.review("model", uri, {}, taxonomy)
