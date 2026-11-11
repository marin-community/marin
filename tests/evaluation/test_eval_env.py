# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.eval_env import EVAL_ENV_KEYS, env_vars_from_keys


def test_eval_jobs_receive_explicit_data_and_temp_prefixes(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-rno2a/marin")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", "s3://marin-rno2a")

    forwarded = env_vars_from_keys(EVAL_ENV_KEYS)

    assert forwarded["MARIN_PREFIX"] == "s3://marin-rno2a/marin"
    assert forwarded["MARIN_TEMP_PREFIX"] == "s3://marin-rno2a"
