# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.filesystem.s3_compat import fsspec_s3_conf


def test_fsspec_s3_conf_bounds_coreweave_requests():
    conf = fsspec_s3_conf("http://cwlota.com")

    assert conf["config_kwargs"] == {
        "connect_timeout": 30,
        "read_timeout": 120,
        "retries": {"max_attempts": 5, "mode": "standard"},
        "s3": {"addressing_style": "virtual"},
    }


def test_fsspec_s3_conf_bounds_non_virtual_host_requests():
    conf = fsspec_s3_conf("https://acct.r2.cloudflarestorage.com")

    assert conf["config_kwargs"] == {
        "connect_timeout": 30,
        "read_timeout": 120,
        "retries": {"max_attempts": 5, "mode": "standard"},
    }
