# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from skyrl_train.host_memory import parse_status_kib

STATUS = "Name:\tpython\nVmHWM:\t  2097152 kB\nVmRSS:\t  1048576 kB\n"


def test_status_fields_parse_in_kib():
    assert parse_status_kib(STATUS, "VmRSS") == 1048576 and parse_status_kib(STATUS, "VmHWM") == 2097152
    with pytest.raises(ValueError, match="MemAvailable is missing"):
        parse_status_kib(STATUS, "MemAvailable")
