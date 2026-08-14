# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.network import interface_for_ipv4


def test_interface_for_ipv4_finds_loopback():
    assert interface_for_ipv4("127.0.0.1") == "lo"
