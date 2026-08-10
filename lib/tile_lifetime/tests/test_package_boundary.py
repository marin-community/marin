# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tile_lifetime


def test_tile_lifetime_root_does_not_reexport_moved_shuttle_types() -> None:
    assert not hasattr(tile_lifetime, "DType")
