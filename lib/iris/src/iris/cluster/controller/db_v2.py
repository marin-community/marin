# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy-backed controller database wrapper.

This module is the future home of the SA ``Engine`` factory, the ``Tx``
wrapper, and the ``write_transaction`` / ``read_snapshot`` context managers.
"""


class Tx:
    # TODO(stage-4): populate with conn, execute, executemany, register, _fire_hooks.
    __slots__ = ()
