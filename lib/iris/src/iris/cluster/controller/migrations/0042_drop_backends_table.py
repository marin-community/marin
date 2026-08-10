# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``backends`` table.

``0033`` created it and seeded one row per backend, and nothing ever read it back:
the controller builds its backend registry from cluster config at startup, and the
meta-scheduler routes against what each backend advertises in memory. Its last
distinguishing column, ``allow_policy_json``, held the per-backend allow policy,
which is gone — admission belongs to the cluster a job lands on.

Idempotent, and it carries no data: ``DROP TABLE IF EXISTS`` no-ops on a fresh DB,
whose baseline never declares the table, and on a re-run after a crash. A legacy DB
replaying the deltas still creates and seeds the table in ``0033``; this drops it
again, landing on the same schema the baseline declares.
"""


def migrate(raw_conn) -> None:
    raw_conn.execute("DROP TABLE IF EXISTS backends")
