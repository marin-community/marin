# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the SA Core scaffolding landed in stage 1."""

import sqlalchemy
from iris.cluster.controller import db_v2, schema_v2


def test_schema_v2_metadata_is_sa_metadata():
    assert isinstance(schema_v2.metadata, sqlalchemy.MetaData)


def test_db_v2_tx_placeholder_exists():
    assert isinstance(db_v2.Tx, type)


def test_sqlalchemy_major_version_is_2():
    assert sqlalchemy.__version__.startswith("2.")
