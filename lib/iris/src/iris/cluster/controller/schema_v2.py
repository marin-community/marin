# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy Core schema for the controller database.

Holds the shared ``MetaData`` registry that future stages will populate with
``Table`` declarations matching today's hand-rolled migrations.
"""

from sqlalchemy import MetaData

metadata = MetaData()
