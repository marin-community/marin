# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Producer-row SQL shared by RL picker and recent-run queries."""

from pathlib import Path

RL_PRODUCER_ROW_NAMES_SQL = Path(__file__).with_suffix(".sql").read_text().strip()
