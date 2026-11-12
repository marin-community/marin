# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete actor censuses bound to exact input pins and converter versions."""

from pathlib import Path

from experiments.post_training.task_curation.census_binding import read_census_receipts

ACTOR_CENSUS_RECEIPTS = read_census_receipts(Path(__file__).with_name("daytona_census_receipts"))
