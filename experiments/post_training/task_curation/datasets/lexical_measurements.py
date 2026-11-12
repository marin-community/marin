# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete frozen-population lexical measurements keyed by current catalog ID."""

from pathlib import Path

from experiments.post_training.task_curation.lexical_binding import read_lexical_measurements

LEXICAL_MEASUREMENTS = read_lexical_measurements(Path(__file__).with_name("lexical_measurement_receipts"))
