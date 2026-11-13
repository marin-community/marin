# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence geometry and eligible chromosomes for the fixed VEP protocol."""

AUTOSOMES = frozenset(str(i) for i in range(1, 23))
SEQUENCE_LENGTH = 255
VARIANT_INDEX = SEQUENCE_LENGTH // 2
VARIANT_POSITION = VARIANT_INDEX + 1
