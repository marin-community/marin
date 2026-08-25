# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worked example configs for the general SFT launcher (``experiments.sft.launcher``).

Each module exposes an ``SFTSpec`` named ``SPEC``. These are *instances* of the generic launcher,
not defaults baked into it — e.g. ``delphi_1e22`` supplies the Delphi v0 chat template + the
magpie(math-strong)/warmup mixture.
"""
