# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locked Python fixture shared by interpreter tests and host-command fixtures."""

PYTHON_VERSION = "3.12.13"


LOCK = """\
six==1.17.0 \\
    --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274 \\
    --hash=sha256:ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81
"""
