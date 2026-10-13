# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declare the ports proxy using ambient provider credentials."""

import pulumi
from infrastructure import deploy, stack_config

pulumi.export("url", deploy(stack_config()))
