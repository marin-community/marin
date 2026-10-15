# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Shellsim image and declare its proxy using ambient provider credentials."""

import pulumi
from infrastructure import deploy, stack_config

deployment = deploy(stack_config())
pulumi.export("url", deployment.url)
pulumi.export("workerImage", deployment.worker_image)
