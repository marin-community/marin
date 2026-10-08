# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry point for ``python -m experiments.post_training.task_curation.images``.

The build module is never run as a script: an artifact recorded from ``__main__`` names
``__main__.ImageArtifact`` as its result type, which a driver importing the module cannot load.
"""

from experiments.post_training.task_curation.images.build import main

main()
