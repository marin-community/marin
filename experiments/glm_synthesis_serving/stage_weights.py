# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the pinned GLM-5.3-Flash weights into the pool's in-region prefix.

Run as an Iris CPU job on cw-rno2a. The cache helper resumes partial staging
and reuses a completed snapshot; the 306 GiB weights never pass through the Mac.
"""

import json
import logging

from levanter.model_cache import cache_hf_model

from experiments.glm_synthesis_serving.pool import MODEL_REPO, MODEL_REVISION, WEIGHTS_PREFIX

logger = logging.getLogger(__name__)


def stage() -> str:
    """Mirror MODEL_REPO at MODEL_REVISION into WEIGHTS_PREFIX; return the snapshot path."""
    path = cache_hf_model(WEIGHTS_PREFIX, MODEL_REPO, revision=MODEL_REVISION)
    logger.info("staged %s", json.dumps({"path": path, "revision": MODEL_REVISION}))
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[stage-weights] %(message)s")
    stage()
