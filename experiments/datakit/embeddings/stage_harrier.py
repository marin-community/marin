# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the pinned Harrier model for the 50M-document embedding job."""

import json
import logging

from experiments.datakit.embeddings.harrier import stage_model

logger = logging.getLogger(__name__)


def main() -> None:
    result = stage_model()
    logger.info("HARRIER_MODEL=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
