# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


logger = setup_logger(__name__)
