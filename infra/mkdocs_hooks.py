# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority

_MKDOCSTRINGS_LOGGER = "mkdocs.plugins.mkdocstrings"
_EXTERNAL_INVENTORY_LOAD_ERROR = "mkdocstrings: Couldn't load inventory http"


def _lower_external_inventory_failure(record: logging.LogRecord) -> bool:
    if record.name == _MKDOCSTRINGS_LOGGER and record.getMessage().startswith(_EXTERNAL_INVENTORY_LOAD_ERROR):
        record.levelno = logging.INFO
        record.levelname = logging.getLevelName(logging.INFO)
    return True


@event_priority(100)
def on_config(_config: MkDocsConfig) -> None:
    """Keep external inventory failures out of MkDocs strict counts."""
    logging.getLogger(_MKDOCSTRINGS_LOGGER).addFilter(_lower_external_inventory_failure)
