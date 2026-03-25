# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Backward-compat shim. Canonical location: marin.datakit.download.ar5iv

from marin.datakit.download.ar5iv import Ar5ivDownloadConfig as DownloadConfig  # noqa: F401 - used by tests
from marin.datakit.download.ar5iv import download as download
from marin.datakit.download.ar5iv import process_shard as process_shard
