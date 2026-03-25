# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Backward-compat shim. Canonical location: marin.datakit.download.uncheatable_eval

from marin.datakit.download.uncheatable_eval import UncheatableEvalDataset as UncheatableEvalDataset
from marin.datakit.download.uncheatable_eval import (
    UncheatableEvalDownloadConfig as UncheatableEvalDownloadConfig,
)
from marin.datakit.download.uncheatable_eval import (
    download_latest_uncheatable_eval as download_latest_uncheatable_eval,
)
from marin.datakit.download.uncheatable_eval import make_uncheatable_eval_step as make_uncheatable_eval_step
