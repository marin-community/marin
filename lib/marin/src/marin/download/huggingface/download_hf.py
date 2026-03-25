# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Backward-compat shim. Canonical location: marin.datakit.download.huggingface

from marin.datakit.download.huggingface import DownloadConfig as DownloadConfig
from marin.datakit.download.huggingface import _relative_path_in_source as _relative_path_in_source
from marin.datakit.download.huggingface import download_hf as download_hf
from marin.datakit.download.huggingface import ensure_fsspec_path_writable as ensure_fsspec_path_writable
from marin.datakit.download.huggingface import main as main
from marin.datakit.download.huggingface import stream_file_to_fsspec as stream_file_to_fsspec

if __name__ == "__main__":
    main()
