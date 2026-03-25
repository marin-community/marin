# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Backward-compat shim. Canonical location: marin.datakit.download.upload_gcs_to_hf

from marin.datakit.download.upload_gcs_to_hf import UploadConfig as UploadConfig
from marin.datakit.download.upload_gcs_to_hf import main as main
from marin.datakit.download.upload_gcs_to_hf import upload_gcs_to_hf as upload_gcs_to_hf

if __name__ == "__main__":
    main()
