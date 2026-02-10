# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bundle storage utilities for the controller."""

import hashlib
import logging

import fsspec
from connectrpc.code import Code
from connectrpc.errors import ConnectError

logger = logging.getLogger(__name__)


class BundleStore:
    """Manages bundle storage.

    Args:
        bundle_prefix: URI prefix for storing bundles (e.g., gs://bucket/path or file:///path).
                      Uses fsspec for storage.
    """

    def __init__(self, bundle_prefix: str):
        self._prefix = bundle_prefix.rstrip("/")

    @property
    def prefix(self) -> str:
        return self._prefix

    def write_bundle(self, job_id: str, blob: bytes) -> str:
        """Write bundle blob to storage.

        Args:
            job_id: Job identifier used to construct the bundle path
            blob: Bundle data to write

        Returns:
            Full URI path where bundle was stored

        Raises:
            ConnectError: If bundle storage fails
        """
        bundle_hash = hashlib.sha256(blob).hexdigest()
        bundle_path = f"{self._prefix}/{bundle_hash}/bundle.zip"
        try:
            # Create parent directory if needed (use same filesystem instance)
            fs, path = fsspec.core.url_to_fs(bundle_path)
            parent_dir = path.rsplit("/", 1)[0]
            fs.makedirs(parent_dir, exist_ok=True)

            with fs.open(path + ".tmp", "wb") as f:
                f.write(blob)
            fs.rename(path + ".tmp", path)
            logger.info("Uploaded bundle for job %s to %s (%d bytes)", job_id, bundle_path, len(blob))
            return bundle_path
        except Exception as e:
            raise ConnectError(Code.INTERNAL, f"Failed to store bundle: {e}") from e
