# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for an Iris-registered TEI endpoint pool."""

import concurrent.futures
import json
import random
import time
import urllib.error
import urllib.request

import numpy as np
from iris.client import iris_ctx
from rigging.timing import Deadline, ExponentialBackoff

from experiments.datakit.embeddings.harrier.config import TEI_REQUEST_BATCH_SIZE

REQUEST_CONCURRENCY = 16
REQUEST_MAX_ATTEMPTS = 32
ENDPOINT_READY_TIMEOUT = 600
ENDPOINT_POLL_DELAY = 2
RETRYABLE_HTTP_CODES = frozenset({429, 502, 503, 504})


class TeiEmbeddingClient:
    """Submit embedding batches across an Iris endpoint pool."""

    def __init__(self, endpoint_name: str, embedding_dim: int) -> None:
        self.endpoint_name = endpoint_name
        self.embedding_dim = embedding_dim

    def _endpoint_urls(self) -> list[str]:
        deadline = Deadline.from_seconds(ENDPOINT_READY_TIMEOUT)
        while True:
            endpoint_urls = [endpoint.url for endpoint in iris_ctx().resolver.resolve(self.endpoint_name).endpoints]
            if endpoint_urls:
                random.shuffle(endpoint_urls)
                return endpoint_urls
            deadline.raise_if_expired(f"Timed out waiting for TEI endpoint {self.endpoint_name}")
            time.sleep(ENDPOINT_POLL_DELAY)

    def _request(self, endpoint_urls: list[str], endpoint_index: int, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"inputs": texts, "normalize": True, "truncate": True}, ensure_ascii=False).encode()
        backoff = ExponentialBackoff(initial=0.1, maximum=5, factor=2)
        last_error: Exception | None = None
        for attempt in range(REQUEST_MAX_ATTEMPTS):
            endpoint_url = endpoint_urls[(endpoint_index + attempt) % len(endpoint_urls)]
            request = urllib.request.Request(
                f"{endpoint_url.rstrip('/')}/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    embeddings = json.load(response)
                if len(embeddings) != len(texts):
                    raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                return embeddings
            except urllib.error.HTTPError as error:
                if error.code == 413 and len(texts) > 1:
                    middle = len(texts) // 2
                    return self._request(endpoint_urls, endpoint_index, texts[:middle]) + self._request(
                        endpoint_urls, endpoint_index + 1, texts[middle:]
                    )
                if error.code not in RETRYABLE_HTTP_CODES:
                    raise
                last_error = error
            except urllib.error.URLError as error:
                last_error = error

            if attempt + 1 < REQUEST_MAX_ATTEMPTS:
                time.sleep(backoff.next_interval())
                endpoint_urls = self._endpoint_urls()

        assert last_error is not None
        raise last_error

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts while preserving input order."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        endpoint_urls = self._endpoint_urls()
        batches = [
            texts[start : start + TEI_REQUEST_BATCH_SIZE] for start in range(0, len(texts), TEI_REQUEST_BATCH_SIZE)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(REQUEST_CONCURRENCY, len(batches))) as executor:
            futures = [
                executor.submit(self._request, endpoint_urls, index, batch) for index, batch in enumerate(batches)
            ]
            embeddings = [embedding for future in futures for embedding in future.result()]

        result = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (len(texts), self.embedding_dim)
        if result.shape != expected_shape:
            raise ValueError(f"Expected embeddings with shape {expected_shape}, got {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError("TEI returned non-finite embeddings")
        return result
