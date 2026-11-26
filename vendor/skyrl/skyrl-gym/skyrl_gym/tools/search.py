"""A task-owned HTTP client for retrieval tools."""

import json
import logging
import time

import requests

logger = logging.getLogger(__name__)
DEFAULT_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
RETRYABLE_STATUSES = {500, 502, 503, 504}


def format_passages(retrieval_result: list[dict]) -> str:
    return "".join(
        f"Doc {index}: {item['document']['contents'].strip()}\n" for index, item in enumerate(retrieval_result, 1)
    )


class SearchClient:
    """Reuse one HTTP session for one task, then close it with the task session."""

    def __init__(self, search_url=DEFAULT_URL, topk=3, timeout=DEFAULT_TIMEOUT, log_requests=True):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        self.session = requests.Session()

    def _request(self, query: str) -> dict:
        for attempt in range(MAX_RETRIES):
            try:
                if self.log_requests:
                    logger.info("Search attempt %d: %s", attempt + 1, self.search_url)
                response = self.session.post(
                    self.search_url,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json={"query": query, "topk": self.topk, "return_scores": True},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                retryable = isinstance(error, (requests.ConnectionError, requests.Timeout)) or (
                    error.response is not None and error.response.status_code in RETRYABLE_STATUSES
                )
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                logger.warning("Search request failed: %s", error)
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
        raise RuntimeError("Search retries produced no response")

    def search(self, query: str | None) -> str:
        """Return retrieval output or an error observation in the source JSON format."""
        if query is None:
            return ""
        try:
            response = self._request(query.strip())
            results = response.get("result", [])
            output = (
                "\n---\n".join(format_passages(retrieval) for retrieval in results)
                if results
                else "No search results found."
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            output = f"Search error: {error}"
            logger.warning("%s", output)
        return json.dumps({"result": output})

    def close(self) -> None:
        self.session.close()
