"""Read-only, authenticated vLLM completion transport for Pi teacher traces.

This transport is not a SkyRL weight-synchronized inference engine: it only
serves /tokenize and /chat/completions. Teacher model/revision are declared
provenance, not a remote weight attestation (vLLM exposes no revision hash).
"""

from typing import Any

import httpx
from rigging.secrets import resolve_secret_spec


class ExternalPiTeacherEngine:
    """Call a separately owned vLLM endpoint without routing through the student."""

    def __init__(self, *, endpoint: str, model: str, auth: str, max_model_len: int, timeout: float = 180):
        if not endpoint or not model or max_model_len <= 0 or timeout <= 0 or not auth.startswith("env:"):
            raise ValueError("External Pi teacher requires an endpoint, model, context limit and env-backed auth")
        self.max_model_len = max_model_len
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {resolve_secret_spec(auth).value}"},
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Fail loudly without including a scoped endpoint capability in errors."""
        try:
            response = await self._client.post(f"{self.endpoint}/{path}", json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ValueError(f"External Pi teacher {path} returned HTTP {error.response.status_code}") from None
        except httpx.RequestError as error:
            raise ValueError(f"External Pi teacher {path} transport failed: {type(error).__name__}") from None
        return response.json()

    async def tokenize(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Return exact IDs from the same remote model used to generate replies."""
        result = await self._post("tokenize", request_payload["json"])
        if not isinstance(result.get("tokens"), list) or any(type(token) is not int for token in result["tokens"]):
            raise ValueError("External Pi teacher /tokenize did not return exact integer token IDs")
        return result

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Require vLLM token evidence, not text-only OpenAI completions."""
        result = await self._post("chat/completions", request_payload["json"])
        if result.get("model") != self.model or len(result.get("choices", [])) != 1:
            raise ValueError("External Pi teacher response has the wrong model or completion count")
        choice = result["choices"][0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(type(token) is not int for token in token_ids):
            raise ValueError("External Pi teacher must supply sampled token IDs")
        scores = choice.get("logprobs")
        if scores is not None and (
            not isinstance(scores, dict)
            or not isinstance(scores.get("content"), list)
            or len(scores["content"]) != len(token_ids)
        ):
            raise ValueError("External Pi teacher returned incomplete sampled logprobs")
        return result

    async def close(self) -> None:
        """Release the HTTP transport without changing the external server."""
        await self._client.aclose()
