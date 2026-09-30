"""Harbor verifier used solely to retain prompt-only reset trial artifacts."""

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier


class PromptCaptureVerifier(BaseVerifier):
    @staticmethod
    def name() -> str:
        return "prompt-capture"

    async def verify(self) -> VerifierResult:
        return VerifierResult(rewards={"reward": 0.0})
