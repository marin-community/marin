import asyncio

from eval import robust_api  # noqa: F401 - installs the rolling-likelihood patch
from lm_eval.models.api_models import TemplateAPI


class _CacheHook:
    def __init__(self):
        self.values = {}

    def add_partial(self, request_type, key, value):
        self.values[(request_type, key)] = value


class _ConcurrentRollingAPI(TemplateAPI):
    def __init__(self):
        self._world_size = 1
        self._batch_size = 1
        self._concurrent = 3
        self.max_length = 32
        self.max_retries = 1
        self.timeout = 30
        self.verify_certificate = True
        self.tokenizer = object()
        self.cache_hook = _CacheHook()
        self.active_requests = 0
        self.max_active_requests = 0

    @property
    def prefix_token_id(self):
        return 0

    def tok_encode(self, string, **kwargs):
        return list(range(1, len(string) + 1))

    async def amodel_call(self, *, sem, messages, ctxlens, **kwargs):
        await sem.acquire()
        try:
            self.active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)
            await asyncio.sleep(0)  # Let the other scheduled endpoint calls start.
            self.active_requests -= 1
            return [
                (-float(len(token) - ctxlen), False)
                for token, ctxlen in zip(messages, ctxlens, strict=True)
            ]
        finally:
            sem.release()

    def parse_logprobs(self, *, outputs, tokens, ctxlens, **kwargs):
        return [(-float(len(token) - ctxlen), False) for token, ctxlen in zip(tokens, ctxlens, strict=True)]

    def _create_payload(self, *args, **kwargs):
        raise NotImplementedError

    def parse_generations(self, *args, **kwargs):
        raise NotImplementedError


def test_rolling_loglikelihood_batches_documents_concurrently():
    model = _ConcurrentRollingAPI()
    documents = ["one", "three", "seven", "nine", "eleven", "thirteen", "fifteen"]
    from eval.contracts.eval_spec import ModelRequest, RequestKind
    from eval.execution.backend import LMBackend
    from eval.runtime.pool import ExecutionPool
    requests = [ModelRequest(RequestKind.ROLLING_LIKELIHOOD, document) for document in documents]

    async def run():
        async with LMBackend(model) as backend:
            return sorted([completed async for completed in ExecutionPool(3).map(requests, backend.execute)],
                          key=lambda completed: completed.position)

    completed = asyncio.run(run())
    assert all(item.error is None for item in completed)
    assert [item.result for item in completed] == [-float(len(document)) for document in documents]
    assert model.max_active_requests == 3
