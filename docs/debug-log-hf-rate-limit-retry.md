# Debug Log: Hugging Face Rate-Limit Retry

## Context

The older parity rerun lineage failed in:

```text
/calvinxu/dm-qsplit240-300m-6b-parity-rerun-20260414-124603/evaluation-lm_evaluation_harness_levanter-lmeval_debug_run_00028_189770f1-31b038fa
```

The first root-cause line was a Hugging Face quota error:

```text
We had to rate limit you, you hit the quota of 2500 api requests per 5 minutes period.
```

The traceback showed this happened while loading `NousResearch/Llama-2-7b-hf` tokenizer files through `AutoTokenizer.from_pretrained(...)`.

## Root Cause

Levanter already had a retry helper for Hugging Face Hub 429s, but Transformers wrapped the underlying `HfHubHTTPError` as an `OSError`:

```text
OSError: There was a specific connection error when trying to load NousResearch/Llama-2-7b-hf:
429 Client Error: Too Many Requests
```

The retry classifier only recognized direct `HfHubHTTPError`, timeout, and connection-error exceptions, so it did not retry this wrapped 429.

## Fix

`_is_retryable_hf_exception(...)` now walks the exception chain and treats wrapped Hugging Face 429/rate-limit errors as retryable.

## Validation

Ran:

```bash
uv run --with pytest --with pytest-xdist --with pytest-timeout --with pytest-asyncio --with pytest-flakefinder python -m pytest lib/levanter/tests/test_hf_utils.py::test_wrapped_hf_429_is_retryable
```

Result: 1 passed.

