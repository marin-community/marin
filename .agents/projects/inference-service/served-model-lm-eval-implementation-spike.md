# Served-Model LM Eval Implementation Spike

Use this as the sub-agent brief for testing [`01-rfc-served-model-lm-eval.md`](./01-rfc-served-model-lm-eval.md). Assume the RFC is the source of truth.

## Goal

Implement the smallest credible RFC 1 vertical slice, specifically to find whether the design works in code.

## Setup

- Pull latest `main` in `/Users/romain/dev/marin`.
- Create a fresh worktree under `/Users/romain/dev/marin-wt`.
- Do all implementation work in that new worktree.
- Create one local `report.md` at the new worktree root.
- Do not push or open a PR.

## Files To Inspect First

- `.agents/projects/inference-service/01-rfc-served-model-lm-eval.md`
- `.agents/projects/inference-service/02-rfc-levanter-http-parity.md`
- `.agents/projects/inference-service/03-rfc-iris-inference-service.md`
- `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`
- `lib/marin/src/marin/inference/vllm_server.py`
- `tests/evals/test_lm_eval.py`

## Scope

- Keep the write surface narrow.
- Implement only the RFC 1 served-model + lm-eval path.
- Add focused tests, including a deterministic fake HTTP server if useful.
- Make sure the new focused tests pass.
- Do not migrate existing eval call sites.
- Do not delete legacy eval code.
- Do not implement Harbor, Evalchemy, raw PPL, long-tail PPL, FineWeb2, perplexity-gap, Levanter parity, or Iris.
- Do not make normal CI require real vLLM.

## Final Report

Return:

- what works end-to-end
- what remains unproven
- biggest RFC holes, ordered by severity, especially around lm-eval adapter args, tokenizer handling, prompt-logprob response shape, result layout, and Iris compatibility
- concrete RFC edits recommended before implementation PR work continues

Put the final report in the local `report.md`.
