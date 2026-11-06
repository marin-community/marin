# OpenRouter first-12 routing/solver diagnostic

**This is one task × 12 models, not a 12-respondent item-response matrix.** We
paused after this checkpoint, before expanding to other tasks or fitting IRT.
The common frozen v2 task is
`biotasks-read_counting_modes-78a95b29f211f753-v2`, one of the two
independently corrected references. Execution was ShellSim, with the frozen
OpenAI chat policy (4,096 tokens/turn; 32,768 cumulative; eight turns;
temperature 0; top_p 1). One response per model from its pinned provider.

Private raw trial artifacts (trajectories, grading, request IDs, per-generation
provider and price evidence) are under:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/combined/v2/calibration/
    openrouter-first12-one-item-diagnostic-v1/results.jsonl
    openrouter-first12-one-item-diagnostic-v1/recovery1.jsonl
```

- Initial 12-row artifact SHA-256:
  `a76cb60fac6c27215c2ae2511d79cf4fb02d87918dfc8ecbe201c1d67347a517`.
- Separately preserved one-row recovery SHA-256:
  `6b41fd8a40000ba338060564a19fdeb3a3e5148b4656cc6339c49f2466d9973e`.
- Panel manifest SHA-256:
  `30532853c029141bbd9d7f41cc4924c4e01eb88c93b8b0fb8052afa5bd3e6bea`.
- Iris jobs: `/zack/biotasks-openrouter-first12-one-item` and
  `/zack/biotasks-openrouter-first12-recovery1` (both succeeded).

The initial Llama provider (`google-vertex/us-east5`) was BYOK-only and
returned a 404 before generating; that infrastructure failure was not scored.
A separately recorded, pinned `digitalocean` recovery produced its one
response. All **12 resulting responses have generation-metadata provider
verification**. Total observed OpenRouter generation spend: **$0.34650**
(excludes any credit purchase fee and compute). There were 13 attempted pairs,
but only 12 model responses; preserve the failed route attempt as evidence.

On this *single* complex read-counting item: one exact correct; seven other
completed trials scored zero; four trials were ungraded because of length/turn
limits. Of the 12 attempts, three produced valid JSON (one correct, two wrong)
and nine had submission-format failures; some latter trials ended at a budget
limit. **Do not infer overall model ability or bank difficulty from one item.**
The fact that many capable models could not return a compliant answer suggests
we should inspect the prompt/submit contract and tool traces before scaling,
not simply treat all failures as inability to solve biology.

Next: review the 12 raw trajectories and format contract; separately validate
more frozen references; then decide whether to extend the same 12 provider-pinned
models to a vetted task cohort. Do not launch the 58 unresolved v2 items or
expand respondents for IRT until the item bank and runtime are audited.
