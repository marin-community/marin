# AlpacaEval grading support

This vendored subset retains the AlpacaEval dataset adapter, four judge configurations,
their prompts and completion parsers, pairwise annotation processors, and win-rate metrics.
Evalchemy executes candidate and judge requests through its shared evaluator.

The original standalone CLI, model configurations, API and local decoders, analysis tools,
and saved leaderboard weights are omitted. Annotation execution in parity checks accepts
an explicit completion callback.

See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff) for upstream attribution.
