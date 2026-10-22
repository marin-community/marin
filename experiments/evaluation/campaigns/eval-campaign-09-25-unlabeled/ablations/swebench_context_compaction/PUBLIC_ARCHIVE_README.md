---
pretty_name: SWE-bench Verified Context and Compaction Ablation Traces
tags:
  - evaluation
  - swe-bench
  - mini-swe-agent
---

# SWE-bench Verified context and compaction ablation

This public archive contains the accepted trial-level mini-swe-agent trajectories behind a
four-model, four-arm SWE-bench Verified ablation. Each compressed JSONL file contains one row
per accepted task, including the task name, reward, original result URI and SHA-256, execution
metadata, and all available mini-swe-agent attempt trajectories. The export omits the agent's
endpoint configuration because it embeds short-lived signed proxy credentials. Every exported
row is checked against the SHA-256 and reward in the sealed audit before publication, and the
serialized public row is scanned for credential and signed-endpoint patterns.

| Model | 32k/8k no summary (earlier tracker reference) | 65k/16k no summary | 131k/16k no summary | 32k/8k summary | 65k/16k summary |
|---|---:|---:|---:|---:|---:|
| Qwen 3.6 | 0.567 | 0.650 | 0.679 | 0.370 | 0.336 |
| Gemma 4 | 0.367 | 0.513 | 0.496 | 0.159 | 0.162 |
| Nemotron Lightning | 0.302 | 0.442 | 0.460 | 0.209 | 0.181 |
| Snowball v0.1 | 0.243 | 0.322 | 0.334 | 0.055 | 0.042 |

The 16 new arms each include 451–500 of 500 valid scored tasks. The four 32k/8k no-summary
values are earlier tracker references, not new ablation arms, and their traces are not included
here. Summarization was configured to retain 8,192 **characters** of accumulated serialized
history after the original system and task messages. The output caps were 8,192 or 16,384
tokens as shown. All arms used canonical SWE-bench Verified, mini-swe-agent 2.1.0, 1,800-second
agent and 300-second verifier timeouts. Early-sealed valid-task subsets differ, so differences
are descriptive rather than matched-task causal estimates.

`accepted-manifest.json` fixes the exact accepted task/result choices and their source SHA-256s.
`MANIFEST-*.json` records exported row counts, compressed file hashes, and missing-trajectory
counts. `completion-audit.json` maps scores to the original sealed audit archives. The
checked-in [exporter](https://github.com/marin-community/marin/blob/038f4839d0/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/swebench_context_compaction/publish_traces.py)
shows the scrubbing and source verification logic.

Each `traces/<model>/<arm>.jsonl.zst` file can be read with `zstdcat` or Python's `zstandard`
package. This archive is for evidence preservation; it does not alter the original benchmark
results or the release policy by itself.
