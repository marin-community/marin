# Corrected SFT cold-start agentic evaluations

These diagnostic profiles pin reproduction checkpoints 222, 471, 1024, and 2048, force thinking, and stop on both `128001` and `128009`. The earlier exports omitted `generation_config.json`; their tokenizer default also did not request thinking. Checkpoint revisions here include the metadata correction with unchanged weights.

The six benchmark profiles preserve the historical task identities, budgets, and attempt counts. Terminal-Bench is 2.1 revision 6 (89 tasks), pinned by its resolved content digest. SWE uses separate 131,072-context server profiles; the other benchmarks share 73,728-context servers. BixBench requires the pinned MiniMax judge profile. These configs are diagnostic comparisons, not a new policy release or an isolated EOS ablation.

Launch through `experiments.evaluation.cli` using these model/Harbor profiles and the existing evaluation campaign tooling. Use the `model-configs/swebench-verified` variant for SWE. Keep the runtime revision, model revision, source config digest, coverage, and verifier failures with each result. An unsuccessful verifier bootstrap must not be accepted as a model-scored zero.

The source profiles were exercised in the 2026-10-30 rerun at commit `2a174c5fff27ecbcf2db4a300441e247ea18ebbd`. Live step-222 smoke probes verified both EOS tokens and thinking on the served endpoints. Reproduction snapshots and broader six-model control configs are preserved separately on the reference branch.
