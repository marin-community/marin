# Vendored swarm-run data

Provenance for the files in this directory. Both are extracted from the swarm branch snapshot
tagged `swarm-branch` (commit `bf26b666a97690b9dfcdd702aea5a513b40fad8f`, branch
`calvin/swarm-olmo3-regmix-test`).

- `two_phase_many.csv` — byte-identical copy of
  `experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many.csv` at that commit
  (`git show swarm-branch:experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many.csv`).
  Canonical run table for the qsplit240 60M/1.2B swarm: 241 completed rows (238
  `ngd3dm2_qsplit240`, 2 `ngd3dm2_hybrid_canary`, 1 `ngd3dm2_olmix_bpb`), 39 domains x 2 phases of
  mixture weights, and summary eval metrics. This CSV is the label source of truth for the 60M
  scale.
- `domain_token_counts.json` — `TOP_LEVEL_DOMAIN_TOKEN_COUNTS` (plus partition counts and totals)
  evaluated from
  `git show swarm-branch:experiments/domain_phase_mix/dolma3_dolmino_top_level_domains.py` at the
  same commit: per-domain available token counts for the 39 swarm domains.
