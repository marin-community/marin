---
topic: moonep-jax
issue: https://github.com/marin-community/marin/issues/7891
description: MoonEP and global histogram quantile balancing on one A08 NVL72 rack
author: rav
---

# MoonEP JAX: Task Logbook

## Current TL;DR

The work starts from PR #7890 at `e38ae4f8`. The first gate requires correct gradients and zero drops for 25 EP64 steps. The final gate requires at least 21.7% median MFU.

## Scope

- Goal: Implement global histogram QB and MoonEP expert routing in Levanter.
- Primary metrics: Assignment drops, output and gradient parity, median MFU, and tokens per second.
- Constraints: Use one A08 NVL72 rack. Do not change cluster configuration. Keep one active rack request.
- Coordinating issue: https://github.com/marin-community/marin/issues/7891
- Baseline PR: https://github.com/marin-community/marin/pull/7890

## Baseline

- Date: 2026-08-02.
- Code reference: `e38ae4f8b2477d420575b7335676328b5dd88172`.
- MHEP-004: 25 steps, 24.1231% median MFU, and 9.6786% final drops.
- MHEP-008: 200 steps, 23.6969% median MFU, and 7.4113% final drops.
- Hardware: 16 A08 nodes with four GB200 GPUs on one NVL72 rack.
- Model: Batch 1024, sequence 4096, 256 experts, top-8, and EP64.

## References

- MoonEP: https://github.com/moonshotAI/moonep at `0f385f038fc33bec22e3bcf5a07a8a22693e754c`.
- Kimi K3 report: https://arxiv.org/abs/2607.24653.
- Local MoonEP clone: `.agents/tmp/moonep`.
- Local report: `.agents/tmp/moonep/moonep-paper.pdf`.
- Report SHA-256: `936a7a3b655947b014ba96b8a790c3cdb6ea8b37eea514e4b7b52655e20af0f8`.

## Hypothesis Queue

### Active

- `MNEP-H1`: A global 1,000-bin histogram estimates the pooled QB target within one bin width.
- `MNEP-H2`: The MoonEP allocation gives each EP64 rank exactly `S*K` assignments with no drops.
- `MNEP-H3`: Sparse remote expert copies and grouped GEMM reach at least 21.7% median MFU after profile-based changes.

### Blocked

- None.

### Falsified / Dead End

- Receiver-ECHO from #7279 reached 18.2099% median MFU and did not remove all drops.
- Three-choice spill from #7279 reached 24.0829% median MFU and left 5.8806% final drops.

### Promoted

- None.

## Experiment Matrix

| ID | Transport | QB | Gate |
| --- | --- | --- | --- |
| MNEP-001 | Fixed all-to-all | Global histogram | QB-only effect |
| MNEP-002 | MoonEP JAX | Local exact | Zero-drop MoonEP effect |
| MNEP-003 | MoonEP JAX | Global histogram | Combined correctness and MFU |

## Entry Log

### 2026-08-02 09:08 UTC - Research start

- Hypothesis: A portable JAX path can prove MoonEP semantics before native CUDA fabric work.
- Commit Hash: `e38ae4f8b2477d420575b7335676328b5dd88172`.
- Command: Source preparation and issue creation only.
- Config: PR #7890 EP64 hero configuration.
- Result: Created issue #7891. The MoonEP clone and report are present and verified.
- Interpretation: No open MoonEP issue or competing branch was found.
- Next action: Add the global histogram QB reference and behavior tests.

### 2026-08-02 09:19 UTC - Global histogram QB reference

- Hypothesis: A 1,000-bin pooled histogram removes the error from averaging per-rank quantiles.
- Commit Hash: `b3a4e0b65`.
- Command: `uv run pytest tests/test_moe_hero_ep.py tests/test_grug_variant_contracts.py -q`.
- Config: Global histogram range `[min(bias)-1, max(bias)+1]` with one integer reduction per layer.
- Result: Pooled quantile, local-average counterexample, and two-device reduction checks pass. Pyrefly reports zero errors.
- Interpretation: The implementation matches the report algorithm within one bin width and preserves the local estimator as a control.
- Next action: Implement the MoonEP allocation planner and compare it with the independent reference.
