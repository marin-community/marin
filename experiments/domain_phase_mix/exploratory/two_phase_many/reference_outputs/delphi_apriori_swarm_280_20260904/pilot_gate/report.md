# Delphi a-priori swarm pilot gate

- Decision: **stop_after_pilot**
- Table-9 targets passing: **0/4** (continuation requires at least 2)
- Table-9 pooled seed SD: `0.008532` BPB; threshold: `0.017064` BPB
- Uncheatable pooled seed SD: `0.000636` BPB; reported but not decision-making

| Target | Anchor | Table-9 half mean Δ | Table-9 quarter Δ blocks | Quarter mean Δ | >2σ | Same sign | Dose monotone | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cc_high_group` | A | -0.002551 | -0.006835, -0.002955 | -0.004895 | False | True | True | False |
| `cc_high_group` | B | -0.000388 | +0.004610, -0.010167 | -0.002778 | False | False | True | False |
| `dolma3_stack_edu` | A | -0.004693 | -0.002761, -0.003421 | -0.003091 | False | True | False | False |
| `dolma3_stack_edu` | B | -0.004191 | +0.011285, -0.001512 | +0.004886 | False | False | True | False |
| `dolmino_olmocr_pdfs_hq` | A | -0.002367 | +0.000661, -0.004844 | -0.002091 | False | False | False | False |
| `dolmino_olmocr_pdfs_hq` | B | +0.004198 | +0.001865, -0.001114 | +0.000375 | False | False | False | False |
| `dolmino_synth_qa` | A | -0.004504 | -0.004232, -0.004493 | -0.004363 | False | True | False | False |
| `dolmino_synth_qa` | B | +0.002390 | +0.009720, +0.000502 | +0.005111 | False | True | True | False |

Table 9 is the preregistered decision metric. Uncheatable contrasts are in `gate_anchor_summary.csv`.
