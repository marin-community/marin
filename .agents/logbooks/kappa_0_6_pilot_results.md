# Option A 0-6 pilot — analysis vs 1-5 baseline

Pilot statements: ['no_erotica_or_gore', 'no_topic_off_limits']
Conditions: bare-0-6 (var_A_0_6), phase_4-0-6 (rubric_plus_spec_0_6)
Judges: gpt-5.1, gemini-3-flash, claude-sonnet-4-6 (tool-use forced)

## Per-judge marginal distributions

### no_erotica_or_gore — bare

0-6:
```
  gpt      [0: 1  1: 0  2: 1  3: 0  4: 2  5:33  6:23]
  gemini   [0: 0  1: 0  2: 1  3: 0  4: 0  5:43  6:16]
  claude   [0: 0  1: 0  2: 0  3: 1  4: 1  5: 9  6:49]
```

1-5 (baseline):
```
  gpt      [1: 1  2: 0  3: 0  4:16  5:43]
  gemini   [1: 0  2: 0  3: 0  4: 0  5:60]
  claude   [1: 0  2: 0  3: 1  4: 3  5:55]
```

### no_erotica_or_gore — phase_4

0-6:
```
  gpt      [0: 2  1: 0  2:10  3: 2  4: 5  5:34  6: 7]
  gemini   [0: 0  1: 0  2: 0  3: 0  4: 1  5:45  6:14]
  claude   [0: 0  1: 0  2: 0  3: 1  4: 3  5:33  6:23]
```

1-5 (baseline):
```
  gpt      [1: 1  2: 6  3: 2  4:21  5:30]
  gemini   [1: 0  2: 1  3: 1  4: 4  5:52]
  claude   [1: 0  2: 1  3: 3  4:20  5:36]
```

### no_topic_off_limits — bare

0-6:
```
  gpt      [0: 0  1: 0  2: 5  3: 0  4:10  5:35  6:10]
  gemini   [0: 0  1: 0  2: 1  3: 0  4: 0  5:21  6:38]
  claude   [0: 0  1: 0  2: 0  3: 2  4: 5  5: 9  6:44]
```

1-5 (baseline):
```
  gpt      [1: 0  2: 3  3: 2  4:24  5:31]
  gemini   [1: 0  2: 0  3: 0  4: 0  5:60]
  claude   [1: 0  2: 1  3: 2  4:10  5:47]
```

### no_topic_off_limits — phase_4

0-6:
```
  gpt      [0: 0  1: 0  2: 5  3:14  4: 5  5:16  6:20]
  gemini   [0: 0  1: 1  2: 0  3: 0  4: 1  5:14  6:44]
  claude   [0: 0  1: 0  2: 0  3: 1  4: 8  5:15  6:36]
```

1-5 (baseline):
```
  gpt      [1: 0  2: 5  3: 4  4:38  5:13]
  gemini   [1: 1  2: 0  3: 1  4: 2  5:56]
  claude   [1: 0  2: 1  3: 6  4:11  5:42]
```

## Headline diagnostics — did Option A unstack the ceiling?

| statement | cond | trivial-clean (0-6: all in 5-6) | ceiling-watch (0-6: all = 6) | trivial-clean (1-5: all in 4-5) |
|---|---|--:|--:|--:|
| no_erotica_or_gore | bare | 91.7% | 16.7% | 98.3% |
| no_erotica_or_gore | phase_4 | 66.7% | 8.3% | 79.3% |
| no_topic_off_limits | bare | 73.3% | 16.7% | 91.7% |
| no_topic_off_limits | phase_4 | 58.3% | 28.3% | 78.3% |

## Agreement metrics — 0-6 vs 1-5 (same statements)

| statement | cond | n_06 | α_06 | k2_06 | k4_06 | n_15 | α_15 | k2_15 | k3_15 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| no_erotica_or_gore | bare | 60 | +0.435 | +0.322 | +0.137 | 59 | +0.199 | -0.006 | +0.244 |
| no_erotica_or_gore | phase_4 | 60 | +0.182 | -0.071 | +0.173 | 58 | +0.251 | -0.055 | +0.108 |
| no_topic_off_limits | bare | 60 | +0.366 | +0.138 | +0.136 | 60 | +0.202 | +0.233 | +0.289 |
| no_topic_off_limits | phase_4 | 60 | +0.195 | +0.138 | +0.248 | 60 | +0.290 | +0.405 | +0.218 |

## Δ within each scale (phase_4 − bare)

| statement | Δα (0-6) | Δk2 (0-6) | Δk4 (0-6) | Δα (1-5) | Δk2 (1-5) | Δk3 (1-5) |
|---|--:|--:|--:|--:|--:|--:|
| no_erotica_or_gore | -0.253 | -0.393 | +0.036 | +0.052 | -0.049 | -0.135 |
| no_topic_off_limits | -0.171 | -0.000 | +0.112 | +0.088 | +0.172 | -0.071 |

## Verdict criteria

Option A is working if:
- ceiling-watch (% all=6) is **low** (target < 20%) — 6 isn't the new ceiling
- trivial-clean (0-6: % all in 5-6) is **lower** than the 1-5 baseline trivial-clean (% all in 4-5)
- Δk4 (0-6) ≥ 0 on both pilot statements — the rubric is no longer introducing fine-grained disagreement

If ceiling-watch is high (≥ 50%): Option A failed; 6 became the new 5.
If trivial-clean (0-6) is similar to 1-5: anchors aren't differentiating cases enough.
If Δk4 (0-6) < 0: rubric still introduces disagreement on borderline cases — bigger problem than scale.
