# Generator-diversity analysis: existing-3 vs +Grok-opposite

Loaded 1280 cells across (sid × cond × scen × gen).

## Score distribution per generator under bare condition (variant_A, 1-5)

Per-judge marginal distribution. We expect Grok-opposite to push mass left.

### Condition: bare


**gpt-5.1**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         0   11    0   35  114    160  4.58    6.9%
  gemini      2   11    2    4  141    160  4.69    9.4%
  claude      1    6   10   16  126    159  4.64   10.7%
```

**Qwen/Qwen2.5-7B-Instruct-Turbo**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         3   25   14   58   60    160  3.92   26.2%
  gemini      2   16    5   10  127    160  4.53   14.4%
  claude      5   16   32   41   66    160  3.92   33.1%
```

**gemini-3-flash-preview**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         0    6    2   33  119    160  4.66    5.0%
  gemini      0    3    2    2  153    160  4.91    3.1%
  claude      1    1    5   16  136    159  4.79    4.4%
```

**grok-4-1-fast-non-reasoning-opposite**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         8   23    6   39   84    160  4.05   23.1%
  gemini     11    7    1    6  135    160  4.54   11.9%
  claude      4   12    7   21  116    160  4.46   14.4%
```

### Condition: phase_4


**gpt-5.1**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         6   17    1   41   95    160  4.26   15.0%
  gemini     11    7    0    2  140    160  4.58   11.2%
  claude      2    5   11   25  117    160  4.56   11.2%
```

**Qwen/Qwen2.5-7B-Instruct-Turbo**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         2   32   20   55   51    160  3.76   33.8%
  gemini     18    9   12   14  105    158  4.13   24.7%
  claude      6   20   34   49   51    160  3.74   37.5%
```

**gemini-3-flash-preview**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt         1   14    1   37  107    160  4.47   10.0%
  gemini      7    2    0    1  149    159  4.78    5.7%
  claude      0    1    6   18  135    160  4.79    4.4%
```

**grok-4-1-fast-non-reasoning-opposite**
```
  judge       1    2    3    4    5  total  mean   %≤3
  gpt        13   25    3   43   76    160  3.90   25.6%
  gemini     21    5    0   10  124    160  4.32   16.2%
  claude      4   19   10   19  108    160  4.30   20.6%
```


## Trivial-clean (all 3 judges in {4,5}) per generator

| condition | gpt-5.1 | Qwen | gemini | grok-opposite |
|---|--:|--:|--:|--:|
| bare | 80.5% (n=159) | 60.6% (n=160) | 89.9% (n=159) | 73.8% (n=160) |
| phase_4 | 76.2% (n=160) | 50.6% (n=158) | 87.4% (n=159) | 70.0% (n=160) |

## Population κ across the 8 statements — existing-3 generators vs +Grok-opposite

| condition | metric | existing-3 only | all-4 (with Grok) | Δ |
|---|---|--:|--:|--:|
| bare | alpha | +0.512 (n=478) | +0.564 (n=638) | +0.052 |
| bare | k3    | +0.309 (n=478) | +0.400 (n=638) | +0.091 |
| bare | k2    | +0.295 (n=478) | +0.437 (n=638) | +0.142 |
| phase_4 | alpha | +0.548 (n=477) | +0.580 (n=637) | +0.032 |
| phase_4 | k3    | +0.414 (n=477) | +0.463 (n=637) | +0.049 |
| phase_4 | k2    | +0.437 (n=477) | +0.498 (n=637) | +0.061 |

## Agreement on Grok-opposite responses ALONE (does the new generator carry signal?)

| condition | n | α | k3 | k2 |
|---|--:|--:|--:|--:|
| bare | 160 | +0.655 | +0.610 | +0.680 |
| phase_4 | 160 | +0.642 | +0.583 | +0.607 |

## Per-statement α — existing-3 vs +Grok-opposite

| statement | α_bare (3-gen) | α_bare (4-gen) | Δ | α_p4 (3-gen) | α_p4 (4-gen) | Δ |
|---|--:|--:|--:|--:|--:|--:|
| do_not_make_unprompted_personal_comments | +0.088 | +0.926 | +0.838 | +0.409 | +0.903 | +0.495 |
| be_professional | +0.632 | +0.632 | +0.001 | +0.785 | +0.740 | -0.046 |
| no_erotica_or_gore | +0.199 | +0.455 | +0.256 | +0.251 | +0.552 | +0.301 |
| present_perspectives | +0.648 | +0.558 | -0.089 | +0.749 | +0.643 | -0.106 |
| be_thorough_but_efficient | +0.405 | +0.419 | +0.014 | +0.475 | +0.491 | +0.016 |
| avoid_hateful_content | +0.343 | +0.574 | +0.232 | +0.407 | +0.518 | +0.112 |
| no_topic_off_limits | +0.202 | +0.168 | -0.035 | +0.290 | +0.264 | -0.026 |
| be_clear | +0.314 | +0.240 | -0.074 | +0.346 | +0.300 | -0.047 |
