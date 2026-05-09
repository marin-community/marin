# DART Run 4 — Iterative validation report

Generated: 2026-05-09T08:54:03.928860+00:00
Statements: 13

## Per-statement summary

| statement | rounds | α baseline | α final | Δα total | rubric edits | spec edits | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| ask_clarifying_questions | 1 | -0.083 | 0.533 | +0.616 | 1 | 1 | converged |
| assume_objective_pov | 2 | 0.276 | 0.505 | +0.228 | 4 | 1 | converged |
| avoid_abuse | 1 | -0.795 | 0.694 | +1.489 | 3 | 1 | converged |
| be_clear | 1 | 0.170 | 0.597 | +0.427 | 3 | 1 | converged |
| be_thorough_but_efficient | 1 | 0.097 | 0.618 | +0.521 | 3 | 0 | converged |
| comply_with_laws | 2 | -0.555 | -0.100 | +0.455 | 9 | 1 | stuck |
| do_not_lie | 1 | -0.367 | 0.713 | +1.080 | 0 | 1 | converged |
| formatting | 1 | 0.194 | 0.510 | +0.316 | 2 | 0 | converged |
| letter_and_spirit | 1 | 0.402 | 0.449 | +0.047 | 0 | 1 | stuck |
| no_topic_off_limits | 2 | -0.135 | 0.107 | +0.242 | 7 | 0 | stuck |
| prevent_imminent_harm | 1 | 0.815 | 0.418 | -0.397 | 3 | 2 | stuck |
| protect_privileged_messages | 1 | 0.522 | 0.530 | +0.008 | 0 | 1 | converged |
| refusal_style | 2 | 0.000 | 0.458 | +0.458 | 4 | 2 | stuck |


**Verdict distribution**: {'converged': 8, 'stuck': 5}

**Converged** (8 statements with α ≥ 0.5):
- `ask_clarifying_questions` (α 0.533, Δ+0.616, 1 rounds)
- `assume_objective_pov` (α 0.505, Δ+0.228, 2 rounds)
- `avoid_abuse` (α 0.694, Δ+1.489, 1 rounds)
- `be_clear` (α 0.597, Δ+0.427, 1 rounds)
- `be_thorough_but_efficient` (α 0.618, Δ+0.521, 1 rounds)
- `do_not_lie` (α 0.713, Δ+1.080, 1 rounds)
- `formatting` (α 0.510, Δ+0.316, 1 rounds)
- `protect_privileged_messages` (α 0.530, Δ+0.008, 1 rounds)

**Stuck** (5 statements — escalated for spec-author review):
- `comply_with_laws` (α -0.100, Δ+0.455)
- `letter_and_spirit` (α 0.449, Δ+0.047)
- `no_topic_off_limits` (α 0.107, Δ+0.242)
- `prevent_imminent_harm` (α 0.418, Δ-0.397)
- `refusal_style` (α 0.458, Δ+0.458)

## Per-round summary


### Round 1

Verdict counts: {'converged': 7, 'improving': 4, 'stuck': 2}

| statement | α before | α after | Δα | verdict | rubric edits adopted | spec edits adopted |
|---|--:|--:|--:|---|--:|--:|
| ask_clarifying_questions | -0.083 | 0.533 | +0.616 | converged | 1 | 1 |
| assume_objective_pov | 0.276 | 0.425 | +0.149 | improving | 0 | 1 |
| avoid_abuse | -0.795 | 0.694 | +1.489 | converged | 3 | 1 |
| be_clear | 0.170 | 0.597 | +0.427 | converged | 3 | 1 |
| be_thorough_but_efficient | 0.097 | 0.618 | +0.521 | converged | 3 | 0 |
| comply_with_laws | -0.555 | -0.133 | +0.422 | improving | 4 | 1 |
| do_not_lie | -0.367 | 0.713 | +1.080 | converged | 0 | 1 |
| formatting | 0.194 | 0.510 | +0.316 | converged | 2 | 0 |
| letter_and_spirit | 0.402 | 0.449 | +0.047 | stuck | 0 | 1 |
| no_topic_off_limits | -0.135 | 0.248 | +0.383 | improving | 3 | 0 |
| prevent_imminent_harm | 0.815 | 0.418 | -0.397 | stuck | 3 | 2 |
| protect_privileged_messages | 0.522 | 0.530 | +0.008 | converged | 0 | 1 |
| refusal_style | 0.000 | 0.446 | +0.446 | improving | 0 | 2 |

### Round 2

Verdict counts: {'converged': 1, 'stuck': 3}

| statement | α before | α after | Δα | verdict | rubric edits adopted | spec edits adopted |
|---|--:|--:|--:|---|--:|--:|
| assume_objective_pov | 0.425 | 0.505 | +0.079 | converged | 4 | 0 |
| comply_with_laws | -0.133 | -0.100 | +0.033 | stuck | 5 | 0 |
| no_topic_off_limits | 0.248 | 0.107 | -0.141 | stuck | 4 | 0 |
| refusal_style | 0.446 | 0.458 | +0.011 | stuck | 4 | 0 |