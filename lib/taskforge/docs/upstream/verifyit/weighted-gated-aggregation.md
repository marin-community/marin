# [verifyit] Add weighted and gated reward aggregation

`aggregate_rewards` combines component grades with `ALL`, `MEAN`, `MAX`, `MIN`, or `PRODUCT`, and every component has equal weight. A rubric often needs three kinds of components: gates that must pass or the reward is 0, weighted criteria, and weighted penalties. `MEAN` lets a failed gate reduce the reward without zeroing it. `ALL` drops partial credit. A caller that needs gates and weights must write its own aggregation and repeat the status rules that `aggregate_rewards` already enforces.

Today, on `origin/main`, in `lib/verifyit/src/verifyit/grade.py`:

- `aggregate_rewards(verdicts, *, expected_total, policy, round_digits=None) -> Reward`. A missing component earns 0. Any `infra_error` or `invalid_task` component discards all credit, and `infra_error` takes precedence. Detail holds `passed`, `total`, and `missing`.
- `aggregate_first_fit` handles one-to-one assignment.
- No weights, gates, or penalties exist. The `judge` mode has `constraints` (IFEval checks that must pass before the model is called), which is a gate only for IFEval checks.

Proposed:

```python
class ComponentRole(StrEnum):
    GATE = "gate"          # must score 1.0, otherwise the total is 0
    CRITERION = "criterion"
    PENALTY = "penalty"    # subtracts weight * reward

@dataclass(frozen=True)
class Component:
    name: str
    role: ComponentRole
    weight: float = 1.0    # finite and positive; ignored for gates

def aggregate_weighted(
    components: Sequence[Component], verdicts: Mapping[str, Reward], *, round_digits: int | None = None
) -> Reward: ...
```

- Status rules match `aggregate_rewards`: `infra_error` first, then `invalid_task`, then a missing component counted as 0.
- If any gate scores below 1.0, the result is `scored(0.0, failed_gates=[...])`, and components after that gate do not need to be graded.
- Otherwise the reward is `max(0, sum(weight * reward for criteria) - sum(weight * reward for penalties)) / sum(criterion weights)`, clamped to [0, 1].
- `detail` holds each component's reward, `failed_gates`, `positive_sum`, `penalty_sum`, and `denominator`.
- Invalid weights, duplicate names, or no criteria return `invalid_task`.
- A helper `gates_passed(components, verdicts) -> bool` lets a caller skip judge calls after a failed gate.

Usage: a task's private grading script runs its own deterministic checks, each returning a `Reward`. If `gates_passed` is true, it calls `grade_checklist_criteria` from the companion judge issue, aggregates everything with `aggregate_weighted`, and writes the result with `write_reward`.

Evidence:

- `experiments/post_training/capability_env_gen/capability_pipeline/composite_policy.py` on branch `mark/autoenv`: a 552-line aggregation contract with gates, weighted criteria, penalties, critical criteria, and caps, written outside verifyit because `aggregate_rewards` has no weights.
- In the construct-003 production run, about 129 items were lost to disagreement between the admission check and that combined gate (run audit; not in the repository).

Not in scope for verifyit: the checks a task runs and its weights (the task's own scripts), and conditional caps tied to a specific check, which a task script can apply before aggregation. A TOML `mode = "composite"` that nests child specs can follow if callers need it from the CLI.
