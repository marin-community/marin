# [verifyit] Return per-criterion checklist grades and support repeated judgments

The `judge` checklist rubric returns one fraction for all criteria. A caller cannot combine the judged criteria with other checks under different weights, and cannot reduce judge noise by asking more than once. A rubric that mixes deterministic checks with model-judged criteria has to rebuild the checklist prompt and parser outside verifyit to get these.

Today, on `origin/main`, in `lib/verifyit/src/verifyit/modes/grade_judge.py`:

- `_judge_checklist` asks one yes/no question per criterion and returns `scored(passed / len(results), criteria=[{"criterion", "passed", "reasoning"}, ...])`. All criteria have equal weight.
- Each criterion is judged once at `temperature=0.0`. `JudgeSpec` has no sample count or resolution rule.
- `grade_judge_candidate(spec, candidate, *, connection, runtime, context, gate_candidate)` is the public entry point for callers that already hold the candidate text.

Proposed:

- Add a public function that returns one `Reward` per criterion, in `spec.criteria` order, so a caller can pass them to an aggregation function:

  ```python
  def grade_checklist_criteria(
      spec: JudgeSpec, candidate: str, *, connection: JudgeConnection | None = None,
      runtime: JudgeRuntimeSource | None = JudgeRuntimeSource.ENVIRONMENT, context: str = "",
  ) -> tuple[Reward, ...]: ...
  ```

  `_judge_checklist` becomes the mean over this tuple, so current behavior does not change.
- Add repeated judgments to `JudgeSpec`:
  - `samples: int = 1`
  - `sample_resolution: str = "majority"`, with `"two_then_third"` as the other value: two judgments, and a third only when the first two disagree.

  Each criterion's `detail` keeps every sample verdict and the call count.
- An unresolved criterion (for example, a sample that ends in an infrastructure error) makes the whole grade unscored.

Usage: a task's private grading script calls `grade_checklist_criteria`, weights the criteria together with its own machine checks, and combines them with the weighted aggregation proposed in the companion issue.

Evidence:

- `experiments/post_training/capability_env_gen/capability_pipeline/composite_policy.py` and `docs/judge.md` on branch `mark/autoenv`. That pipeline implemented weighted judge criteria, critical criteria, and `two_then_third` consensus outside its judge client, because the client returned only a scalar.
- In the construct-003 production run, about 129 items were lost to disagreement between the admission check and the combined machine-plus-judge gate (run audit; not in the repository).

Not in scope for verifyit: which criteria a task has and their wording (the task), and ordinal rubric anchors, which a task can encode as binary threshold criteria.
