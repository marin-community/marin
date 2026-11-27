# Micro210

The sealed Micro210 evaluation binds a content-pinned policy to the native task
runtime. `eval.py` launches the canonical graph for a model binding; `policy.py`
holds the frozen protocol references; `inputs.py` derives the native task table;
`images.json` pins the container images provisioning runs.

## Judge policy

Micro210 uses a [Together serverless judge](../judge_config.py) and a
[content-pinned policy](policy.py). Scores are not directly comparable
to the 2026.11.18 MiniMax policy; see [#11818](https://github.com/marin-community/marin/pull/11818)
for the proposed paired re-score.

To read a past run's policy, pass the `policy_reference` from its
`canonical-evaluation.json` to `read_micro210_policy` in `policy.py`.
The sealed 434-row table runs through native TaskSpecs with per-task provisioning
and runtime judge access. Only the judged grading plans receive the judge
connection; exact-match and code-execution graders run without provider
credentials. See the
[native launch and smoke controls](../../../docs/references/rl-launching.md#micro210-native-evaluation)
for source pins and diagnostic row selection. A smoke row selection is recorded
in its native table's `binding.json`; diagnostic runs are not full Micro210
scores. Mini840 keeps its existing judge policy.
