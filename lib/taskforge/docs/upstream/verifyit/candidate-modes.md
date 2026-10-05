# [verifyit] Grade extracted text candidates for every answer mode

`verifyit.candidate` grades an already extracted answer for only four modes: `exact`, `numeric`, `mcq`, and `predicted_action`. The other answer-file modes (`math`, `ifeval`, `json-schema`, `xml-elements`, `csv-columns`, `reasoning-gym`) can be graded only through `grade(spec, tests_dir, workspace)`, which reads the answer from a file in a workspace and resolves spec paths against a tests directory on disk. A caller that holds the candidate as text and keeps private verifier files in memory, such as a chat rollout with no machine, cannot use these modes.

Today, on `origin/main`:

- `lib/verifyit/src/verifyit/candidate.py`: `supports_candidate_mode` returns true for the four modes above. `candidate_spec` and `grade_text_candidate` dispatch only `ExactSpec | NumericSpec | McqSpec`.
- Direct graders already exist for some of the other modes: `grade_math.grade_math_candidate(spec, candidate)`, `grade_ifeval.grade_ifeval_candidate(spec, candidate)`, `grade_json_schema.grade_json_schema_candidate(schema, instance)`, and `grade_reasoning_gym.grade_reasoning_gym_candidate(spec, entry, candidate, params=...)`. `grade_xml` and `grade_csv` have only file-based `grade`.
- `JsonSchemaSpec.schema`, `ReasoningGymSpec.entry`, and `ReasoningGymSpec.params` are paths relative to the directory that holds `verifier.toml`.
- TaskCompendium stores a verifyit mode name and its parameters in `VerifierSpec(kind, parameters_json)`. It calls `candidate_spec(kind, parameters)` and `grade_text_candidate` (`lib/taskcompendium/src/taskcompendium/grading.py`), so it gains each mode added here without schema changes.

Proposed:

- Add `grade_xml_candidate(spec: XmlElementsSpec, candidate: str) -> Reward` and `grade_csv_candidate(spec: CsvColumnsSpec, candidate: str) -> Reward`, factored out of the file-based graders.
- Extend `candidate.py` to the six modes above. Path-valued fields resolve against caller-supplied file contents instead of a directory:

  ```python
  def candidate_spec(mode: str, parameters: dict[str, Any], *, files: Mapping[str, bytes] = {}) -> CandidateSpec: ...
  def grade_text_candidate(spec: TextSpec, candidate: str, *, files: Mapping[str, bytes] = {}) -> Reward: ...
  ```

  `files` maps the spec-relative path (for example `schema.json` or `entry.json`) to its bytes. A path missing from `files` raises `InvalidTask`.
- Keep `empty_output` behavior identical to the file-based graders.

Usage:

```python
spec = candidate_spec("json-schema", {"schema": "schema.json"}, files={"schema.json": schema_bytes})
reward = grade_text_candidate(spec, extracted_answer, files={"schema.json": schema_bytes})
```

TaskCompendium would pass its private verifier files as `files`.

Evidence: tasks in the construct-003 production run used `math`, `json-schema`, `xml-elements`, `csv-columns`, `ifeval`, and an instruction-constraint check for short structured answers (`experiments/post_training/capability_env_gen/docs/task_contract.md`, "Verification contract", on branch `mark/autoenv`). On `origin/main`, these tasks can be graded only by a shell verifier in a separate machine that runs the `verifyit` CLI.

Not in scope for verifyit: how an answer is extracted from a conversation (TaskCompendium submission conventions), and where private files are stored (TaskCompendium `TaskSpec`).
