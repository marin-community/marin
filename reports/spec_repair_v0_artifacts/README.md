# Spec Repair v0 Full Artifact Bundle

This directory stores the large files that could not be committed at their
original paths because Marin's pre-commit large-file check rejects files over
500 KB.

The bundle is a split base64-encoded `tar.zst` archive. Restore it from the
repository root with:

```bash
bash reports/spec_repair_v0_artifacts/restore.sh
```

The archive contains:

- `.agents/logbooks/executable_specs_claude.md`
- `experiments/posttrain/disagreement_primitive/e8_rubrics_v1.jsonl`
- `experiments/posttrain/disagreement_primitive/grounding/`
- `experiments/posttrain/disagreement_primitive/repair_v0/`
- `results/raw/e9_compile_edit_round_1/`
- `results/raw/e9_compile_edit_round_2/`
- `results/raw/e9_qualifier_rubrics_round_1/`
- `results/raw/e9_verify_edit_round_1/`
- `results/raw/e9_verify_edit_round_2/`
- `claude_subagents/lm_judge_full_spec/`
- `claude_subagents/lm_judge_rubric/`
- `claude_subagents/lm_judge_rubric_plus_spec/`
- `claude_subagents/lm_judge_single_statement/`

