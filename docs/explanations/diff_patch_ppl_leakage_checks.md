# Diff/Patch PPL Leakage Checks

Issue: #5095

This checklist applies to the diff/patch raw eval slices in `experiments/exp5095_diff_patch_ppl.py`.

1. Remove provenance-only identifiers before linearizing eval text.
For SWE-bench slices, drop `instance_id`, `repo`, `base_commit`, `version`, `FAIL_TO_PASS`, and `PASS_TO_PASS`.
For CommitPack slices, drop `repo_name`, `commit_hash`, and `url`.

2. Keep patch-only and context-plus-patch metrics separate.
Patch-only slices (`*_patch_text`) measure code edit modeling directly.
Context-plus-patch slices (`*_context_plus_patch`) measure issue/commit-message conditioning plus patch quality.

3. Run train/eval overlap checks before publishing metrics.
Use exact and normalized hashes on patch bodies and context text separately.
Fail the run if any eval row shares a hash with training rows after normalization.

4. Keep eval source snapshots immutable.
Pin source revisions (dataset snapshot or commit hash) in the build logs.
Do not regenerate eval rows from moving HEAD references.
