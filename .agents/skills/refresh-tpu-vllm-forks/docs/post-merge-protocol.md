# Post-Marin-Merge Fork Main Promotion Protocol

Run this only after the Marin fork-refresh PR has merged and Marin `main`
contains the new exact SHA pins for both forks. Do not run it while preparing,
reviewing, or validating the Marin PR.

The goal is cosmetic but useful: keep each fork's `main` branch pointing at the
same commit Marin actually pins, so human readers see the active fork tip
without opening Marin's root `pyproject.toml`.

For clarity, both `vllm` and `tpu-inference` fork `main` branches should point
to the same commits pinned in Marin's root `tool.uv.sources`, and each fork
should keep linear history: Marin overlay commits over upstream commits, with no
merge commits.

For each fork:

- Read the target SHA from Marin `main` root `pyproject.toml`.
- Fetch the fork and record current `origin/main`.
- Verify the target SHA exists in that fork.
- Create or verify `main-backup/YYYYMMDD/pre-<old-main-shortsha>`.
- Update fork `main` with `--force-with-lease` from the old `origin/main` SHA
  to the target SHA.
- Verify remote `main` resolves to the target SHA.

If the backup branch already exists and points to the old fork `main` SHA, reuse
it. If it exists and points elsewhere, stop and inspect instead of guessing.

If one fork promotion succeeds and the other fails, Marin remains correct
because it pins exact SHAs. Do not roll back the successful fork automatically;
record the mismatch and ask whether to retry the failed fork promotion. If the
retry still fails, file a new `marin-community/marin` issue assigned to
`@yonromai` with the pinned SHAs, promoted fork, failed fork, backup branches,
commands attempted, and error output.
