# Fixed-Base Overlay Fork PR Protocol

Use this when adding or fixing a Marin overlay commit while keeping the current
upstream base fixed. Do not use it for automated release/LKG refreshes.

1. Branch from the affected fork's current `main`, add the overlay commit(s), and
   open a fork PR.
2. If Marin validation is needed, open a Marin draft PR that pins the fork PR
   head SHA as `commit` in `config/external/vllm/tpu-forks.toml` and regenerates
   `external_dependencies.py` (`uv run config/update-external.py`). Treat this
   pin as temporary.
3. Run the required Marin validation from the draft PR.
4. Squash-merge the fork PR, fetch fork `main`, and read the landed squash SHA.
   Do not assume it matches the pre-merge PR head SHA.
5. Update the Marin draft PR to pin the landed fork `main` SHA in
   `tpu-forks.toml`, regenerate the pins and fix `upstream_base` if needed,
   rerun focused validation, then undraft.

Final check: `git ls-remote <fork-url> main` must match the relevant
`config/external/vllm/tpu-forks.toml` `commit`. Repeat per fork when an overlay
spans both forks.
