# Fixed-Base Overlay Fork PR Protocol

Use this when adding or fixing a Marin overlay commit while keeping the current
upstream base fixed. Do not use it for automated release/LKG refreshes.

Because the base does not move, the new commit sits on top of the pin's current
tip and history stays linear. Unlike a rebase refresh, a fork PR that
fast-forwards the pin's stable branch is fine here. All maintained sources use
`main`, including vLLM for both GPU and TPU wheels.

1. Branch from the pin's stable `branch`, add the overlay commit(s), and open a fork
   PR against that branch.
2. If Marin validation is needed, open a Marin draft PR that pins the fork PR head SHA
   as `commit` in `config/external/vllm/tpu.toml` and regenerates
   `external_dependencies.py` (`uv run config/update-external.py`). Treat this pin as
   temporary.
3. Run the required Marin validation from the draft PR.
4. Merge the fork PR into `main`, fetch it, and read the landed SHA. Do
   not assume it matches the pre-merge PR head SHA.
5. Update the Marin draft PR to pin the landed `main` SHA in `tpu.toml`,
   regenerate the pins and fix `upstream_base` if needed, rerun focused validation,
   then undraft.

Final check for tpu-inference: `git ls-remote <fork-url> main` must match its
`config/external/vllm/tpu.toml` commit. A vLLM pin may remain at an older exact
main-line source; prove it is an ancestor of `origin/main` or is named by the
recorded immutable source tag. A vLLM source edit also invalidates the GPU
artifact and TPU pair, so run both device-specific release gates before moving
their consumer pins.
