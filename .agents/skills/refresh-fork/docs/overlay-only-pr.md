# Fixed-Base Overlay Fork PR Protocol

Use this when adding or fixing a Marin overlay commit while keeping the current
upstream base fixed. Do not use it for automated release/LKG refreshes.

Because the base does not move, the new commit sits on top of the pin's current tip
and history stays linear — so unlike a rebase refresh, a fork PR that fast-forwards
the pin's stable branch is fine here. Work on the descriptor's `branch` for the pin: on
the vllm fork the GPU pin builds from `main` and the TPU pin lives on `tpu`; a single-pin
fork (tpu-inference) uses `main`.

1. Branch from the pin's stable `branch`, add the overlay commit(s), and open a fork
   PR against that branch.
2. If Marin validation is needed, open a Marin draft PR that selects the fork PR head
   in `config/external/vllm/tpu.toml`. Follow the
   [paired-release procedure](../SKILL.md#pin-at-the-staged-tip) to build, qualify, and
   pin a wheel candidate from that exact source pair. Treat both selections as temporary.
3. Run the required Marin validation from the draft PR.
4. Merge the fork PR into the stable `branch`, fetch it, and read the landed SHA. Do
   not assume it matches the pre-merge PR head SHA.
5. Update the Marin draft PR to select the landed `branch` SHA in `tpu.toml` and fix
   `upstream_base` if needed. Because the source changed, build and qualify a replacement
   wheel pair, pin it, rerun focused validation, then undraft.

Final check: `git ls-remote <fork-url> <branch>` must match the relevant
`config/external/vllm/tpu.toml` `commit`, and `uv run config/update-external.py
--check` must pass. Repeat per source when an overlay spans both forks.
