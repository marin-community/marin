# Fixed-Base Overlay Fork PR Protocol

Use this when adding or fixing a Marin overlay commit while keeping the current
upstream base fixed. Do not use it for an upstream-base refresh.

Because the base does not move, the new commit sits on the current fork tip and
history stays linear. Branch from that tip, add the smallest overlay, run the
fork's ordinary checks, and open a fork PR. Record why the patch is still needed
and what upstream change will let us drop it.

For a TPU vLLM or tpu-inference overlay, do not put a temporary source SHA in
Marin. Freeze the exact reviewed fork commit and use it as one source input to the
TPU release procedure in `../SKILL.md`. Marin changes only after that procedure
selects one public vLLM requirement. A later source edit invalidates the release
receipt and must be reviewed before any rebuild.

For another fork, follow its pin kind in the refresh skill. A branch-based pin
may stage the exact PR head for validation and then must read back the landed
commit; never assume a GitHub merge preserves the PR head SHA. A release-based
pin must use its existing producer and artifact validation path.
