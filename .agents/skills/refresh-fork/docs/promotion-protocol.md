# Prepare and promote `<branch>-next`

An unattended refresh prepares promotion after the fork's e2e passes against
`<branch>-next`, then opens the draft Marin PR. It creates immutable refs for the old
and new tips and leaves the protected stable branch unchanged. An admin hard-swaps
the stable branch after reviewing the draft PR and before that PR merges.

A refresh rebases our patches onto a new upstream base, so `<branch>-next` does not
descend from the current `<branch>` — their upstream bases differ. The promotion is
therefore a **hard swap** (a backed-up force-update), not a merge or fast-forward. A
PR/merge would splice two upstream bases into a merge commit and break the linear
history the fork depends on.

Marin pins exact SHAs and wheels, never a bare branch name, so the branch pointer can
move without changing what Marin resolves. That is what makes the swap safe.

## Prepare the refs

For each pin in the refresh after its e2e passes:

- Confirm `<branch>-next` is exactly the tip the e2e ran against. For a `release:` pin
  also confirm it is the `source_commit` the candidate wheel was built from.
- Record the current remote `<branch>` SHA. Tag it as
  `<branch>-backup/YYYYMMDD/pre-<old-shortsha>` and tag the validated staged tip as
  `<branch>-YYYYMMDD`. Reuse a tag that already points at the expected SHA; stop and
  inspect if either name points elsewhere.
- Push both tags and verify the remote tags resolve to the recorded SHAs.
- Leave `<branch>` unchanged and keep `<branch>-next` available for admin review.

Pin Marin at the exact staged SHA or candidate wheel and regenerate
`external_dependencies.py`. The draft PR must list the staged and stable SHAs, both
tags, and the pending admin hard swap. For an `isolated_project`, keep its uv source
on `main-next` in this draft so regenerating the lock cannot move it back to the old
`main` tip. The date tag keeps the staged SHA reachable. Do not mark the PR ready or
merge it in this state.

## Admin promotion

After reviewing the fork overlay and the draft Marin PR, an admin with ruleset bypass
promotes each pin:

- Confirm `<branch>-next` still resolves to the validated and date-tagged tip.
- Confirm `<branch>` still resolves to the SHA recorded by the rollback tag.
- Hard-swap with a lease so a concurrent move is caught:
  `git push --force-with-lease=<branch>:<old-sha> origin <branch>-next:<branch>`.
- Verify remote `<branch>` resolves to the validated tip. Delete `<branch>-next` or
  leave it for the next cycle; the next refresh force-updates it.

Descriptor and release pins need no edit after this swap because they already record
the exact validated SHA or wheel. For an `isolated_project`, restore the uv source
from `main-next` to `main`, rerun `uv run config/update-external.py <fork>`, and verify
the lock still records the validated SHA. Commit and push that follow-up to the draft
Marin PR before marking it ready or merging it.

## The two-branch vllm fork

The vllm fork carries two pins on different upstream bases, so they cannot share one
branch. It splits them across two stable branches: the GPU wheel builds from `main`
(the release candidate triggers on `push: main`), and the TPU source pin lives on
`tpu`. Each promotes on its own: `main-next` to `main` for the GPU pin, `tpu-next` to
`tpu` for the TPU pin. A partial failure leaves the other pin correct because Marin
resolves an exact wheel or SHA either way. Single-pin forks track `main` directly, so
for them `<branch>` is `main`.

## Partial failure

If one pin's admin promotion succeeds and another fails, Marin stays correct because
it pins exact SHAs. Record the mismatch and ask whether to retry the failed promotion.
If the retry still fails, file a
`marin-community/marin` issue assigned to the descriptor's `blocker_assignee` with the
pinned SHAs, the promoted and failed branches, the rollback tags, the commands
attempted, and the error output.
