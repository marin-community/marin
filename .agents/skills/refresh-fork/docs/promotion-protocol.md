# Promote `<branch>-next` onto the pin's stable branch

Run this once the fork's e2e has passed against `<branch>-next`, as the last step
before opening the Marin PR. It makes the pin's stable `branch` point at the
validated tip, keeps the fork's history linear (Marin overlays over upstream, no
merge commits), and leaves a rollback.

A refresh rebases our patches onto a new upstream base, so `<branch>-next` does not
descend from the current `<branch>` — their upstream bases differ. The promotion is
therefore a **hard swap** (a backed-up force-update), not a merge or fast-forward. A
PR/merge would splice two upstream bases into a merge commit and break the linear
history the fork depends on.

Marin pins exact SHAs and wheels, never a bare branch name, so the branch pointer can
move without changing what Marin resolves. That is what makes the swap safe.

## Per pin

For each pin being promoted (a group promotes each of its pins):

- Confirm `<branch>-next` is exactly the tip the e2e ran against. For a `release:` pin
  also confirm it is the `source_commit` the promoted wheel was built from.
- Record the current remote `<branch>` SHA and create
  `<branch>-backup/YYYYMMDD/pre-<old-shortsha>` at it. If that backup already exists at
  the old SHA, reuse it; if it points elsewhere, stop and inspect.
- Hard-swap with a lease so a concurrent move is caught:
  `git push --force-with-lease=<branch>:<old-sha> origin <branch>-next:<branch>`.
- Verify remote `<branch>` resolves to the validated tip. Delete `<branch>-next` (or
  leave it for the next cycle; it is force-updated on the next refresh either way).

Then re-pin Marin at the now-stable `<branch>` tip (the same SHA/wheel it validated,
now reachable via `<branch>`) and regenerate `external_dependencies.py`. Because the
tip is unchanged by the swap, a pin set from `<branch>-next` before promotion already
holds the right revision.

## Blank `main` on multi-pin forks

A fork with more than one pin (vllm: `gpu` and `tpu`) cannot host both on `main` —
they sit on different upstream bases. Keep `main` blank: it carries only the README in
`fork-main-readme.md`, pointing readers at the pin branches. Never advance `main` to a
pin tip. Single-pin forks track `main` directly, so for them `<branch>` is `main` and
this section does not apply.

## Partial failure

If one pin's promotion succeeds and another fails, Marin stays correct because it pins
exact SHAs — do not roll back the promoted pin automatically. Record the mismatch and
ask whether to retry the failed promotion. If the retry still fails, file a
`marin-community/marin` issue assigned to the descriptor's `blocker_assignee` with the
pinned SHAs, the promoted and failed branches, the backup branches, the commands
attempted, and the error output.
