# Debugging log for the shared consult-agent action

Replace Marin's repeated Claude invocation and failure-classification steps with
one reusable action in `marin-style`.

## Initial status

Marin PR #7515 references
`marin-community/marin-style/actions/classify-claude@main`, but `main` does not
contain that action until the dependent marin-style PR merges. The review and
lint-review jobs fail while downloading the missing action.

## Hypothesis 1

The classifier is the wrong abstraction boundary. Each workflow must still
configure `continue-on-error`, forward the Claude execution file, and add a
second step that distinguishes quota exhaustion from other failures.

## Changes to make

Add `actions/consult-agent` to `marin-style`. It will invoke the pinned Claude
Code action, treat HTTP 429 and weekly-limit responses as a successful skipped
consultation, preserve all other failures, and forward the Claude action's
outputs. Replace Marin's paired invocation and classifier steps with this one
action.

## Results

`consult-agent` now owns the Claude action pin, `continue-on-error`, quota
classification, and output forwarding. Seven Marin call sites use one action
step instead of a Claude step followed by a classifier step.

The classifier probe covered success, HTTP 429, and a non-quota failure. Success
and quota exhaustion wrote the expected outputs; the non-quota trace raised an
error. `actionlint` accepted the three changed workflows after ignoring the
existing deliberate `if: false` guard on the disabled prose-cleanup workflow.

## Future work

- [ ] Merge the `marin-style` PR before the dependent Marin PR so `@main`
  resolves during CI.
