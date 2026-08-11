# Debugging log for H100 loaded shared-library SASS topology

Diagnose the first fail-closed boundary in the third reviewed H100 evidence
execution and repair it without substituting the separately compiled cubin or
weakening exact topology validation.

## Initial status

Job `/dlwh/shuttle-h100-contract-map-evidence-dbbd9e-v3` used exact source
`dbbd9e4fe53e8ec7ad2c8d409dbaa0351ac064ff`, source tree
`fc78276a63b1e64ff7abe3c976c619d0492727f2`, and immutable image
`sha256:945f44cca0aa44be922c9d806e7b8e6b98915ed22323cca26ca89f23bf3a4e19`.
The task authenticated the launcher and manifest, passed runtime and H100 tool
preflight, compiled the first generated candidate's `.so`, PTX, and cubin, and
disassembled the cubin. It failed at the exact topology comparison after
disassembling the `.so`. The task exited 1 after 15.32 seconds with one failure,
zero preemptions, and zero failure retries.

The task did not export its temporary `.so` or SASS, and the old diagnostic did
not print the actual parsed tuple. The sealed artifact therefore records the
mismatch without inferring its exact shape.

## Hypothesis 1

`cuobjdump` may emit the same unique six functions from a linked fatbinary in
an order different from CUDA source definition order. The current comparison
requires tuple order even though link-time fatbinary function order is not the
execution topology. The authoritative contract is exact unique coverage of all
six generated names with no extras, not tool-emission order.

## Changes to make

- Preserve and hash the authoritative loaded-`.so` SASS before reporting any
  topology mismatch so future negative evidence retains the actual parser
  input.
- Parse the loaded image into a closed exact-coverage record that rejects
  missing, extra, duplicate, lookalike, or malformed function sections while
  allowing emission-order variation.
- Keep every evidence record bound to the loaded `.so`; do not use the separate
  cubin as a substitute.
- Add behavior tests using a representative multi-function cuobjdump fixture,
  including reordered exact coverage and all fail-closed mutations.

## Results

The existing loaded-image evidence path rejected a representative cuobjdump
fixture containing the exact two expected names in reverse section order. The
focused pre-fix test failed at `generated_kernel_records` with
`loaded shared-library SASS kernel identities changed after compilation`.

The repaired parser requires one or more valid addressed instructions in each
unique function section. It rejects empty output, NUL, diagnostics, malformed
function anchors, malformed address records, standalone encodings, empty
sections, repeated or reordered addresses, duplicate names, and any exact
coverage difference. Once coverage matches, it preserves generated source
order for evidence and leaves actual launch-order validation to Nsight Systems.

The compile path writes loaded-image SASS before topology validation, so a
local failure retains the parser input and reports the actual, missing, and
unexpected names. Evidence construction rehashes both the shared library and
its loaded-image SASS before producing kernel records. The separate cubin is
still retained but is not used for loaded-image SASS.

The focused repaired matrix passed 10 tests, and the complete runner module
passed 50 tests on macOS without CUDA or a GPU. A compile-boundary regression
makes fake `cuobjdump` output depend on whether its final input is the loaded
`.so` or the separately compiled cubin. The correct path passes, and an explicit
`.so`-to-cubin mutation fails before artifact construction. The broader package
suite passed 947 tests with one pre-existing snapshot test deselected because
its committed checksum list names an ignored raw log absent from the exact
checkout. The exact remote `.so` was ephemeral and unavailable after task
termination. The repair proves that order-only differences no longer fail, but
it does not prove that section order was the only mismatch in the third H100
attempt.

## Future work

- [ ] If another reviewed execution is authorized, retain the complete loaded
  SASS before validation and include the actual/expected topology in the
  diagnostic.
