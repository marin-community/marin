# TaskSpec source pin

`task-spec-v0.9.json` is copied byte-for-byte from
`lib/taskcompendium/schema/task-spec-v0.9.json` at Marin commit
`dc6b501c8604bcd2e3c20c1e9947679845fdfef8` (PR #9187). Its SHA-256 is
`8ffdde2cd6e80d24fc630d36e864ce25280b91c2105564b64da640ea641cf1fd`.

`source.lock.json` pins the executable Python sources and transitive Harbor,
TaskTrove verifier, and ShellSim revisions. It also pins `uv.lock`, because that
file controls the executable dependency graph. Synthesis does not implement this
schema itself. It hash-checks a staged checkout, then runs
`capability_pipeline/taskcompendium_driver.py` inside that checkout's `uv`
environment so decoding, verifier validation, grading, and lowering use the
official implementation.

`composite_extension.lock.json` and
`patches/composite_required_extension.patch` define one exact overlay on that
base. The overlay makes native `SemanticVerifier` reject mandatory extensions
and teaches the Harbor runner to load the preserved specification only when the
manifest, configured composite verifier, and exact extension hashes match.
Composite Harbor packages also replace ordinary `specification.json` with an
unsupported-extension sentinel and retain the valid record under
`composite-specification.json`, so even an unpatched old decoder fails closed.
The controller applies the patch only to a private temporary copy after the base
tree passes `source.lock.json`; it records and checks both patched files' base,
patch, and result hashes.
