# Observer artifact handoff — historical WIP

This branch preserves an incomplete prototype based on commit `114e278e`.
The learner goal remains paused. This checkpoint is not a runnable qualification
packet and must not be used to launch or validate a training run.

The preserved work adds a physical-task route for opt-in shutdown-observer
artifacts, a bounded uploader with file/identity checks and an index written
last, parent-owned uploader timeout/reaping, and an attempted-once handoff during
observer retirement. It also contains draft tests for local file safety,
partial uploads, process ownership and preservation of shutdown errors.

The current learner branch already integrates later reviewed and validated
counterparts. This older checkout is historical context, not the canonical
implementation. Do not apply this patch wholesale over the current learner.

Only publication lint and Python syntax checks are being performed for this
checkpoint. Its behavioral tests, uploader subprocesses, remote storage writes,
native learner and GPU paths are not being executed or requalified. Lint and
syntax results do not establish durable upload, safe lifecycle behavior or
compatibility with the current runtime.

Artifact durability is separate from observer completion and natural native
exit. An uploaded index cannot waive an incomplete observer result, alter the
original post-ACK five-second exit contract, or establish full Snowball
continuation, publication/generation or matched-25 acceptance. No deployment,
hardware action or resumed implementation is authorized by this checkpoint.
