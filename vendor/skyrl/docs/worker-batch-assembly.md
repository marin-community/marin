# Batch assembly

`trainer.batch_builder` selects `driver` (the default), `worker`, or `verify`.
The split Snowball Ultra RLVR1 recipe selects `worker`.

`worker` reads admitted groups on policy workers and retains each DP slice
through forward, advantage preparation, and native policy optimization.
Global prompt and response widths and complete-group outcome credit are shared
across ranks. Compact route rows retain the model's layer, top-k, and expert
geometry, including the prediction position for the first response token.

`verify` also builds an isolated driver reference. Every policy rank must match
its forward-input and training-input digests before worker optimization.
The reference has private metrics and does not publish callbacks or update weights.
Dense dumps in this mode contain the reference training input.

The base trainer supports the regular, dual-clip, GSPO, CISPO, SAPO,
importance-sampling, and behavior-clip losses with RLOO, RLOO-N, or GRPO.
SFT with uniform credit also supports the preference-pair producer.
The shared correction and diagnostic math runs in both modes; worker ratio
percentiles use the distributed histogram, while driver percentiles are exact.

`UnsupportedWorkerBatchError` identifies unsupported worker/verify requests:
DPO or reference-model objectives, critics, distillation, custom trainers or
losses, selectors, step-wise training, advantage batch normalization,
nonzero loop credit, token-valued outcome credit, required DP padding, and
expert tensor parallelism greater than one. Worker mode also rejects dense
batch dumps. Driver mode handles its existing objective and trainer contracts.
Checkpoints admitted under worker/verify require row facts in their verdicts.
