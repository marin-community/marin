# NeMo Gym two-row sweep

The current NeMo Gym collection contains 29 task datasets and three training
blends. Two deterministic rows were sampled from each task dataset at its Hub
revision. Six rows were accepted: two competitive-coding rows, two
instruction-following rows, and two Workplace rows. Six rows were rejected.
Forty-six rows were deferred because preserving their source semantics needs an unsupported verifier,
an unavailable stateful environment, or a reactive interaction.

The action-prediction findings in `reviews/agent-corpora.json` were superseded by
`reviews/action-wire-format.json`. The four rows depend on hidden tool results or
authentication state, so removing those records would change the task. They are
deferred instead of being normalized into bare next-action prediction.

The code tasks are `nemo/code-answer/ced8ee9e1a54129fab130febf5c4e3f5` and
`nemo/code-answer/c6ab22b152348b2663857766a436bfe0`, from the pinned
competitive-coding source revision at offsets 5135 and 13176. Their tests and
source provenance remain private; model-visible tasks are answer-only code
generation tasks. The instruction-following tasks are
`nemo/ifeval/17616/binary` and `nemo/ifeval/44654/binary`, from offsets 0 and 1
of their pinned source revision; their instruction constraints remain private.

The Workplace tasks `nemo/workplace/536` and `nemo/workplace/1163` use the shared pinned Workplace seed and provider. Their source rows and target actions remain private; the model receives only the source conversation and the declared tool surface.
