# Resolved score-centering run configurations

Each JSON file is a byte-for-byte copy of `resolved-skyrl.json` from the corresponding
`s3://marin-us-east-02a/marin/users/romain/checkpoints/async-rl/<run>/<version>/` artifact.
The file contains the 151 materialized Hydra arguments and the model/data references used
by the Iris child. `r23_r26.json` describes the shared PPO artifact continued by both jobs.
Run labels, artifact names, versions, and launcher source commits are in
[`score_centering_study.md`](../../score_centering_study.md).

After excluding only run-owned output paths and names, the cap-1.05 Qwen pairs
`r24/r25`, `r27/r28`, and `r30/r31` differ only in
`trainer.algorithm.score_centering_topk` (`0` versus `32`). Their TIS controls retain
`generator.sampling_params.logprobs=32`. Across pairs, the only training setting that
changes is `trainer.seed` (`17`, `18`, or `19`). The Snowball smoke pair likewise differs
only in `score_centering_topk` after excluding run-owned paths and names.
