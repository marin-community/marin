# Pre-training under infinite compute

Paper link: TODO

[WandB report](https://wandb.ai/stanford-mercury/suhas-data-efficiency/reports/Pre-training-under-infinite-compute--VmlldzoxNDM5NzUzMQ)

[Issue #1573](https://github.com/marin-community/marin/issues/1573)

## Summary
Compute is growing faster than data. We study how to do pre-training if you had infinite compute. Here are the main experimental results:

- Only increasing epoch count and parameter count isn't great.
- If you tune weight decay (with optimal wd over 30x larger than default of 0.1!) you get way better performance and you get clean power law scaling.
- If you ensemble models, you also get a power law, but it can surprisingly have a lower asymptote. You can also compose both parameter scaling and ensemble scaling for a better asymptote.
- These results preliminarily scale up with the token count.
- You can distill ensembles into a single model and retain most of the loss improvement. In fact, you can distill a model into itself and get an improvement!
- These results generalize to downstream benchmarks.
- These tricks apply out of the box to CPT and improve data-efficiency on math mid-training.

## Using this directory

### Training code

We release code that captures the configs passed into our pre-training runs for {epoching, parameter scaling, regularization, ensembling, distillation, CPT}.

Standard marin instructions to run all the files, please poke around. You will need a custom branch of levanter with ensembling implemented (you will not need to do this once ensembling is merged into main).

To get this branch, please
- `mkdir submodules` so that `marin/submodules` exists. Go to this directory
- `git clone https://github.com/marin-community/levanter.git`
- `git checkout suhas/eval-ensemble`

### Plotting code

`plotting.py` which is a behemoth that can reproduce all of our plots programatically from our WandB runs. Instructions on usage:

- The plotting code has multiple "modes" which you specify through the `--mode` flag, take a look at the bottom of the file.
- The first time you run the plotting code, it will need to build a cache of the runs from WandB, please run with `--build_cache` (can take up to 15 minutes). After this, plotting should be quite fast.
