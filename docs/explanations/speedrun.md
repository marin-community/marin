# Speedrun Overview

## Introduction
Marin Speedrun, inspired by the [nanogpt Speedrun](https://github.com/KellerJordan/modded-nanogpt), is a benchmark
aimed at improving the compute efficiency of language model training, by
incentivizing researchers to develop new model architectures, optimizers, and
training strategies. We fix the dataset to be the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and allow participants to specify their own FLOPs budget. With a minimal amount of code, participants can submit their own runs to the leaderboard. See the [submission guide](../tutorials/submitting-speedrun.md) to get started, the [leaderboard](https://marin.community/speedrun) for looking at current results, and examples of submissions in the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun).


## Dataset
We fix the dataset to be the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Since the full dataset is more than 9TB, we provide a pre-tokenized, shuffled version of FineWeb-Edu consisting of 10B tokens to enable participants to be able to run their speedruns on local GPUs. You can find it at https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B. The speedrun script has code for automatically download it to the path you specify; see the [submission guide](../tutorials/submitting-speedrun.md) for more details.

## Metrics

We track the following metrics in Marin Speedrun:
- **C4-EN BPB (Bits per Byte)** (quality): Bits-per-byte on the validation portion of the `c4-en` (English portion of the C4) dataset; lower values are better.
- **Hardware FLOP Cost** (compute): Floating point operations the hardware used to train the model could have performed in the time the model was trained; lower values are better. See [the explainer on how we calculate FLOPs](../explanations/speedrun-flops-accounting.md) for more details.

We also track the **Pareto frontier**, the set of submissions that are not
outperformed in both metrics by any other submission. In other words, a
submission is Pareto-optimal if no other model achieves better BPB and uses
fewer FLOPs. This allows us examine the tradeoff between compute and model quality.

Submitting to the Speedrun Leaderboard consists of the following steps:

1. Define a configuration for training a model.
2. Train it on your hardware; both GPUs and TPUs are supported.
3. Submit the configuration and the generated results file to the Marin GitHub repository via a pull request.
4. Watch it appear on the leaderboard!

Here is an [example submission](https://github.com/marin-community/marin/blob/main/experiments/speedrun/llama_75m/llama_75m.py),
and here is the [current leaderboard](https://marin.community/speedrun).

## Next Steps

- To get started, see the [submission guide](../tutorials/submitting-speedrun.md).
- To see examples of submissions, see the [speedrun directory](https://github.com/marin-community/marin/tree/main/experiments/speedrun).
- To see current runs and their results, see the [leaderboard](https://marin.community/speedrun).
