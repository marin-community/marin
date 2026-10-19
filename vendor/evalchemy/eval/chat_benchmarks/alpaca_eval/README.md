# <a href="https://tatsu-lab.github.io/alpaca_eval/" target="_blank"><img src="https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/docs/AlpacaFarm_small.png" width="35"></a> [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) : An Automatic Evaluator for Instruction-following Language Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![discord](https://img.shields.io/badge/discord-server-blue?logo=discord&logoColor=white)](https://discord.gg/GJMxJSVZZM)


**AlpacaEval 2.0 with length-controlled win-rates** ([paper](https://arxiv.org/abs/2404.04475)) has a spearman correlation of **0.98** with [ChatBot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) while costing less than **$10** of OpenAI credits run and running in less than 3 minutes. Our goal is to have a benchmark for chat LLMs that is: fast (< 5min), cheap (< $10), and highly correlated with humans (0.98). Here's a comparison with other benchmarks:

<p float="left" align="middle">
<img src="figures/chat_correlations_no_ae.png" alt="LC AlpacaEval is the most highly correlated benchmark with Chat Arena." width="500"/>
</p>

---

Updates:

:tada: **Length-controlled Win Rates** are out and used by default! This increases the correlation with ChatBot Arena from 0.93 to 0.98, while significantly decreasing length gameability. The raw win rates are still shown on the website and the CLI. More details [here](#length-controlled-win-rates).

:tada: **AlpacaEval 2.0** is out and used by default! We improved the auto-annotator (better and cheaper) and use GPT-4 preview as baseline. More details [here](#alpacaeval-20). For the old version, set your environment variable `IS_ALPACA_EVAL_2=False`.

---

<details open>
  <summary><b>Table of Contents</b></summary>

1. [Overview](#overview)
2. [Quick Start](#quick-start)
2. [Leaderboards and how to interpret them](#leaderboards-and-how-to-interpret-them)
    - [Models](#models)
    - [Evaluators](#evaluators)
3. [Use-cases](#use-cases)
    - [Evaluating a model](#evaluating-a-model)
    - [Making a new leaderboard](#making-a-new-leaderboard)
    - [Making a new evaluator](#making-a-new-evaluator)
4. [Contributing](#contributing)
    - [Contributing a model](#contributing-a-model)
    - [Contributing an evaluator](#contributing-an-evaluator)
    - [Contributing an eval set](#contributing-an-eval-set)
    - [Contributing a completion function](#contributing-a-completion-function)
5. [Limitations](#limitations)
6. [Analysis](#additional-analysis-and-plots)
    - [Length-controlled AlpacaEval](#length-controlled-alpacaeval-lcae)
    - [Analyzing an evaluator](#analyzing-an-evaluator)
    - [Analyzing an eval set](#analyzing-an-eval-set)
7. [Citation](#citation)
8. [Additional information](#additional-information)
   - [Length-controlled win rates](#length-controlled-win-rates)
   - [AlpacaEval 2.0](#alpacaeval-20)
   - [Data Release](#data-release)
   - [Differences with AlpacaFarm](#differences-with-alpacafarm)
   - [Related work](#related-work)
   - [Interpreting annotations](#interpreting-annotations)
   - [Major updates](#major-updates)

</details>

# Overview


Evaluation of instruction-following models (e.g., ChatGPT) typically requires human interactions. This is
time-consuming, expensive, and hard to replicate. AlpacaEval in an LLM-based automatic evaluation that is fast, cheap,
replicable, and validated against 20K human annotations.
It is particularly useful for model development.
Although we improved over prior automatic evaluation pipelines, there are still fundamental [limitations](#limitations) like the preference for longer outputs.
AlpacaEval provides the following:

- [**Leaderboard**](https://tatsu-lab.github.io/alpaca_eval/): a leaderboard of common models on the AlpacaEval
  evaluation set. **Caution**: Automatic evaluators (e.g. GPT-4) may be biased towards models that generate longer outputs and/or that were fine-tuned on the model underlying the evaluator (e.g. GPT-4).
- [**Automatic evaluator**](#evaluators): an automatic evaluator that has high agreement with humans (validated on 20K
  annotations). We evaluate a
  model by
  measuring the fraction of times a powerful LLM (e.g. GPT-4) prefers the outputs from that model
  over
  outputs from a reference model. Our evaluators enable caching and output randomization by default.
- [**Toolkit for building automatic evaluators**](#analysis): a simple interface for
  building advanced automatic evaluators (e.g. with caching, batching, or multi-annotators) and analyzing them (quality,
  price, speed, statistical power, bias, variance etc).
- [**Human evaluation data**](#data-release): 20K human preferences between a given and reference model
  on the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm/tree/main)
  evaluation set. 2.5K of these are cross-annotations (4 humans annotating the same 650 examples).
- [**AlpacaEval dataset**](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json): a simplification
  of [AlpacaFarm's](https://github.com/tatsu-lab/alpaca_farm/tree/main) evaluation set, where "instructions" and "inputs" are merged into one field, and reference outputs are longer. [Details here](#data-release).




<details>
  <summary><b>When to use and not use AlpacaEval?</b></summary>

**When to use AlpacaEval?**
Our automatic evaluator is a quick and cheap proxy for human evaluation of simple
instruction-following tasks.
It is useful if you
have to run many evaluations quickly, e.g., during model development.

**When not to use AlpacaEval?**
As any other automatic evaluator, AlpacaEval should **not replace human evaluation in
high-stake decision-making**, e.g., to decide on model release. In particular, AlpacaEval is limited by the fact
that (1) the instructions in the eval set might not be representative of advanced usage of LLMs; (2) automatic
evaluators may have biases such as favoring style over
factuality of the answer; and (3) AlpacaEval does not measure the risks that a model could cause.
Details in [limitations](#limitations).

</details>


# Quick Start

To install the stable release, run

```bash
pip install alpaca-eval
```

To install the nightly version, run

```bash
pip install git+https://github.com/tatsu-lab/alpaca_eval
```

Then you can use it as follows:

```bash
export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see client_configs/README.md 
alpaca_eval --model_outputs 'example/outputs.json' 
```

This will print the leaderboard to the console, and save both the leaderboard and the annotations to the same directory as the `model_outputs` file. Important parameters are the following:

- **model_outputs** : A path to a json file for the outputs of the model to add to the leaderboard. Each dictionary
  should
  contain the keys `instruction` and `output`.
- **annotators_config**: This is the annotator to use. We recommend using `weighted_alpaca_eval_gpt4_turbo` (
  default for AlpacaEval 2.0), which has a
  high agreement rate with our human annotation data, large context size, and is pretty cheap. For a comparison of all annotators see [here](#evaluators).
- **reference_outputs**:  The outputs of the reference model. Same format as `model_outputs`. By default, this
  is `gpt4_turbo` for AlpacaEval 2.0.
- **output_path**: Path for saving annotations and leaderboard.

If you don't have the model outputs, you can
use [`evaluate_from_model`](https://github.com/tatsu-lab/alpaca_eval/tree/main#evaluating-a-model) and
pass a local path or a name of a
HuggingFace
model, or a model from a standard API (OpenAI, Anthropic, Cohere, google, ...). Other commands:

<details open>
  <summary><code>>>> alpaca_eval -- --help</code></summary>

```
SYNOPSIS
    alpaca_eval COMMAND

COMMANDS
    COMMAND is one of the following:

     evaluate
       Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

     evaluate_from_model
       Evaluate a model from HuggingFace or an API provider. This is a wrapper around `evaluate` which includes generating from a desired model.

     make_leaderboard
       Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

     analyze_evaluators
       Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).
```

</details>

For more information about each function use `alpaca_eval <command> -- --help`.

# Leaderboards and how to interpret them

## Models

Our leaderboards are computed on the [AlpacaEval dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_eval).
We precomputed the leaderboard for important models using different baseline models and autoannotators. 
Our two main leaderboards ("AlpacaEval 2.0" and "AlpacaEval") can be found
[on this page](https://tatsu-lab.github.io/alpaca_eval/).
"AlpacaEval 2.0" uses `weighted_alpaca_eval_gpt4_turbo` for the annotator and `gpt4_turbo` for the baseline.
"AlpacaEval" uses `alpaca_eval_gpt4` for the annotator and `text_davinci_003` for the baseline.
For all precomputed leaderboards see [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/leaderboards).
Later we also show how to [add your model](https://github.com/tatsu-lab/alpaca_eval#evaluating-a-model) to the
leaderboard and how to make
a [new leaderboard for your evaluator/dataset](https://github.com/tatsu-lab/alpaca_eval#making-a-new-leaderboard).
See [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs) for the configs of all
models that are available out of the box.

**AlpacaEval minimal leaderboard**:

|                       | Win Rate | Std Error |
|:----------------------|---------:|----------:|
| gpt4                  |     95.3 |       0.7 |
| claude                |     88.4 |       1.1 |
| chatgpt               |     86.1 |       1.2 |
| guanaco-65b           |     71.8 |       1.6 |
| vicuna-13b            |     70.4 |       1.6 |
| text_davinci_003      |     50.0 |       0.0 |
| alpaca-farm-ppo-human |     41.2 |       1.7 |
| alpaca-7b             |     26.5 |       1.5 |
| text_davinci_001      |     15.2 |       1.2 |

<details>
  <summary><b>How exactly are those metrics computed?</b></summary>

**Win Rate**: the win rate measures the fraction of time the model's output is preferred over the reference's outputs (`test-davinci-003` for AlpacaEval and `gpt4_turbo` for AlpacaEval 2.0).
More specifically, to compute the win rate we collect pairs of outputs of the desired model on every instruction from
the ApacaEval dataset.
We then pair each output with the output of our reference model (e.g. `text-davinci-003`) on the same instruction.
We then ask our automatic evaluator which output they prefer.
See [AlpacaEval's](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4)
and [AlpacaEval 2.0's](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo) prompts and configs, in particular we randomize the order of
outputs to avoid position bias.
We then average the preferences over all instructions in the dataset to get the win rate of the model over the baseline.
If both outputs are exactly the same we use a half preference for both models.

**Standard error**: this is the standard error (normalized by N-1) of the win rate, i.e., the preferences averaged over
the different instructions.

</details>

<details>
  <summary><b>Details about our auto-annotator: <code>alpaca_eval_gpt4</code></b></summary>

Our `alpaca_eval_gpt4` (
see [configs](#https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/configs.yaml#L5))
annotator averages over preferences, where preferences are obtained as follows:

1. it takes in an instruction and a pair of outputs (from the desired model and the reference model)
2. if a preference was this triple was already computed, it returns it (i.e. it uses caching)
3. it randomizes the order of the outputs to avoid position bias
4. it formats the instruction and outputs into
   the [following zero-shot prompt](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt),
   which asks to order the outputs in order of preference
5. it completes the prompt using GPT4 with `temperature=0`
6. it parses the preference from the completions and returns it

The annotator is a mix between (and was highly influenced by) [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm)
and [Aviary](https://github.com/ray-project/aviary/tree/master) evaluators.
In particular, we use the same code as for AlpacaFarm (caching/randomization/hyperparameters) but use a ranking prompt
similar to that of Aviary.
We make changes to Aviary's prompt to decrease the bias for longer outputs.
Details in [Related work](#related-work).

For AlpacaEval 2.0 we use `weighted_alpaca_eval_gpt4_turbo`, which uses logprobs to compute continuous preference and uses GPT4_turbo as model (
see [configs](#https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml)).

</details>



## Evaluators

We evaluate different automatic annotators on the AlpacaEval set by comparing to
2.5K [human annotations](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json)
we collected (~650 instructions each with 4 human annotations).
Below we show metrics for our suggested evaluators (`weighted_alpaca_eval_gpt4_turbo`,`alpaca_eval_gpt4`), for prior
automatic
evaluators ([`alpaca_farm_greedy_gpt4`](https://github.com/tatsu-lab/alpaca_farm),[`aviary_gpt4`](https://aviary.anyscale.com/),[`lmsys_gpt4`](https://chat.lmsys.org/)),
for humans (`humans`), and for different base models with essentially the same
prompt (`gpt4`,`claude`,`text_davinci_003`,`chatgpt_fn`,`guanaco_33b`, `chatgpt`).
See [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs) for the configs of all
evaluators that are available out of the box and their associated metrics.

|                                 |   Human agreement |   Price [$/1000 examples] |   Time [seconds/1000 examples] |   Spearman corr. |   Pearson corr. |   Bias |   Variance |   Proba. prefer longer |
|:--------------------------------|------------------:|--------------------------:|-------------------------------:|-----------------:|----------------:|-------:|-----------:|-----------------------:|
| alpaca_eval_gpt4                |              69.2 |                      13.6 |                           1455 |             0.97 |            0.93 |   28.4 |       14.6 |                   0.68 |
| alpaca_eval_cot_gpt4_turbo_fn   |              68.6 |                       6.3 |                           1989 |             0.97 |            0.90 |   29.3 |       18.4 |                   0.67 |
| alpaca_eval_gpt4_turbo_fn       |              68.1 |                       5.5 |                            864 |             0.93 |            0.82 |   30.2 |       15.6 |                   0.65 |
| alpaca_eval_llama3_70b_fn       |              67.5 |                       0.4 |                            209 |             0.90 |            0.86 |   32.3 |       8.2 |                   0.79 |
| gpt4                            |              66.9 |                      12.5 |                           1037 |             0.88 |            0.87 |   31.5 |       14.6 |                   0.65 |
| alpaca_farm_greedy_gpt4         |              66.4 |                      15.3 |                            878 |             0.85 |            0.75 |   30.2 |       19.3 |                   0.60 |
| alpaca_eval_cot_gpt4_turbo_fn |              65.7 |                       4.3 |                            228 |             0.78 |            0.77 |   33.9 |       23.7 |                   0.61 |
| humans                          |              65.7 |                     300.0 |                          36800 |             1.00 |            1.00 |    0.0 |       34.3 |                   0.64 |
| claude                          |              65.3 |                       3.3 |                            173 |             0.93 |            0.90 |   32.4 |       18.5 |                   0.66 |
| lmsys_gpt4                      |              65.3 |                      13.9 |                          17982 |             0.98 |            0.97 |   31.6 |       15.9 |                   0.74 |
| text_davinci_003                |              64.1 |                       8.7 |                            121 |             0.85 |            0.83 |   33.8 |       22.7 |                   0.70 |
| longest                         |              62.2 |                       0.0 |                              0 |             0.27 |            0.56 |   37.8 |        0.0 |                   1.00 |
| chatgpt                         |              57.3 |                       0.8 |                            285 |             0.72 |            0.71 |   39.4 |       34.1 |                   0.59 |



<details>
  <summary><b>How exactly are those metrics computed?</b></summary>

We now explain in words how we compute the metrics in the table
above. [The code is here](https://github.com/tatsu-lab/alpaca_eval/blob/f05cbd651b79ac93906b19d01fe443b45828b0f2/src/alpaca_eval/analyze.py#L366).

**Human agreement**: this measures the agreement between the current annotator and the majority preferences of
humans on
our
~650 annotations from
our [cross-annotation set](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_farm_human_crossannotations.json),
which contains 4 human annotations per example.
To estimate the agreement between a single human (`humans` row in the table above) and the majority of humans, we take
one of the 4 annotations and compute the accuracy that it has when predicting the mode of the other 3 annotations.
We then average this accuracy over all 4 annotations and over the 650 instructions to get the human agreement, i.e., we
compute the expected (over humans and samples)
leave-one-out agreement.
If the mode is not unique, we take one of the modes at random.
We perform exactly the same computation for the automatic annotators, so that the final numbers are comparable.

**Price [$/1000 examples]**: this is the average price of every 1000 annotations.
For humans, it is the price that [we paid Mechanical Turkers](https://arxiv.org/abs/2305.14387) to collect those
annotations ($21/hour).
If the price depends on the machine used to compute the annotations (e.g. Guanaco) we leave it empty.

**Time [seconds/1000 examples]**: this is the average time it takes to compute 1000 annotations.
For humans, it is the estimated median time that each Mechanical Turker took to annotate 1000 examples.
For automatic annotators, it is the average time that it took us when running the annotations. Note that this can depend
on API limits that are different for different users and the number of requests that the clusters are
processing.

**Spearman corr.**: this measures the Spearman correlation between a leaderboard computed with the auto-annotator's preference and the leaderboard computed with human preferences. As with `Human agreement`, we use the human annotations from AlpacaFarm but we now consider the method-level agreement rather than only the sample-wise agreement with humans. Note that we only use have 9 models and so the correlation is not very reliable. 

**Pearson corr.**: same as with `Spearman corr.` but with Pearson correlation.

**Bias**: agreement between the most likely human label and the most likely automatic one.
For automatic annotators we estimate it by sampling 4 different annotations for each example.
The randomness here comes from the order of the outputs in the prompt, sampling from the LLM, and if applicable the
order of the instruction in the batch and the choice of annotator in the pool.
We then take the mode of the 4 annotations and compute the accuracy of the mode when predicting the mode of the 4 human
annotations.
Note that this is likely an overestimate on the real bias that we would get if we had an "infinite" number of
cross-annotations.
A low bias means that the annotator has in expectation the same preferences as humans.
For the case of humans, the bias is zero by definition.
Note that this is related to but not the standard statistical bias, because we take the mode instead of average over
annotations and we consider 0-1 loss instead of squared loss.


**Variance**: expected agreement a single automatic preference and the most likely one.
We estimate it the same way as we estimated "human agreement" for humans, i.e., we take the expected leave one out error
when predicting the mode of the 3 annotations using the 4th annotation.
A low variance means that the annotator is consistent with its preference, i.e., if you sample from it with different
seeds it will give the same result.
As with the bias, this is not exactly the standard statistical variance, because we take the mode instead of average
over annotations and we
consider 0-1 loss instead of squared loss.

Note that the "human agreement" is tightly related to the bias and variance. In particular, the variance
measures the error due to the fact that we only use a single annotation while the bias aims to measure the irreducible
error
for the current annotator.

**Proba. prefer longer**: this is the probability that the annotator prefers the longer output when one of the two
outputs is significantly longer than the other (more than 30 characters difference).

In the [full table](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md) we
also provide the following metrics:

**Proba. prefer lists**: this is the probability that the annotator prefers the output that contains a list/bullet
points when one output does but not the other.

**Proba. prefer 1**: this is the probability that the annotator prefers the first of the pair of outputs. All our
proposed annotators randomize over outputs in the prompt, so this should be 0.5. Prior annotators, such as `lmsys`
and `aviary`, do not.

**# parsed**: this is the number of examples that the annotator was able to parse.

Note that if the variance and bias is empty, it means that we only performed one single annotation for each 648 example
due to resource (time and price) constraints. This explains why the #parsed is 648, otherwise it should be 2592.

</details>

<details>
  <summary><b>Tips for choosing evaluators</b></summary>

Overall we recommend using `annotators_config=weighted_alpaca_eval_gpt4_turbo` if you want the high agreement with humans, and
`annotators_config=chatgpt_fn` if you are on a tight budget.

When choosing an annotator we recommend you to consider the following (the first three are obvious):

- `"Human agreement [%]"`
- `"Price [$/1000 examples]"`
- `"Time [seconds/1000 examples]"`
- `"* corr."` approx. > 0.7. It is important that the correlation is not too low, but we do not recommend using it as the main metric as the correlation is computed on only 9 models. 
- `"Proba. prefer longer"` approx. < 0.7. Indeed, we found see that the majority of preference of human annotators have
  strong bias for longer answers (as shown by the
  high [performance=62.2](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md)
  of
  the `"longest"` evaluator that always
  prefers the longest output). This suggests that it might more of a bias with the human annotators. In order to avoid
  having leaderboards with strong biases for length, we suggest using automatic annotators with less than 0.7 "Proba.
  prefer longer".
- `"Variance"` approx. < 0.2. We believe that a good evaluator should have as little variance as possible so that
  results are mostly reproducible. Note that variance can be desirable in the case where we are simulating humans
  as shown in [AlpacaFarm](https://arxiv.org/abs/2305.14387).

We filtered the annotators that do not satisfy those requirements in the table above (besides humans / ChatGPT / 003 /
lmsys for
reference purposes). For
all
results see [here](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md).
In general, we found `weighted_alpaca_eval_gpt4_turbo` to be a good trade-off between quality / price / time /
variance / length bias.

</details>

The above metrics are computed with respect to annotations from crowd-workers. Although useful, those annotations are
not perfect, e.g., crowd-workers often favor style
over
factuality. We thus recommend users to validate automatic evaluators on their own instructions and human annotations.
Details in [limitations](#limitations).

# Use-cases


## Evaluating a model

<details>
  <summary><code>>>> alpaca_eval evaluate -- --help</code></summary>

```
NAME
    alpaca_eval evaluate - Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

SYNOPSIS
    alpaca_eval evaluate <flags>

DESCRIPTION
    Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

FLAGS
    --model_outputs=MODEL_OUTPUTS
        Type: Optional[Union]
        Default: None
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe, datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction` and `output` with optional `input`. If None, we just print the leaderboard.
    -r, --reference_outputs=REFERENCE_OUTPUTS
        Type: Union
        Default: <func...
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are a specific set of Davinci 003 outputs on the AlpacaEval set:
    --annotators_config=ANNOTATORS_CONFIG
        Type: Union
        Default: 'alpaca_eval_gpt4_turbo_fn'
        The path the (or list of dict of) the annotator's config file. For details see the docstring of `PairwiseAnnotator`.
    -n, --name=NAME
        Type: Optional[Optional]
        Default: None
        The name of the model to add to the leaderboard. If None we check if `generator is in model_outputs` if not we use "Current model".
    -o, --output_path=OUTPUT_PATH
        Type: Union
        Default: 'auto'
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save. If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.
    -p, --precomputed_leaderboard=PRECOMPUTED_LEADERBOARD
        Type: Union
        Default: 'auto'
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.
    --is_overwrite_leaderboard=IS_OVERWRITE_LEADERBOARD
        Type: bool
        Default: False
        Whether to overwrite the leaderboard if the model is already in it.
    -l, --leaderboard_mode_to_print=LEADERBOARD_MODE_TO_PRINT
        Type: Optional
        Default: 'minimal'
        The mode of the leaderboard to use. Only used if the precomputed leaderboard has a column `mode`, in which case it will filter the leaderboard by this mode. If None keeps all.
    -c, --current_leaderboard_mode=CURRENT_LEADERBOARD_MODE
        Type: str
        Default: 'community'
        The mode of the leaderboard for the current method.
    --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        Type: bool
        Default: False
        Whether to return the metrics instead of printing the results.
    -f, --fn_metric=FN_METRIC
        Type: Union
        Default: 'pairwise_to_winrate'
        The function or function name in `metrics.py` that will be used to convert preference to metrics. The function should take a sequence of preferences (0 for draw, 1 for base win, 2 when the model to compare wins) and return a dictionary of metrics and the key by which to sort the leaderboard.
    -s, --sort_by=SORT_BY
        Type: str
        Default: 'win_rate'
        The key by which to sort the leaderboard.
    --is_cache_leaderboard=IS_CACHE_LEADERBOARD
        Type: Optional[Optional]
        Default: None
        Whether to save the result leaderboard to `precomputed_leaderboard`. If None we save only if max_instances not None. A preferred way of adding models to the leaderboard is to set `precomputed_leaderboard` to the previously saved leaderboard at `<output_path>/leaderboard.csv`.
    --max_instances=MAX_INSTANCES
        Type: Optional[Optional]
        Default: None
        The maximum number of instances to annotate. Useful for testing.
    --annotation_kwargs=ANNOTATION_KWARGS
        Type: Optional[Optional]
        Default: None
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.
    -A, --Annotator=ANNOTATOR
        Default: <class 'alpaca_eval.annotators.pairwise_evaluator.PairwiseAn...
        The annotator class to use.
    Additional flags are accepted.
        Additional arguments to pass to `PairwiseAnnotator`.
```

</details>

<details>
  <summary><code>>>> alpaca_eval evaluate_from_model -- --help</code></summary>

```
NAME
    alpaca_eval evaluate_from_model - Evaluate a model from HuggingFace or an API provider. This is a wrapper around `evaluate` which includes generating from a desired model.

SYNOPSIS
    alpaca_eval evaluate_from_model MODEL_CONFIGS <flags>

DESCRIPTION
    Evaluate a model from HuggingFace or an API provider. This is a wrapper around `evaluate` which includes generating from a desired model.

POSITIONAL ARGUMENTS
    MODEL_CONFIGS
        Type: Union
        A dictionary or path (relative to `models_configs`) to a yaml file containing the configuration of the model to decode from. If a directory,we search for 'configs.yaml' in it. The keys in the first dictionary should be the generator's name, and the value should be a dictionary of the generator's configuration which should have the

FLAGS
    -r, --reference_model_configs=REFERENCE_MODEL_CONFIGS
        Type: Optional[Union]
        Default: None
        Same as in `model_configs` but for the reference model. If None, we use the default Davinci003 outputs.
    -e, --evaluation_dataset=EVALUATION_DATASET
        Type: Union
        Default: <func...
        Path to the evaluation dataset or a function that returns a dataframe. If None, we use the default evaluation
    -a, --annotators_config=ANNOTATORS_CONFIG
        Type: Union
        Default: 'alpaca_eval_gpt4_turbo_fn'
        Path to the annotators configuration or a dictionary. If None, we use the default annotators configuration.
    -o, --output_path=OUTPUT_PATH
        Type: Union
        Default: 'auto'
        Path to save the generations, annotations and leaderboard. If auto saves at `results/<model_name>`
    -m, --max_instances=MAX_INSTANCES
        Type: Optional[int]
        Default: None
        Maximum number of instances to generate and evaluate. If None, we evaluate all instances.
    --is_strip_output=IS_STRIP_OUTPUT
        Type: bool
        Default: True
        Whether to strip trailing and leading whitespaces from the outputs.
    --is_load_outputs=IS_LOAD_OUTPUTS
        Type: bool
        Default: True
        Whether to try to load outputs from the output path. If True and outputs exist we only generate outputs for instructions that don't have outputs yet.
    -c, --chunksize=CHUNKSIZE
        Type: int
        Default: 64
        Number of instances to generate before saving. If None, we save after all generations.
    Additional flags are accepted.
        Other kwargs to `evaluate`

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

</details>

To evaluate a model you need to:

1. Choose an evaluation set and compute outputs specified as `model_outputs`. By default, we use
   the 805 examples from [AlpacaEval](#data-release). To compute outputs on AlpacaEval use:

```python
import datasets

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
for example in eval_set:
    # generate here is a placeholder for your models generations
    example["output"] = generate(example["instruction"])
    example["generator"] = "my_model" # name of your model
```

if your model is a HuggingFace model or from a standard API provider (OpenAI, Anthropic, Cohere). Then you can
directly use `alpaca_eval evaluate_from_model` to also take care of generating outputs.

2. Compute the reference outputs `reference_outputs`. By default, we use precomputed outputs of [`gpt4_turbo` on
   AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval).
   If you
   want to use a different model or a different dataset follow the same steps as (1.).
3. Choose an evaluator specified via `annotators_config`. We recommend using `alpaca_eval_gpt4_turbo_fn`. For other options and comparisons
   see [this table](#evaluators). Depending on the evaluator you might need to
   set the appropriate API_KEY in your environment
   or int the [client_configs](https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs).

Running all together:

```bash
alpaca_eval --model_outputs 'example/outputs.json' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'
```

If you don't have decoded outputs, you can use `evaluate_from_model` which takes care of decoding (model and reference)
for you.
Here's an
example:

```bash
# need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs 'oasst_pythia_12b' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'      
```

Here the `model_configs` and `reference_model_configs` (optional) are paths to a directory that specifies the prompt,
the model
provider (here HuggingFace) and decoding parameters.
See [this directory](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs) for examples.
For all model providers that are available out-of-the-box
see [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/decoders).

<details>
  <summary><b>Information about annotators</b></summary>

- **Caching**: by default all annotations are cached on
  disk at `caching_path`. Annotations are thus never recomputed, which makes annotations faster, cheaper and allow for
  reproducibility. This helps even when evaluating different models as many models
  have
  the same outputs.
- **Output randomization** by default, we randomize over the examples of outputs, as we found that annotators tend to
  prefer the first examples
  they see.
- **Batching** we provide code and examples to batch annotations, which decreases cost and time for annotations if the
  prompt is long. See for
  example [alpaca_farm_greedy_gpt4](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_farm_greedy_gpt4).
- **Pool of annotators** we provide code and examples to evaluate using a pool of automatic annotators, which is helpful
  for replicating the variance of [human annotations](https://arxiv.org/abs/2305.14387). See for
  example [alpaca_farm](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_farm).
- **Seeding based on instructions** For reproducibility and more fair comparison between models, we seed all
  randomness (output order, order in batches,
  examples for each annotator in a pool) based on the instruction.

</details>



<details>
  <summary><h2 tabindex="-1" dir="auto">Making a new leaderboard</h2></summary>

<details>
  <summary><code>>>> alpaca_eval make_leaderboard -- --help</code></summary>

```
NAME
    alpaca_eval make_leaderboard - Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

SYNOPSIS
    alpaca_eval make_leaderboard <flags>

DESCRIPTION
    Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

FLAGS
    --leaderboard_path=LEADERBOARD_PATH
        Type: Optional[Union]
        Default: None
        The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
    --annotators_config=ANNOTATORS_CONFIG
        Type: Union
        Default: 'alpaca_eval_gpt4_turbo_fn'
        The path the (or list of dict of) the annotator's config file.
    --all_model_outputs=ALL_MODEL_OUTPUTS
        Type: Union
        Default: <fu...
        The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe, datasets.Dataset) or a path to read those (json, csv, tsv potentially with globbing) or a function to generate those. If the path contains a globbing pattern, we will read all files matching the pattern and concatenate them. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction` and `output` with optional `input`. It should also contain a column `generator` with the name of the current model.
    -r, --reference_outputs=REFERENCE_OUTPUTS
        Type: Union
        Default: <func...
        The outputs of the reference model. Same format as `all_model_outputs` but without needing `generator`. By default, the reference outputs are the 003 outputs on AlpacaEval set.
    -f, --fn_add_to_leaderboard=FN_ADD_TO_LEADERBOARD
        Type: Callable
        Default: 'evaluate'
        The function to use to add a model to the leaderboard. If a string, it should be the name of a function in `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`, `precomputed_leaderboard`, `is_return_instead_of_print`, `reference_outputs`.
    --leaderboard_mode=LEADERBOARD_MODE
        Type: str
        Default: 'verified'
        The mode of the leaderboard to save all new entries with.
    -i, --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        Type: bool
        Default: False
        Whether to return the metrics instead of printing the results.
    Additional flags are accepted.
        Additional arguments to pass to `fn_add_to_leaderboard`.
```

</details>

If you want to make a new leaderboard using a single command (rather than multiple `alpaca_eval` calls), for your
desired evaluation
set and evaluators, you can use the following:

```bash
alpaca_eval make_leaderboard \
  --leaderboard_path <path_to_save_leaderboard> \
  --all_model_outputs <model_outputs_path> \
  --reference_outputs <reference_outputs_path> \
  --annotators_config <path_to_config.yaml>
```

where:

- `leaderboard_path`: path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists
  it will append.
- `all_model_outputs` : The json path to the outputs of all models to add to the leaderboard (as a single file or by
  globbing multiple files). Each dictionary should contain
  the keys (`instruction` and `output`) that are formatted in the prompts and a column `generator` with the name of the
  current model. As an example
  see [this file](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval_all_outputs.json).
- `reference_outputs` the path to the outputs of the reference model. Each dictionary should contain
  the keys (`instruction` and `output`) that are formatted in the prompts. By
  default, the reference outputs are the 003 outputs on AlpacaEval set.
- `annotators_config`: The path to the annotator's config file. Defaults to `alpaca_eval_gpt4`.

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Making a new evaluator</h2></summary>

<details>
  <summary><code>>>> alpaca_eval analyze_evaluators -- --help</code></summary>

```
NAME
    alpaca_eval analyze_evaluators - Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).

SYNOPSIS
    alpaca_eval analyze_evaluators <flags>

DESCRIPTION
    Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).

FLAGS
    --annotators_config=ANNOTATORS_CONFIG
        Type: Union
        Default: 'alpaca_eval_gpt4_turbo_fn'
        The path the (or list of dict of) the annotator's config file.
    -A, --Annotator=ANNOTATOR
        Default: <class 'alpaca_eval.annotators.pairwise_evaluator.PairwiseAn...
        The annotator class to use.
    --analyzer_kwargs=ANALYZER_KWARGS
        Type: Optional[Optional]
        Default: None
        Additional arguments to pass to the analyzer.
    -p, --precomputed_leaderboard=PRECOMPUTED_LEADERBOARD
        Type: Union
        Default: PosixPath('/Users/yanndubois/Desktop/GitHub/alpaca_eval/src/...
        The precomputed (meta)leaderboard of annotators or a path to it (json, csv, or tsv).
    --is_save_leaderboard=IS_SAVE_LEADERBOARD
        Type: bool
        Default: False
        Whether to save the leaderboard (ie analyzed results).
    --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        Type: bool
        Default: False
        Whether to return the leaderboard (ie analyzed results). If True, it will not print the results.
    --is_overwrite_leaderboard=IS_OVERWRITE_LEADERBOARD
        Type: bool
        Default: False
        Whether to overwrite the leaderboard if it already exists.
    -m, --max_instances=MAX_INSTANCES
        Type: Optional[Optional]
        Default: None
        The maximum number of instances to analyze.
    --is_single_annotator=IS_SINGLE_ANNOTATOR
        Type: bool
        Default: False
        Whether to analyze a single annotator. If True, will not be able to estimate the annotator's bias.
    -l, --leaderboard_mode_to_print=LEADERBOARD_MODE_TO_PRINT
        Type: str
        Default: 'minimal'
        The mode of the leaderboard to print.
    -c, --current_leaderboard_mode=CURRENT_LEADERBOARD_MODE
        Type: str
        Default: 'minimal'
        The mode of the leaderboard to save all new entries with.
    -o, --output_path=OUTPUT_PATH
        Type: Union
        Default: 'auto'
        Path to save the leaderboard and annotataions. If None, we don't save.
    Additional flags are accepted.
        Additional arguments to pass to `Annotator`.
```

</details>

AlpacaEval provides a simple way of making new evaluators. All you need is to make a new `configs.yaml` configuration
file, which you will then pass
as `--annotators_config <path_to_config.yaml>` to `alpaca_eval`.
Here are some ways you can make a new evaluator:

- **Changing the prompt**: Write a new prompt in a text file and specify the path in `prompt_template` of the
  configuration file. Paths are relative to the configuration file.
- **Changing decoding parameters**: Specify the desired parameters in `completions_kwargs` in the configuration file. To
  see all available parameters refer to the docstrings of the corresponding
  function [in this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py)
  specified by `fn_completions`
  in the configuration file.
- **Changing the model**: Specify the desired model in `model_name` and the corresponding
  prompt in `prompt_template`. If the model comes from another provider you
  will
  have
  to change `fn_completions` which maps to the corresponding function
  in [this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py). We
  provide `fn_completions` functions to use models from OpenAI, Anthropic, Cohere, or HuggingFace. To
  install packages needed for
  all providers
  use `pip install alpaca_eval[all]`.


<details>
  <summary><b>Other parameters in the configuration file</b></summary>

The easiest is to check the docstrings
of [`SinglePairwiseAnnotator`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/annotators/pairwise_evaluator.py#L537).
Here are some important ones:

```
Parameters
----------
prompt_template : path
    A prompt that will be given to `fn_prompter` or path to the prompts. Path is relative to
    `evaluators_configs/`

fn_completion_parser : callable or str
    Function in `completion_parsers.py` to use for parsing the completions into preferences. For each completion,
    the number of preferences should be equal to the batch_size if not we set all the preferences in that batch to
    NaN.

completion_parser_kwargs : dict
    Kwargs for fn_completion_parser.

fn_completions : callable or str
    Function in `decoders.py` to use for decoding the output.

completions_kwargs : dict
    kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

is_randomize_output_order : bool
    Whether to randomize output_1, output_2 when formatting.

batch_size : int
    Number of examples that will be added in a single prompt.
```

</details>

Once you made the evaluator you can also analyze it and add it to the _evaluator's_ [leaderboard](#evaluators) using the
following command:

```bash
alpaca_eval analyze_evaluators --annotators_config '<path_to_config.yaml>'    
```

To estimate the bias and variance this evaluates every example with 4 seeds, i.e., 2.5K
evaluation.
If you want a cheaper evaluation you can use a single seed using `--is_single_annotator True` which will skip the
estimation of bias and variance.

</details>

# Contributing

We are accepting PRs for new models, evaluators, and eval sets, in addition to bug fixes.
We will update the [leaderboard website](https://tatsu-lab.github.io/alpaca_eval/) regularly with new community
contributions.
We have also created a [support discord](https://discord.gg/GJMxJSVZZM) for AlpacaEval in case you run into any issues
and
wish to ask help from the community.

To get started, please first fork the repo, and install the package from source `pip install -e .`

## Contributing a model

First, you'll need to add a model config definition in the [models_configs](src/alpaca_eval/models_configs/) folder. As
an example, you can look at
the [falcon-7b-instruct yaml](src/alpaca_eval/models_configs/falcon-7b-instruct/configs.yaml). Please make sure the
folder name and key name in the yaml match exactly.

Then, please follow the steps in [Evaluating a model](#evaluating-a-model) to run inference on the model to produce
outputs on the eval set and score the model according to one of the evaluators.
An example command may look like:

```sh
alpaca_eval evaluate_from_model \
  --model_configs 'falcon-7b-instruct'
```

After running this command, you should have generated an outputs json and a new entry in the corresponding [leaderboard
file](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/leaderboards/data_AlpacaEval). Please make a PR
with the
config, outputs file, and updated leaderboard.

Concretely you should do something like:

1. Fork the repository in github
2. Clone the forked repository `git clone <URL>`
3. Make a model config at `src/alpaca_eval/models_configs/<model_name>` and evaluate it `evaluate_from_model --model_configs '<model_name>'`
4. Add the model configs, output, and leaderboard entry to the forked repository
```sh
git add src/alpaca_eval/models_configs/<model_name> # add the model config
git add src/alpaca_eval/leaderboards/ # add the actual leaderboard entry
git add src/alpaca_eval/metrics/weights # add the weights for LC
git add -f results/<model_name>/model_outputs.json # force add the outputs on the dataset
git add -f results/<model_name>/*/annotations.json # force add the evaluations from the annotators
git commit -m "Add <model_name> to AlpacaEval"
git push
``` 
5. Create a [pull request on AlpacaEval](https://github.com/tatsu-lab/alpaca_eval/pulls)

Note: if you are generating outputs outside of AlpacaEval you should still add a model config but with `fn_completions: null`. 
See [this config](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/dolphin-2.2.1-mistral-7b/configs.yaml) for an example.

<details>
  <summary><h3 tabindex="-1" dir="auto">Getting your model verified</h3></summary>

<p align="center">
<img align="center" alt="verified.png" src="figures/verified.png" width="500"/>
</p>

A verified result in AlpacaEval indicates that a core maintainer has decoded the outputs from the model and performed the evaluation. Unfortunately, we, the AlpacaEval maintainers, lack the resources to verify all the models and so we will only do that for models that are in the top-5 of the leaderboard. We apologize for any inconvenience this may cause and appreciate your understanding. To have your model verified, please follow the steps below:

1. Contact `@yann`  on Discord, or email us if you have our email, providing a brief rationale for why your model should be verified.
2. Await our response and approval before proceeding.
3. Prepare a script to decode from your model that does not require a GPU, typically the same script used for your model contribution. It should run using `alpaca_eval evaluate_from_model --model_configs '<your_model_name>'` without requiring a local GPU.
4. Generate temporary API keys for running the script and share them with us. Specifically, we need the keys for both decoding your model and for evaluation (e.g., OpenAI or Anthropic key).
5. We will execute `alpaca_eval evaluate_from_model --model_configs '<your_model_name>'`, update the results, and inform you so that you can revoke the temporary keys.

Note that we will not re-evaluate the same model. Due to sampling variance, the results might slightly differ from your initial ones. We will replace your previous community results with the verified ones. 


</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Contributing an evaluator</h2></summary>

Please first follow the directions in [Making a new evaluator](#making-a-new-evaluator).
Once you're created the annotator config, we ask that you create a new leaderboard for the annotator by evaluating the
minimal set of models. The outputs for these models can be found by
downloading [alpaca_eval_all_outputs.json](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval_all_outputs.json).

```bash
alpaca_eval make_leaderboard \
  --leaderboard_path src/alpaca_eval/leaderboards/data_AlpacaEval/<evaluator>_leaderboard.csv \
  --all_model_outputs alpaca_eval_all_outputs.json \
  --annotators_config <evaluator_config>
```

Then, please create a PR with the annotator config and leaderboard csv.

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Contributing an eval set</h2></summary>

To contribute a new eval set, you'll first need to specify a set of textual instructions.
Then, you'll need to specify a set of reference outputs (model win-rates are computed against this reference).
For ease of use, you may use the default [text-davinci-003](src/alpaca_eval/models_configs/text_davinci_003/) reference
config.

Place these together into a json, where each entry specifies the fields `instruction`, `output`, and `generator`. You
can look to [alpaca_eval.json](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json) as a
guide (the `dataset` field is not necessary).

Finally, we ask that you create a minimal leaderboard on this new evaluation set. You can do this with the following:

```bash
alpaca_eval make_leaderboard \
  --leaderboard_path <src/alpaca_eval/leaderboards/data_AlpacaEval/your_leaderboard_name.csv> \
  --all_model_outputs alpaca_eval_all_outputs.json \
  --reference_outputs <path_to_json_file>
```

Please submit a PR with the eval set json and corresponding leaderboard csv.


</details>



<details>
  <summary><h2 tabindex="-1" dir="auto">Contributing a completion function</h2></summary>

Currently, we allow different completion functions, e.g., `openai`, `anthropic`, `huggingface_local`, `huggingface_hub_api` ... If you want to contribute a new completion function / API with which to perform inference then follow those steps:
1. add a file <name>.py with a function  `<name>_completions(prompts : Sequence[str], model_name :str, ... )`  in the [decoder folder](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/decoders). This function should take as argument the prompts + kwargs and return the completions. Please look at other completion functions in the directory for templates. E.g. [huggingface_local_completions](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/huggingface_local.py) or [anthropic](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/anthropic.py).
2. add `<name>_completions` and dependencies in [__init__](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py) . Again you can follow the example of [huggingface_local_completions](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py#L30)
3. update optional dependencies in [setup.py](https://github.com/tatsu-lab/alpaca_eval/blob/main/setup.py)
4. add a model you want to evaluate in the [models configs](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/models_configs)
5. evaluate your model using `alpaca_eval evaluate_from_model --model_configs '<model_configs>'`
6. (optional) push the results from the previous model on AlpacaEval leaderboard following [those steps](https://github.com/tatsu-lab/alpaca_eval/tree/main#contributing-a-model)

Feel free to start a PR early, we'll be able to provide some help in the process! 

</details>

# Limitations

The AlpacaEval evaluation pipeline, like other current evaluators have important limitations and should therefore not be
used as replacement for human evaluation in important settings, such as to decide whether a model is ready to be
deployed.
Those can broadly be clustered into 3 categories:

1. **Instructions might not be representative of real-usage**:  the AlpacaEval set contains examples from a variety of
   datasets ([self-instruct](https://github.com/yizhongw/self-instruct),
   [open-assistant](https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/OpenAssistant--oasst1/validation), [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [koala](https://github.com/arnav-gudibande/koala-test-set), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/Anthropic--hh-rlhf/test))
   which might not be representative of real-usage and advanced applications of better models like GPT4. This likely makes the best closed models (GPT4 / Claude / ChatGPT / ...) seem more similar to the open models than what they are. Indeed, those closed models seem to be pretrained/finetuned on much more diverse data. See for
   example [this blog](https://medium.com/@marcotcr/exploring-chatgpt-vs-open-source-models-on-slightly-harder-tasks-aa0395c31610)
   for preliminary results on more complex instructions.
   Note, however, that in [AlpacaFarm](https://arxiv.org/abs/2305.14387) we showed that win-rates on our evaluation set
   are highly correlated (0.97 R2) with win-rates on instructions from user interactions with the Alpaca Demo.
   Furthermore, the AlpacaEval leaderboard shows larger
   gap between the open models and OpenAI models than other leaderboards (
   e.g. [lmsys](https://lmsys.org/blog/2023-03-30-vicuna/)).

2. **Biases of automatic annotators**: the raw automatic annotators seem to have implicit biases. In particular, we found
   that they tend to prefer longer outputs and outputs that contain lists (e.g. 0.68 / 0.69 for `alpaca_eval_gpt4`
   and 0.62 / 0.58 for `claude`).
   Although we found that humans have similar biases (0.64 / 0.61), we believe that this could be more of a limitation
   of human annotation pipeline we used rather than a true human bias. More generally, through qualitative analysis, we
   found that automatic annotators give more importance to the style
   of the output than its content (e.g. factuality).
   Finally, we found that automatic evaluators tend to prefer outputs from models that are similar (likely trained on
   the same data) as suggested by the big difference between ChatGPT/GPT4 on `claude`'s and `alpaca_eval_gpt4`'s
   leaderboard. Note that the length bias is partially mitigated in our length-controlled win-rates.
3. **Lack of safety evaluation**: importantly, AlpacaEval only evaluates the instruction-following capabilities of
   models rather than the harm that they could cause (e.g. toxic behavior or bias). As a result the small gap between
   current ChatGPT and the best open source models **should not** be interpreted as if that the latter are ready to be
   deployed.

Beyond those limitations about the evaluation pipelines, there are also limitations about our validation of the
evaluators and our [proposed approach](#analyzing-an-eval-set) to selecting evaluation sets.

<details>
  <summary><b>Limitations about our validation pipeline</b></b></summary>

First, our validation of evaluators based on human cross-annotations suffers from the following limitations: (1) we
qualitatively found that our crowd-workers tend to also favor style such as length and presence of lists over
factuality;
(2) this does not validate whether win-rates against a reference model is a good evaluation strategy in the first place;
(3) preferences from 16 crowd-workers are not representative of preferences of all humans.

Second, our suggested approach to selecting evaluation sets based on statistical power suffers from the following
limitations: (1) statistical power does not ensure the right direction, e.g. you can have an unnatural set of
instructions where Alpaca "performs" better than better model; and
(2) this can push users to select data to support the hypothesis that they want to validate.

</details>


# Additional analysis and plots


[//]: # (AlpacaEval provides a few visualization tools to help you analyze and improve your automatic evaluation pipeline. We)

[//]: # (briefly explain)

[//]: # (them here and provide)

[//]: # (notebooks for more analysis. )

[//]: # (For a description of all the metrics we consider)

[//]: # (refer to [How exactly are those metrics computed?]&#40;https://github.com/tatsu-lab/alpaca_eval#evaluators&#41;)

## Length-controlled AlpacaEval (LCAE)


**Length-controlled AlpacaEval Visualizations:**
[![analyzing an evaluator](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/figured_length_controlled.ipynb)

**Length-controlled AlpacaEval Development:**
[![analyzing an evaluator](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/length_controlled.ipynb)

The notebook shows different options that we considered for mitigating the length bias of automatic annotators. 

Here we briefly summarize the main results. Namely:
- **LCAE increases the correlation with Chat Arena to 0.98** from 0.94 for AlpacaEval 2.0. This makes LCAE the most highly correlated benchmark with Chat Arena as seen in the plot below.


<p float="left" align="middle">
<img src="figures/chat_correlations.png" alt="LC AlpacaEval is the most highly correlated benchmark with Chat Arena." width="500"/>
</p>

- **LCAE decreases length gameability** one of the major issues of AlpacaEval is that you can increase your win-rate by increasing the length of your outputs. For example, in AlpacaEval 2.0 the win-rate for the baseline (50%) increases to 64% when prompted to “give as much detail as possible” and decreases to 23% when prompted to “be as concise as possible while still providing all the necessary information to answer the question”. More generally the relative length gameability was ~21% for AlpacaEval and decreases to ~6% for LCAE, so it's 3x less gameable through prompt length. This is shown in the plot below.  


<p float="left" align="middle">
<img src="figures/length_gameability.png" alt="LC AlpacaEval decreases length gameability of the benchmark." width="500"/>
</p>

- **We can predict performance for different baselines** One other benefit of using a GLM for controlling for length bias. Is that we now have a model that can predict the win-rate of a model for different baselines. In particular, our GLM has many nice properties, for example `win_rate(m,b) = 1 - win_rate(b,m) \in [0,1]` and `win_rate(m,m) = 0.5`. This is shown in the plot below.


<p float="left" align="middle">
<img src="figures/different_baselines.png" alt="Predicted win rate for different baselines" width="500"/>
</p>


Finally, note that we are only controlling for length bias. There are other known biases that we are not controlling for, such as the fact that auto-annotators prefer outputs similar to their model. Although we could control for that, in practice we have found that to be less of an issue than length bias. For two reasons (1) this mostly a single model in the leaderboard because fine-tuning on outputs from the auto-annotator doesn't seem to have doesn't seem to impact the win-rate as much, and (2) the bias is actually less strong that what one could think. For example we show below a subset of the leaderboards auto-annotated by three different models, and we see that the ranking of models is exactly the same. In particular, `claude-3-opus` prefers `gpt4_preview`, and `mistral-large` prefers the former two.

<p float="left" align="middle">
<img src="figures/annotator_bias.png" alt="Leaderboard by different auto-annotators" width="500"/>
</p>

<details>
  <summary><h2 tabindex="-1" dir="auto">Analyzing an evaluator</h2></summary>

[//]: # (## Analyzing an evaluator)

**Caution**: all the following results are about AlpacaEval 1.0 and have not been updated since

**Analyzing evaluators:**
[![analyzing an evaluator](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_annotators.ipynb)

As we saw in [the evaluator's leaderboard](#evaluators), there are many metrics to consider when selecting an evaluator,
e.g. the quality, price, and speed. To assist with selection of the evaluator we provide a few functions to plot those
metrics.
The following shows for example the price/time/agreement of the different evaluators.

![plot_quality_vs_price_and_time.png](figures%2Fplot_quality_vs_price_and_time.png)

Here we see that `alpaca_eval_gpt4` performs very well and is better than humans on all the considered metrics.

Previously we only considered the agreement with human annotators overall.
An additional validation that one could do is checking whether making a leaderboard using our
automatic annotator gives similar results as a leaderboard from humans.
To enable such analysis, we release [human
annotations](#data-release) of outputs from 22 methods from [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) =>
22*805 = ~18K annotations. As a result we
can
test
the correlation between the win-rates of the 22 models as evaluated by the humans and our automatic annotator.
Note that this is arguably a better way of selecting an automatic evaluator than using "human agreement [%]" but is
expensive given that it requires 18K
annotations.
The plot below shows such correlation for the `alpaca_eval_gpt4` evaluator.

<p float="left" align="middle">
<img src="figures/plot_winrate_correlations_alpaca_eval.png" alt="Correlation between humans and alpaca_eval_gpt4" width="400"/>
</p>

We see that the `alpaca_eval_gpt4` leaderboard is highly correlated (0.94 Pearson correlation) to the leaderboard from
humans, which further
suggests that automatic evaluation is a good proxy for human evaluation.
For the code and more analysis,
see [this notebook](https://github.com/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_annotators.ipynb), or the
colab notebook above.

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Analyzing an eval set</h2></summary>

[//]: # (## Analyzing an eval set)

**Caution**: all the following results are about AlpacaEval 1.0 and have not been updated since.

**Making evaluation sets:**
[![analyzing an evaluator](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_evalset.ipynb)

When creating an evaluation set there are two main factors to consider: how much data to use? and what data?

One way of answering those question is by considering a leaderboard of models that you believe are of different
quality and checking what and how much data is needed to distinguish between them in a statistically significant way.
We will do so below using a paired t-test to test if the difference in win-rates between every pair of models
is
statistically significant.

First, let us consider the question of how much data to use.
Below we show the number of random samples needed from AlpacaEval for the paired t-test to give a p-value < 0.05 for
each pair of models in the minimal `alpaca_eval_gpt4`
leaderboard.
Grey cells correspond to pairs that are not significantly different on the 805 samples.
y- and x-axis are ordered by the win-rate of the first and second model respectively.


<p float="left" align="middle">
<img src="figures/plot_paired_ttest_nsamples.png" alt="Number of samples needed to distinguish pairs in the Claude leaderboard" width="500"/>
</p>

We see that most models can already be distinguished with 50 samples, and that 150 samples allows distinguishing the
majority of pairs (74 out of 78). This suggests that we can decrease the evaluation set size by a factor of
4 when testing two models that have similar performance gaps as those on the
minimal `alpaca_eval_gpt4` [leaderboard](#models).

The second question is what data to use. Again we can try to answer this question from a statistical power perspective:
what data allows to best distinguish between models. Let's consider this for all the datasets that are part of
AlpacaEval, but let us control for the size of the evaluation sets as we only care about the quality of the data. The
following plot shows the p-values from the paired t-test of each pairs of models on 80 examples of each subset of
AlpacaEval.

![plot_paired_ttests_per_dataset.png](figures%2Fplot_paired_ttests_per_dataset.png)

We see for example that the self-instruct dataset yields the least statistical power, which suggests that one could
remove this dataset from the evaluation set.
The exact reason should be analyzed in future work.
For the code and more analysis
see [this notebook](https://github.com/tatsu-lab/alpaca_eval/blob/main/notebooks/analyzing_evalset.ipynb), or the
colab notebook above.

</details>

# Citation

Please consider citing the following depending on what you are using and referring to:
- **Code, results, and general benchmark**: `alpaca_eval` (this repo). Specify whether you are using AlpacaEval or AlpacaEval 2.0. For length-controlled win-rates see below.
- **Length-controlled (LC) win rates**: `alpaca_eval_length`.
- **Human annotations**: `dubois2023alpacafarm` ([AlpacaFarm](https://arxiv.org/abs/2305.14387))
- **AlpacaEval evaluation set**: `alpaca_eval`  and [self-instruct](https://github.com/yizhongw/self-instruct),
[open-assistant](https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/OpenAssistant--oasst1/validation), [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [koala](https://github.com/arnav-gudibande/koala-test-set), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/Anthropic--hh-rlhf/test).

Here are the bibtex entries:

```
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  month = {5},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}
```

```
@article{dubois2024length,
  title={Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators},
  author={Dubois, Yann and Galambosi, Bal{\'a}zs and Liang, Percy and Hashimoto, Tatsunori B},
  journal={arXiv preprint arXiv:2404.04475},
  year={2024}
}
```

```
@misc{dubois2023alpacafarm,
  title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
  author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
  year={2023},
  eprint={2305.14387},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

# More information

<details>
  <summary><h2 tabindex="-1" dir="auto">Length-Controlled Win Rates</h2></summary>

Length controlled (LC) win-rates are a debiased version of the win-rates that control for the length of the outputs.

The main idea is that for each model we will fit a logistic regression to  predict the preference of the autoannotator given: (1) the instruction, (2) the model, and (3) the difference of length between the baseline and model output. 
Given such a logistic regression we can then try to predict the counterfactual "what would the preference be if the model's output had the same length as the baseline" by setting the length difference to 0.
By averaging over this length-controlled preference, we then obtain the length-controlled win-rate.
The exact form of the logistic regression is taken such that the interpretation of LC win rates is similar to the raw win rates, for example for any model `m1` and `m2` we have `win_rate(m1, m2) = 1 - win_rate(m2, m1) \in [0,100]` and `win_rate(m1, m1) = 0.5`. 
Length controlled win-rates increase the correlation between AlpacaEval's leaderboard and Chat Arena from **0.93 to 0.98 Spearman correlation, while significantly decreasing the length gameability of the annotator**.
For more information and results about length controlled win-rates see [this notebook](https://github.com/tatsu-lab/alpaca_eval/blob/main/notebooks/length_controlled.ipynb).

This idea of estimating the controlled direct effect, by predicting the outcome while conditioning on the mediator (the length difference), is common in statistical inference.

To get LC win rates on previously annotated models, you can use the following command:

```bash
pip install -U alpaca_eval
alpaca_eval --model_outputs … --is_recompute_metrics_only True
```

</details>


<details>
  <summary><h2 tabindex="-1" dir="auto">AlpacaEval 2.0</h2></summary>

AlpacaEval 2.0 is a new version of AlpacaEval. Here are the differences:
- **reference: `gpt4_turbo`**: we upgraded the baseline from `text-davinci-003` to `gpt4_turbo` to make the benchmark more challenging and have a metric that better reflects the current state of the art.
- **annotator: `weighted_alpaca_eval_gpt4_turbo`**: we improved the annotator in quality and price. First, we use the `gpt4_turbo` model for annotating, which is approximately 2x cheaper than `gpt4`. Second, we changed the prompt such that the model outputs a single token, which further reduced cost and speed. Finally, instead of using a binary preference, we used the logprobs to compute a continuous preference, which gives the final weighted win-rate. Note that the latter two changes had the surprising effect of decreasing the annotators' length biased.

By default, AlpacaEval 2.0 will be used from `pip install alpaca_eval==0.5`. If you wish to use the old configs by default, you can set `IS_ALPACA_EVAL_2=False` in your environment.

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Data Release</h2></summary>

As part of AlpacaEval, we release the following data:

- **Human annotations (17701)** in order to develop and understand automatic evaluators, we release all the human
  pairwise
  evaluation that we collected for AlpacaFarm. This contains comparisons between 22 models with the `text-davinci-003`
  reference on the AlpacaFarm evaluation set. Annotations are from a pool of 16 crowd workers on Amazon Mechanical Turk.
  The different models are: 6 from OpenAI, 2 SFT models from AlpacaFarm, 13 RLHF methods from AlpacaFarm, and LLaMA 7B.
- **Human cross-annotations (2596)** in order to further analyze automatic evaluators we selected (via stratified
  sampling
  across models and datasets) 650 examples from the AlpacaFarm evaluation set and collected 4 human annotations per
  example.
- **AlpacaEval set (805)** we made slight modifications/simplification of the AlpacaFarm evaluation set. In particular,
  we first merged
  the instruction and input fields into a single instruction field. This affects 1/4 of the examples in the AlpacaFarm
  evaluation set, all of which are from the [self-instruct evaluation set](https://arxiv.org/abs/2212.10560). Second we
  regenerated the text-davinci-003 reference outputs without limiting the length of its outputs.

For more details about the human annotations refer to the [AlpacaFarm paper](https://arxiv.org/abs/2305.14387).

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Differences with AlpacaFarm</h2></summary>

AlpacaEval is an improvement and simplification of the automatic pairwise preference simulator
from [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm).
Outside AlpacaFarm, you should be using AlpacaEval.
Here are the main differences:

- **AlpacaEval merges instructions and inputs**: The AlpacaEval evaluation is the same as the AlpacaFarm evaluation
  except that the instruction and input fields are merged as `{instruction}\n\n{input}`. This affects 1/4 of the
  examples in the AlpacaFarm evaluation set (the [self-instruct](https://arxiv.org/abs/2212.10560) subset).
  This simplification provides a more fair comparison for models that were not trained by distinguishing between
  the two fields.
- **AlpacaEval handles longer generations**: Models in AlpacaFarm were limited to a maximum number of 300 tokens for
  generations. We
  change this number to 2000 for AlpacaEval. Note that this also affects the reference generations (`text-davinci-003`),
  so the results on AlpacaEval are not comparable to those on AlpacaFarm even for examples that had no input
  field.
- **AlpacaEval removes intra- and inter-annotator variance**: The AlpacaFarm simulator replicates human annotation in
  terms of both mode behavior and diversity.
  In particular, AlpacaFarm's simulator uses a pool of models and prompts and adds noise to replicate human intra- and
  inter-annotator variance.
  If the goal is to use an automatic annotator for evaluation or simply training better models, then this variance
  may not be desirable. The default annotators in AlpacaEval thus don't have this variance. We give the option to add it
  back by
  using `--anotators_config 'alpaca_farm'` and `--p_label_flip 0.25` when creating an evaluator.

</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Related work</h2></summary>

There have been several work that propose new automatic annotators for instruction-following models. Here we list the
ones that we are aware of and discuss how they differ from ours. We evaluated all of those
in [our evaluator's leaderboard](https://github.com/tatsu-lab/alpaca_eval#evaluators).

- **Vicuna/lmsys** The lmsys annotator (`lmsys_gpt4`) evaluates the pair by asking the annotator a score from 1-10 for
  each output, and then selecting the output with the highest score as preferred. They do not randomize over output
  order and they ask an explanation _after_ the score. Overall, we found that this annotator has strong bias towards
  longer outputs (0.74) and relatively low correlation with human annotations (63.2).
- **AlpacaFarm** The best AlpacaFarm annotator (`alpaca_farm_greedy_gpt4`) evaluates the pair by directly asking the
  annotator
  which output it prefers. Furthermore, it batches 5 examples together to amortize the length of the prompt and
  randomizes the order of outputs. Overall, we
  found that this annotator has much less bias towards longer outputs (0.60) and is faster (878 seconds/1000 examples)
  than others. It has a
  slightly higher correlation with the majority of human annotations (66.4) than humans themselves (65.7).
  However, it is more expensive ($15.3/1000 examples) and doesn't work with very long outputs given the batching.
- **Aviary** The Aviary annotator (`aviary_gpt4`) asks the annotator to order the output by its preference, rather than
  simply selecting the preferred output. It does not randomize the order of outputs and uses high temperature for
  decoding (0.9). Overall, we found that this annotator has relatively strong bias towards longer outputs (0.70) and
  very high
  correlation with human annotations (69.1). By decreasing the temperature and randomizing the order of outputs,
  we [further improved](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md)
  the correlation to 69.8 (`improved_aviary_gpt4`) but this further increased the length bias to 0.73.

Our `alpaca_eval_gpt4` is a mix between the AlpacaFarm and Aviary annotators. It asks the annotator to order the outputs
by preference, but it uses temperature 0, randomizes over outputs, and made some modifications to the prompt to decrease
length bias to 0.68.

Other related work include recent papers which analyze automatic evaluators.
For example:

- [AlpacaFarm Appx C](https://arxiv.org/abs/2305.14387)
  and [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926v1) both found that automatic
  annotators have
  a position bias.
- [AlpacaFarm Sec. 5.2.](https://arxiv.org/abs/2305.14387)
  and [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717) both found that
  automatic
  annotators favor style (e.g. use of list, tone, word choice, length) over factuality.

</details>


<details>
  <summary><h2 tabindex="-1" dir="auto">Interpreting annotations</h2></summary>

For all models you can find the auto-annotations under `results/<model_name>/*/annotations.json`. The annotations have the following columns:
- `instruction`: the prompt
- `generator_1`: the baseline model
- `output_1`: the output of the baseline model
- `generator_2`: the model being evaluated
- `output_2`: the output of the model being evaluated
- `annotator`: the auto-annotator
- `preference`: the result of the auto-annotator. This is a float between 1 and 2. Closer to 1 means that the auto-annotator prefers `output_1`, closer to 2 means that it prefers `output_2`. For AlpacaEval 2.0, `preference-1` corresponds to the probability of `output_1` being preferred. For AlpacaEval 1.0, `preference` is 1 if `output_1` is preferred, 2 if `output_2` is preferred, and 1.5 if they are the same. The win rate is always`(preference -1).mean()`.
- `raw_completion`: the raw output of the auto-annotator. This is field contains the completions before de-randomization of the order between `output_1` and `output_2`! It is thus much harder to interpret, see below for more information.

**Chain of thought**

For some annotators, e.g. `alpaca_eval_cot_gpt4_turbo_fn` we use **chain of thought reasoning** to make the models preferences more interpretable. Those can then be found under `concise_explanation`. To interpret them, you should also look at `referenced_models` which translates the temporary model name (in the prompt) to the actual output. Below, we provide more explanation as to what is happening behind the scenes.

You can check the `raw_annotations["concise_explanation]` column in `annotations.json` (e.g. [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/results/gpt4/alpaca_eval_cot_gpt4_turbo_fn/annotations.json)) which contains the chain of thought reasoning of the auto annotator. Note that the raw_annotations is not modified by the randomization of the order of the outputs. In particular, `"m"` and `"M"` can sometime refer to the first model (the reference) and sometime to the second model (the model being evaluated). To understand which model is being referred to, you should use the column `preference` and `ordered_models`. To make it easier we add a column `"referenced_models"` mapping the model names to the corresponding outputs. For example in the following annotation we see that the preference is 1.0 (i.e. `output_1`) and corresponds to model `M` in `concise_explanation` (see `ordered_models`).  

```json 
{
  "instruction": "How did US states get their names?",
  "output_1": "The names of U.S. states are derived from a variety of languages and sources, including indigenous American languages, European languages (primarily English, Spanish, and French), as well as other historical influences. Here is a brief overview of how some states got their names:\n\n1. **Indigenous American Languages**: Many state names come from Native American words or names of tribes, often from the languages of the indigenous peoples who lived in or near those areas.\n   - **Alabama**: Possibly derived from the Choctaw language, meaning \"thicket clearers.\"\n   - **Connecticut**: From a Mohegan-Pequot word meaning \"long tidal river.\"\n   - **Massachusetts**: [...]",
  "generator_1": "gpt4_1106_preview",
  "dataset": "helpful_base",
  "output_2": "The names of the 50 U.S. states come from a variety of sources, including Native American languages, European languages, and historical figures. Here's a brief overview of how some states got their names:\n\n1. Native American origins: Many states have names derived from Native American languages. For example, Alabama comes from the Choctaw word \"Albah amo,\" meaning \"plant gatherers\" or \"herb gatherers.\" Similarly, the name Mississippi comes from the Ojibwe word \"Misi-ziibi,\" meaning \"great river.\"\n\n2. European languages: [...].",
  "generator_2": "gpt4",
  "annotator": "alpaca_eval_cot_gpt4_turbo_fn",
  "preference": 1.0,
  "raw_completion": {
    "concise_explanation": "Model M provided a more detailed and structured response, including bold headings for each category and a wider range of examples. It also included additional categories such as 'Other European Languages' and 'Combination of Languages and Influences', which added depth to the explanation. Model m's response was accurate but less comprehensive and lacked the clear structure found in Model M's output.",
    "ordered_models": [
      {
        "model": "M",
        "rank": 1
      },
      {
        "model": "m",
        "rank": 2
      }
    ]
  },
  "referenced_models": {
    "M": "output_1",
    "m": "output_2"
  }
}
```


</details>

<details>
  <summary><h2 tabindex="-1" dir="auto">Major updates</h2></summary>

- 12th March 2024: updated to use length-controlled (LC) win rates. This is a debiased version of the win-rates that control for the length of the outputs. 
- 3rd January 2024: updated to AlpacaEval 2.0, which uses GPT4-turbo as baseline and annotator.
- 2nd January 2024: added Azure API and more general way of setting client configs. See [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs/README.md)
- 19th June 2023: add leaderboard `chatgpt_fn` that anyone can use (no waiting lists).
- 19th June 2023: update to
  use [OpenAI's function calling](https://openai.com/blog/function-calling-and-other-api-updates).
  Example: [`chatgpt_fn`](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/chatgpt_fn)
  or [`alpaca_eval_gpt4_fn`](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4_fn).

</details>
