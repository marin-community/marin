# Annotators configs

## Evaluator's leaderboard:

Here's the full leaderboard estimated on 4 different seeds, which allows us to also estimate the variance of the
annotators.
We compute those metrics on our suggested evaluator `alpaca_eval_gpt4`, on prior
evaluators (`aviary_gpt4`, `lmsys_gpt4`, `alpaca_farm_greedy_gpt4`), and on different base models with which we use
essentially the same prompt (`gpt4`, `text_davinci_003`, `claude`, `chatgpt`).
We also provide partial metrics (only 1 seed) for other evaluators, which include our evaluator using OpenAI's function
calls (`alpaca_eval_gpt4_fn`), prior work that we
improved (`improved_aviary_gpt4` and `improved_lmsys_gpt4`), prior work that was not meant to be used as a final
evaluator (`guanaco_33b`), and a ranking evaluator (`alpaca_farm`), and secondary models that use the same prompt as the
models above (`cohere`, `guanaco_33b`):

|                                     |   Human agreement |   Price [$/1000 examples] |   Time [seconds/1000 examples] |   Spearman corr. |   Pearson corr. |   Bias |   Variance |   Proba. prefer longer |   Proba. prefer lists |   Proba. prefer 1 |   # parsed | mode     |
|:------------------------------------|------------------:|--------------------------:|-------------------------------:|-----------------:|----------------:|-------:|-----------:|-----------------------:|----------------------:|------------------:|-----------:|:---------|
| alpaca_eval_gpt4_fn                 |              71.0 |                      14.5 |                           5046 |             0.95 |            0.94 |   27.6 |       11.1 |                   0.75 |                  0.68 |              0.48 |       2592 | verified |
| improved_aviary_gpt4                |              69.8 |                      12.8 |                           1831 |             0.88 |            0.90 |        |            |                   0.73 |                  0.70 |              0.49 |        648 | verified |
| alpaca_eval_gpt4                    |              69.2 |                      13.6 |                           1455 |             0.97 |            0.93 |   28.4 |       14.6 |                   0.68 |                  0.73 |              0.50 |       2592 | minimal  |
| alpaca_eval_clf_cot_gpt4_turbo      |              68.7 |                       6.4 |                           1753 |             0.93 |            0.76 |        |            |                   0.69 |                  0.65 |              0.54 |        639 | verified |
| alpaca_eval_cot_gpt4_turbo_fn       |              68.6 |                       6.3 |                           1989 |             0.97 |            0.90 |   29.3 |       18.4 |                   0.67 |                  0.61 |              0.52 |       2586 | minimal  |
| weighted_alpaca_eval_cot_gpt4_turbo |              68.5 |                       6.4 |                           1869 |             0.93 |            0.77 |        |            |                   0.69 |                  0.66 |              0.53 |        647 | verified |
| aviary_gpt4                         |              68.4 |                      12.8 |                           1821 |             0.92 |            0.91 |        |            |                   0.70 |                  0.65 |              0.56 |        648 | verified |
| alpaca_eval_gpt4_turbo_fn           |              68.1 |                       5.5 |                            864 |             0.93 |            0.82 |   30.2 |       15.6 |                   0.65 |                  0.60 |              0.54 |       2592 | minimal  |
| gpt4_turbo_cot_logprob              |              67.9 |                       5.4 |                           1569 |             0.63 |            0.63 |        |            |                   0.59 |                  0.59 |              0.53 |        648 | verified |
| gpt4_turbo_cot_clf                  |              67.6 |                       5.4 |                           1528 |             0.67 |            0.63 |        |            |                   0.59 |                  0.59 |              0.53 |        645 | verified |
| claude_ranking                      |              67.6 |                       5.0 |                            218 |             0.90 |            0.91 |        |            |                   0.73 |                  0.66 |              0.46 |        648 | verified |
| gpt4                                |              66.9 |                      12.5 |                           1037 |             0.88 |            0.87 |   31.5 |       14.6 |                   0.65 |                  0.67 |              0.54 |       2592 | minimal  |
| alpaca_farm_greedy_gpt4             |              66.4 |                      15.3 |                            878 |             0.85 |            0.75 |   30.2 |       19.3 |                   0.60 |                  0.65 |              0.54 |       2592 | minimal  |
| weighted_alpaca_eval_gpt4_turbo     |              65.7 |                       4.3 |                            228 |             0.78 |            0.77 |   33.9 |       23.7 |                   0.61 |                  0.57 |              0.53 |       2592 | verified |
| humans                              |              65.7 |                     300.0 |                          36800 |             1.00 |            1.00 |    0.0 |       34.3 |                   0.64 |                  0.60 |              0.52 |       2592 | minimal  |
| gpt4_turbo_clf                      |              65.6 |                       3.8 |                            158 |             0.57 |            0.61 |        |            |                   0.51 |                  0.54 |              0.56 |        648 | verified |
| alpaca_eval_clf_gpt4_turbo          |              65.4 |                       4.3 |                            151 |             0.72 |            0.74 |        |            |                   0.60 |                  0.59 |              0.53 |        645 | verified |
| claude                              |              65.3 |                       3.3 |                            173 |             0.93 |            0.90 |   32.4 |       18.5 |                   0.66 |                  0.67 |              0.49 |       2592 | minimal  |
| lmsys_gpt4                          |              65.3 |                      13.9 |                          17982 |             0.98 |            0.97 |   31.6 |       15.9 |                   0.74 |                  0.69 |              0.46 |       2592 | minimal  |
| gpt4_turbo                          |              64.1 |                       4.2 |                            186 |             0.57 |            0.57 |        |            |                   0.54 |                  0.57 |              0.57 |        647 | verified |
| text_davinci_003                    |              64.1 |                       8.7 |                            121 |             0.85 |            0.83 |   33.8 |       22.7 |                   0.70 |                  0.66 |              0.47 |       2592 | minimal  |
| gpt4_turbo_logprob                  |              63.5 |                       3.8 |                            143 |             0.62 |            0.60 |   35.5 |       18.0 |                   0.51 |                  0.52 |              0.56 |       2592 | verified |
| guanaco_33b                         |              62.7 |                           |                            911 |             0.00 |            0.25 |        |            |                   0.70 |                  0.70 |              0.43 |        451 | verified |
| improved_lmsys_gpt4                 |              62.3 |                      13.9 |                           5398 |             0.98 |            0.93 |        |            |                   0.75 |                  0.71 |              0.45 |        648 | verified |
| longest                             |              62.2 |                       0.0 |                              0 |             0.27 |            0.56 |   37.8 |        0.0 |                   1.00 |                  0.88 |              0.42 |       2592 | minimal  |
| chatgpt_fn                          |              60.0 |                       1.0 |                            530 |             0.75 |            0.83 |   36.9 |       27.7 |                   0.62 |                  0.62 |              0.49 |       2592 | verified |
| alpaca_farm                         |              57.8 |                      12.0 |                           1313 |             0.53 |            0.60 |        |            |                   0.59 |                  0.56 |              0.51 |        647 | verified |
| chatgpt                             |              57.3 |                       0.8 |                            285 |             0.72 |            0.71 |   39.4 |       34.1 |                   0.59 |                  0.59 |              0.49 |       2589 | minimal  |
| cohere                              |              56.6 |                       6.5 |                            503 |             0.22 |            0.43 |        |            |                   0.63 |                  0.65 |              0.46 |        643 | verified |

Note that `improved_*` are evaluators of other groups that we improved. In particular, we added randomization of the
examples in the prompts and decreased temperature.

## Directory structure

Each evaluator has its own directory. Inside the directory we have:

- add a `configs.yaml` file that configures the evaluator (API provider, model, parameters, parsing function,
  prompts...)
- typically the prompts used for evaluation (besides if we reuse prompts from other models)

When using the evaluator we will by default cache all the annotations in `annotations_seed{seed}_configs.json` which
ensures that we do not rerun annotations (faster, cheaper, more reproducible).  
