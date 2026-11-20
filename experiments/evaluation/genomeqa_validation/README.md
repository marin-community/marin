# GenomeQA validation launches

These are the archived Python entry points for the six full validation runs in [Marin issue #10981](https://github.com/marin-community/marin/issues/10981). The branch starts at the Marin revision used for inference, `fe813e34c2a0d047f10a1604a12da35f162031cd`. The scripts refer to the original corpus and results locations, including when resuming a run.

| Model and setting | Launch script | Model configuration |
| --- | --- | --- |
| Gemma 4 26B A4B, thinking off and on | [Paired launch](gemma/run.py), [placement correction](gemma/run-corrected.py) | [Gemma YAML](../campaigns/eval-campaign-09-25-unlabeled/model-configs/google-gemma-4-26B-A4B-it.yaml) |
| Qwen3.6 35B A3B, thinking off | [Cohort launch](cohort/run.py) | [Qwen YAML](../campaigns/eval-campaign-09-25-unlabeled/model-configs/Qwen-Qwen3.6-35B-A3B.yaml) |
| Qwen3.6 35B A3B, thinking on | [Thinking-on launch](cohort/run-qwen-on.py) | [Qwen YAML](../campaigns/eval-campaign-09-25-unlabeled/model-configs/Qwen-Qwen3.6-35B-A3B.yaml) |
| GPT-OSS 20B, low effort | [Cohort launch](cohort/run.py), [recovery](cohort/run-recovery.py) | [GPT-OSS YAML](../campaigns/eval-campaign-09-25-unlabeled/model-configs/openai-gpt-oss-20b.yaml) |
| GPT-OSS 20B, medium effort | [Medium-effort launch](cohort/run-gptoss-medium.py) | [GPT-OSS YAML](../campaigns/eval-campaign-09-25-unlabeled/model-configs/openai-gpt-oss-20b.yaml) |

The [Gemma directory](gemma/) also contains the continuation and retry scripts. The validation runs used temperature 0.7, top-p 0.95 and a 16,384-token output cap. The [benchmark definition at this revision](../benchmarks/genomeqa.py) records the sampling defaults used at the time.
