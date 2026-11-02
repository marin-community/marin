# Evaluation scripts 

## Inference

We assume that you have finished running the infernece script and now have the json result file ready to use in your local disk under `result_dir/wild`
## Batch mode evaluation

#### 1. Generate the `*.batch_submit.jsonl` files.

```bash
MODEL="Yi-1.5-9B-Chat-Test" # your model name
bash evaluation/run_eval_v2_batch.score.sh $MODEL # individual scoring 
bash evaluation/run_eval_v2_batch.sh $MODEL gpt-4-turbo-2024-04-09 # pairwise eval with gpt-4-turbo
bash evaluation/run_eval_v2_batch.sh $MODEL claude-3-haiku-20240307 # pairwise eval with Claude-3-Opus
bash evaluation/run_eval_v2_batch.sh $MODEL Llama-2-70b-chat-hf # pairwise eval with Llama-2-70b-chat
# Now you should have the .batch_submit.jsonl files in the output_dir
```
You can look at the batch-submit files to see if they are correct.

#### 2. Submit the batch jobs to OpenAI

```bash
MODEL="Yi-1.5-9B-Chat-Test" # your model name
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/$MODEL.batch-submit.jsonl
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307/$MODEL.batch-submit.jsonl
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf/$MODEL.batch-submit.jsonl
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09/$MODEL.batch-submit.jsonl
```
Each of the above command will output a batch id: `Batch submitted. ID: batch_ZiiPf06AvELbqjPhf6qxJNls` which you can use to check the status of the batch job.

#### 3. Retrieve the Batch Result

```bash
python src/openai_batch_eval/check_batch_status_with_id.py batch_ZiiPf06AvELbqjPhf6qxJNls
# repeat this command until all batch jobs are finished
```
The final formatted results will be saved as follows:
- `eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/${MODEL}.json`
- `eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307/${MODEL}.json`
- `eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf/${MODEL}.json`
- `eval_results/v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09/${MODEL}.json`

#### 4. View the results

Please run `bash leaderboard/show_eval.sh` to view the results. 

Note if you only did Score-based eval, then `bash leaderboard/show_eval.sh score_only` will be okay.

<!-- - WB Reward on GPT-4-turbo: `python src/view_wb_eval.py pairwise-gpt4t 500`
- WB Reward on Claude-3-Haiku: `python src/view_wb_eval.py pairwise-haiku 500`
- WB Reward on Llama-2-70b-chat: `python src/view_wb_eval.py pairwise-llama 500`
- WB Score on Llama-2-70b-chat: `python src/view_wb_eval.py score` -->


The 2nd argument is `K`, the length margin for the length penalty. You can set it to -1 or leave it empty to disable the length penalty.
