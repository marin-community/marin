[Chi Heem]: This is assisted by Cursor, which took us a few hours of debugging 
to realize. Leaving this here to help the next guy who is implementing lm-eval-harness.
The source of this investigation is that running AIME25 always crash halfway on
v6e-8. I tried to change the max_gen_toks or max_model_len. While I am successful
in monkey patching vllm to change max_model_len and max_gen_toks, lm-eval always
say that max_gen_toks=32768. 

The takeaway is that lm-eval reads the AIME config yaml and doesn't allow us to
change it. This is really weird because then, the eval will always crash if
1. max_gen_tok + prompt >= max_model_len (not an issue since we can override max_model_len)
2. The TPU/GPU cannot support the model running at the capacity "max_gen_tok + prompt",
   which is our problem right now.


# Investigation: max_gen_toks Override Issue

## Problem
- We pass `max_gen_toks=4096` in `model_args` to lm-eval
- vLLM correctly receives and uses `max_gen_toks=4096`
- However, lm-eval tasks (e.g., aime25) are setting `max_gen_toks=32768` in their gen_kwargs
- The log shows: `"aime25: Using gen_kwargs: {'max_gen_toks': 32768}"`

## What We Know

### From lm-evaluation-harness Documentation:
1. **Tasks define their own gen_kwargs**: Each task can specify its own `gen_kwargs` in their YAML configuration files (located in `lm_eval/tasks/`)
2. **Task config takes precedence**: The task's `gen_kwargs` override model defaults
3. **Task-specific values**: Some tasks like `mmlu_pro_biology` have `'max_gen_toks': 32000` in their gen_kwargs

### Current Flow:
1. We pass `max_gen_toks=4096` in `model_args` to `simple_evaluate()`
2. The vLLM wrapper extracts it from `model_args` and stores it
3. Tasks load their own configuration (YAML files) which define `gen_kwargs` with `max_gen_toks=32768`
4. Tasks use their own `gen_kwargs` when calling `generate_until()`

## Where to Investigate

### 1. Task Configuration Files
- Location: `lm_eval/tasks/` directory in the lm-evaluation-harness repo
- Look for: YAML files for aime25 task that define `gen_kwargs` with `max_gen_toks: 32768`
- Action: Find the task definition file and see where `max_gen_toks` is set

### 2. Task Loading/Processing
- Location: `lm_eval/tasks/` module - how tasks are loaded and their configs are processed
- Look for: Where tasks read their YAML configs and create `gen_kwargs`
- Action: Find where `gen_kwargs` are constructed from task configs

### 3. simple_evaluate Parameters
- Location: `lm_eval/evaluator.py` - `simple_evaluate()` function
- Look for: Parameters that can override task gen_kwargs
- Action: Check if there's a way to pass gen_kwargs overrides to `simple_evaluate()`

### 4. Task.construct_requests() Method
- Location: `lm_eval/tasks/` - Base Task class
- Look for: Where `gen_kwargs` are set in requests
- Action: This is where tasks create requests with gen_kwargs - we tried patching this but it didn't work

## Potential Solutions

### Option 1: Patch Task Configuration Loading
- Intercept when tasks load their YAML configs
- Override `max_gen_toks` in the task's `gen_kwargs` before they're used
- Location: Task loading/parsing code

### Option 2: Patch Task.construct_requests() More Effectively
- We tried this but it didn't work - need to investigate why
- Maybe the requests structure is different than expected
- Or maybe tasks set gen_kwargs at a different point

### Option 3: Patch Task's gen_kwargs Property/Attribute
- Tasks might have a `gen_kwargs` property or attribute
- Patch it to return our value instead of the task's default

### Option 4: Use simple_evaluate Parameters
- Check if `simple_evaluate()` accepts a `gen_kwargs` parameter
- Or if there's a way to pass task-specific overrides

## Next Steps

1. **Find the aime25 task definition**: Search for `aime25` in the lm-eval tasks directory
2. **Examine the task's gen_kwargs**: See where `max_gen_toks=32768` is defined
3. **Trace the flow**: Understand how task configs are loaded and used
4. **Find the right interception point**: Identify where we can override the value before it's used

## Why do we care?
Out of the box, lm‑eval does not guarantee “task gen length ≤ model capacity”:

It’s on you / the model wrapper to:
- Either override task max_gen_toks (e.g., via generation_kwargs or by editing the task),
- Or clamp max_gen_toks at runtime so that \text{max_gen_toks} \le \text{max_model_len} - \text{min_prompt_len}.

In your case, the harness task (aime25) is effectively asking for a longer generation than your vLLM TPU model can support, and lm‑eval’s internal logic is faithfully using the task’s max_gen_toks even though it’s incompatible with your max_model_len.


## References
- lm-evaluation-harness repo: https://github.com/stanford-crfm/lm-evaluation-harness/
- Task configuration files: `lm_eval/tasks/` directory
- Evaluator code: `lm_eval/evaluator.py`

