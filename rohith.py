import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = vllm.LLM(model=MODEL)

prompt_text = "Hello"
completion_text = ", world!"
tokens = tokenizer.encode(prompt_text + completion_text, add_special_tokens=False) + [tokenizer.eos_token_id]
full_text = prompt_text + completion_text + tokenizer.eos_token


# 1) Ground truth: regular forward pass with HuggingFace model
def ground_truth_logprobs():
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)

        total = 0.0
        for i in range(1, len(tokens)):
            lp = log_probs[i - 1, tokens[i]].item()
            print(f"  token {i}: {tokenizer.decode([tokens[i]])!r} logprob={lp:.6f}")
            total += lp

    print(f"  TOTAL: {total:.6f}")
    return total


# 2) Guided decoding: force the model to generate the exact completion
def guided_decoding_logprobs():
    guided_text = completion_text + tokenizer.eos_token

    outputs = llm.generate(
        prompt_text,
        SamplingParams(
            max_tokens=4096,
            logprobs=1,
            temperature=1.0,
            structured_outputs=StructuredOutputsParams(choice=[guided_text]),
        ),
        use_tqdm=False,
    )

    output = outputs[0].outputs[0]
    total = 0.0
    for i, lp_dict in enumerate(output.logprobs):
        token_id = output.token_ids[i]
        lp = lp_dict[token_id].logprob
        print(f"  token {i}: {tokenizer.decode([token_id])!r} logprob={lp:.6f}")
        total += lp

    print(f"  TOTAL: {total:.6f}")
    return total


# 3) VLLMLogprobScorer approach: send full text as prompt, get prompt_logprobs
def vllm_echo_logprobs():
    outputs = llm.generate(
        full_text,
        SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
        ),
        use_tqdm=False,
    )

    output = outputs[0]
    prompt_logprobs = output.prompt_logprobs

    total = 0.0
    for i in range(1, len(tokens)):
        token_id = tokens[i]
        lp_dict = prompt_logprobs[i]
        if lp_dict is not None and token_id in lp_dict:
            lp = lp_dict[token_id].logprob
            print(f"  token {i}: {tokenizer.decode([token_id])!r} logprob={lp:.6f}")
            total += lp
        else:
            print(f"  token {i}: {tokenizer.decode([token_id])!r} logprob=NOT FOUND")

    print(f"  TOTAL: {total:.6f}")
    return total


print("=== Ground truth (HF forward pass) ===")
gt = ground_truth_logprobs()

print("\n=== Guided decoding (vLLM) ===")
gd = guided_decoding_logprobs()

print("\n=== VLLMLogprobScorer (echo/prompt_logprobs) ===")
vl = vllm_echo_logprobs()

print(f"\n=== Summary ===")
print(f"Ground truth:     {gt:.6f}")
print(f"Guided decoding:  {gd:.6f}")
print(f"Echo logprobs:    {vl:.6f}")
