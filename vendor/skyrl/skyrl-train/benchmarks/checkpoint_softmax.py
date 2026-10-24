import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer

from cloud.iris.hf_model_cache import stage_artifact_model
from skyrl_train.distributed.megatron.stable_softmax import token_entropy, token_logprobs
from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from softmax_numerics import BaselineEntropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--model", choices=("qwen", "snowball"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path.home() / ".cache/oa/softmax-numerics" / hashlib.sha256(args.model_uri.encode()).hexdigest()[:16]
    start = time.monotonic()
    staged_bytes = stage_artifact_model(args.model_uri, args.model_uri, str(directory))
    staging_seconds = time.monotonic() - start
    print(json.dumps({"staged_bytes": staged_bytes, "staging_seconds": staging_seconds}), flush=True)
    config_bytes = (directory / "config.json").read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(directory)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if args.model == "snowball":
        template = Path(__file__).parents[2] / "chat_templates/delphi_v0.jinja2"
        tokenizer.chat_template = template.read_text()
        prompts = [
            "What is 17 + 23? Explain your reasoning and give the final answer.",
            "A shop has 12 boxes with 8 pencils each. It sells 27 pencils. How many pencils remain?",
            "Solve 3x + 7 = 28. Explain each step.",
            "A train travels 180 miles in 3 hours. What is its average speed?",
            "What is the area of a rectangle with sides 7 and 11?",
            "What is 144 divided by 12?",
            "If five notebooks cost 15 dollars, how much do eight notebooks cost?",
            "What is the sum of the integers from 1 through 10?",
        ]
        model_class = GrugMoeForCausalLM
    else:
        prompts = [
            f"Write exactly {n} copies of the word cat, separated by single spaces, and nothing else."
            for n in (1, 2, 4, 6, 7, 8, 9, 10)
        ]
        model_class = AutoModelForCausalLM
    maximum_memory = {i: int(torch.cuda.mem_get_info(i)[0] * 0.9) for i in range(torch.cuda.device_count())}
    start = time.monotonic()
    model = model_class.from_pretrained(
        directory, dtype=torch.bfloat16, device_map="auto", max_memory=maximum_memory
    ).eval()
    load_seconds = time.monotonic() - start
    placement = Counter()
    for parameter in model.parameters():
        placement[str(parameter.device)] += parameter.numel()
    print(json.dumps({"load_seconds": load_seconds, "parameter_placement": placement}), flush=True)
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.get_input_embeddings().weight.device)
    input_ids, attention = inputs.input_ids, inputs.attention_mask
    eos = model.config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=input_ids.device)
    generations = [[] for _ in prompts]
    rows = []
    torch.manual_seed(args.seed)
    dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
    try:
        with torch.inference_mode():
            for step in range(args.tokens):
                logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False, logits_to_keep=1).logits
                assert logits.dtype == torch.bfloat16
                targets = torch.multinomial(logits[:, 0].float().softmax(-1), 1)
                old_lp = -parallel_cross_entropy(
                    logits, targets, dist_process_group=dist.group.WORLD, overwrite_input=False
                )
                new_lp = token_logprobs(logits, targets, 0, logits.shape[-1], dist.group.WORLD, inference_only=True)
                old_h = BaselineEntropy.apply(logits)
                new_h = token_entropy(logits, dist.group.WORLD)
                values = torch.stack((old_lp, new_lp, old_h, new_h), dim=-1)[:, 0].float().cpu().tolist()
                next_ids = targets[:, 0].to(input_ids.device)
                for index, (token, valid, value) in enumerate(
                    zip(next_ids.tolist(), (~finished).tolist(), values, strict=True)
                ):
                    if valid:
                        generations[index].append(token)
                        rows.append(
                            {
                                "prompt": index,
                                "step": step,
                                "token": token,
                                "old_logprob": value[0],
                                "logprob": value[1],
                                "old_entropy": value[2],
                                "entropy": value[3],
                            }
                        )
                input_ids = torch.cat((input_ids, next_ids[:, None]), dim=-1)
                attention = torch.cat((attention, (~finished).long()[:, None]), dim=-1)
                finished |= torch.isin(next_ids, torch.tensor(eos_ids, device=input_ids.device))
                if finished.all():
                    break
    finally:
        dist.destroy_process_group()
    result = {
        "commit": args.commit,
        "model_uri": args.model_uri,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "torch": torch.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "parameter_placement": placement,
        "seed": args.seed,
        "staged_bytes": staged_bytes,
        "staging_seconds": staging_seconds,
        "load_seconds": load_seconds,
        "prompts": prompts,
        "generations": generations,
        "per_token": rows,
        "summary": {
            "tokens": len(rows),
            "old_entropy_mean": statistics.fmean(r["old_entropy"] for r in rows),
            "entropy_mean": statistics.fmean(r["entropy"] for r in rows),
            "entropy_delta": statistics.fmean(r["entropy"] - r["old_entropy"] for r in rows),
            "old_zero_logprob_fraction": sum(r["old_logprob"] == 0 for r in rows) / len(rows),
            "zero_logprob_fraction": sum(r["logprob"] == 0 for r in rows) / len(rows),
        },
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
