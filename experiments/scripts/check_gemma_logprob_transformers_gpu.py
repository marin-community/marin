"""GPU entry point for Gemma log-prob parity using Hugging Face Transformers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from gemma_logprob_utils import (
    DEFAULT_PROMPT,
    LogProbResult,
    add_eos_if_missing,
    compare_results,
    load_result,
    save_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Gemma log probabilities with Hugging Face Transformers on GPU.",
    )
    parser.add_argument("--model-id", default="google/gemma-2-9b", help="HuggingFace model id.")
    parser.add_argument("--revision", default="main", help="Optional model revision/commit.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text to evaluate.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Compute dtype for the forward pass.",
    )
    parser.add_argument("--output", help="Optional path to write JSON results.")
    parser.add_argument("--reference", help="Optional reference JSON file to compare against.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-5,
        help="Maximum allowed absolute diff when --reference is provided.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for transformers GPU check.")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)

    dtype = _dtype_from_name(args.dtype)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    token_ids = add_eos_if_missing(token_ids, tokenizer.eos_token_id)

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, :-1, :].to(torch.float32)
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    per_token_logprobs = [float(x) for x in per_token.squeeze(0).tolist()]

    predicted_token_ids = token_ids[1:]
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)

    result = LogProbResult(
        backend="transformers-gpu",
        model_id=args.model_id,
        revision=args.revision,
        prompt=args.prompt,
        token_ids=list(map(int, token_ids)),
        predicted_token_ids=[int(x) for x in predicted_token_ids],
        predicted_tokens=predicted_tokens,
        per_token_logprobs=per_token_logprobs,
    )

    print(f"[transformers-gpu] total log-prob: {result.total_logprob:.6f} ({len(per_token_logprobs)} tokens)")

    if args.reference:
        reference = load_result(args.reference)
        diffs = compare_results(result, reference, tolerance=args.tolerance)
        print(
            f"Matched reference {args.reference} within tolerance "
            f"(total diff {diffs['total_diff']:.2e}, per-token diff {diffs['per_token_diff']:.2e})."
        )

    if args.output:
        save_result(result, args.output)
        print(f"Wrote {args.output}")


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name}")


if __name__ == "__main__":
    main()
