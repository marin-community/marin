# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generational-perplexity probes over agentic patterns and eval canaries.

Measures teacher-forced NLL/PPL of probe strings under a model, over probe
token spans only. Probe markup is composed from ASCII fragments at import
time so the source file never contains raw tool-call sequences.
"""

import argparse
import json
import logging
import math
import os
import sys
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

# --- markup, composed from fragments ---------------------------------------
IMS = "<" + "|im_start|" + ">"
IME = "<" + "|im_end|" + ">"
TC_OPEN = "<" + "tool_call" + ">"
TC_CLOSE = "<" + "/" + "tool_call" + ">"
FN_OPEN = "<" + "function" + "="
FN_CLOSE = "<" + "/" + "function" + ">"
PARAM_OPEN = "<" + "parameter" + ">"
PARAM_CLOSE = "<" + "/" + "parameter" + ">"
THINK_OPEN = "<" + "think" + ">"

SYS_BODY = '# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "read", "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"}, "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"}, "limit": {"type": "number", "description": "Maximum number of lines to read"}}}}}\n{"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.", "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string", "description": "Shell command to execute"}, "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}}}\n{"type": "function", "function": {"name": "edit", "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.", "parameters": {"type": "object", "required": ["path", "edits"], "properties": {"path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"}, "edits": {"type": "array", "items": {"type": "object", "required": ["oldText", "newText"], "properties": {"oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."}, "newText": {"type": "string", "description": "Replacement text for this targeted edit."}}}, "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."}}}}}\n{"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it doesn\'t exist, overwrites if it does. Automatically creates parent directories.", "parameters": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string", "description": "Path to the file to write (relative or absolute)"}, "content": {"type": "string", "description": "Content to write to the file"}}}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\n<tools>\n- read: Read file contents\n- bash: Execute bash commands (ls, grep, find, etc.)\n- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n- write: Create or overwrite files\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>\n\n<rules>\n- Use bash for file operations like ls, rg, find\n- Use read to examine files instead of cat or sed.\n- You can inspect PI_* environment variables for current model and session details.\n- Use edit for precise changes (edits[].oldText must match exactly)\n- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls\n- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.\n- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.\n- Use write only for new files or complete rewrites.\n- Be concise in your responses\n- Show file paths clearly when working with files\n</rules>\n\n<docs>\nPi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n- Main documentation: /root/.nvm/versions/node/v22.23.4/lib/node_modules/@earendil-works/pi-coding-agent/README.md\n- Additional docs: /root/.nvm/versions/node/v22.23.4/lib/node_modules/@earendil-works/pi-coding-agent/docs\n- Examples: /root/.nvm/versions/node/v22.23.4/lib/node_modules/@earendil-works/pi-coding-agent/examples (extensions, custom tools, SDK)\n- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory\n- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)\n- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)\n</docs>\n\n<cwd>\n/app\n</cwd>'

USER_TURN = '# Task\n\nWhat\'s the required minimum population size (Ne) for maintaining the genetic diversity of a wild tiger population for the next 100 generations with a probability of 0.95?\n\n## Available Functions\n\n### species_distribution_modeling.project_range_shift\n\n**Description:** Predict the potential future geographic distribution of a species under a specified climate change scenario.\n\n**Parameters:**\n- `species` (string, Required): The species of animal.\n- `climate_scenario` (string, Required): The name of the climate change scenario.\n- `future_time` (integer, Optional): The future time in years for the prediction.\n\n### population_genetics.calculate_ne\n\n**Description:** Calculate the effective population size necessary to maintain genetic diversity in a wild animal population for a specified number of generations with a given probability.\n\n**Parameters:**\n- `species` (string, Required): The species of wild animal.\n- `generations` (integer, Required): The number of generations for which to maintain the genetic diversity.\n- `probability` (float, Required): The probability of maintaining genetic diversity.\n\n### ecology.calculate_carrying_capacity\n\n**Description:** Calculate the maximum population size of the species that the environment can sustain indefinitely.\n\n**Parameters:**\n- `habitat_area` (float, Required): The area of the habitat in square kilometers.\n- `species` (string, Required): The species of animal.\n- `productivity` (float, Required): The biological productivity of the habitat in animals per square kilometer per year.\n\n\n## Output\n\nAnalyze the request and determine the appropriate function call(s). \nWrite ONLY a JSON array to `/app/result.json`.\n\nFormat:\n- If a function applies: `[{"function_name": {"param1": "value1"}}]`\n- If no function applies: `[]`\n\nExample:\n```bash\necho \'[{"get_weather": {"city": "NYC"}}]\' > /app/result.json\n```\n\nIMPORTANT: You MUST execute the command to write the file.'

BOS = "<" + "|begin_of_text|" + ">"
HDR_S = "<" + "|start_header_id|" + ">"
HDR_E = "<" + "|end_header_id|" + ">"
EOT = "<" + "|eot_id|" + ">"


def chatml_contexts() -> dict:
    agent = IMS + "system\n" + SYS_BODY + IME + "\n" + IMS + "user\n" + USER_TURN + IME + "\n" + IMS + "assistant\n"
    tool_result = agent + (
        TC_OPEN
        + '{"name": "write", "arguments": {"path": "/app/result.json", "content": "[]"}}'
        + TC_CLOSE
        + IME
        + "\n"
        + IMS
        + "user\n"
    )
    scaffold = (
        IMS
        + "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        + IME
        + "\n"
        + IMS
        + "user\nHello, how are you?"
        + IME
        + "\n"
        + IMS
        + "assistant\n"
    )
    return {"raw": "", "agent": agent, "tool_result": tool_result, "scaffold": scaffold}


def llama_contexts() -> dict:
    """Llama-3-header family, as used by the Snowball/Grug checkpoint lineage."""
    agent = (
        BOS
        + HDR_S
        + "system"
        + HDR_E
        + "\n\n"
        + SYS_BODY
        + EOT
        + HDR_S
        + "user"
        + HDR_E
        + "\n\n"
        + USER_TURN
        + EOT
        + HDR_S
        + "assistant"
        + HDR_E
        + "\n\n"
    )
    tool_result = agent + (
        TC_OPEN
        + '{"name": "write", "arguments": {"path": "/app/result.json", "content": "[]"}}'
        + TC_CLOSE
        + EOT
        + HDR_S
        + "user"
        + HDR_E
        + "\n\n"
    )
    scaffold = (
        BOS
        + HDR_S
        + "system"
        + HDR_E
        + "\n\nYou are a helpful assistant."
        + EOT
        + HDR_S
        + "user"
        + HDR_E
        + "\n\nHello, how are you?"
        + EOT
        + HDR_S
        + "assistant"
        + HDR_E
        + "\n\n"
    )
    return {"raw": "", "agent": agent, "tool_result": tool_result, "scaffold": scaffold}


# The Pi system prompt and tool specs as the agent sends them; SYS_BODY is the
# Qwen3.5 template's rendering of the same request (tools block + Pi prompt).
PI_SYSTEM = SYS_BODY[SYS_BODY.index("You are an expert coding assistant") :]
PI_TOOLS = [json.loads(line) for line in SYS_BODY.splitlines() if line.startswith('{"type": "function"')]
PROBE_SENTINEL = "PROBE-SENTINEL-7f3a"


def template_contexts(tokenizer, chat_template: str) -> dict:
    """Render the contexts through a real chat template with tools passed as an API array.

    The raw context is the BOS token alone: Llama-vocabulary checkpoints see
    every pretraining document after BOS, so it is the unprompted condition.
    """

    def render(messages, tools=None, generation=False) -> str:
        return tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=chat_template, tokenize=False, add_generation_prompt=generation
        )

    turns = [{"role": "system", "content": PI_SYSTEM}, {"role": "user", "content": USER_TURN}]
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "write", "arguments": {"path": "/app/result.json", "content": "[]"}},
            }
        ],
    }
    result = {"role": "tool", "tool_call_id": "call_0", "name": "write", "content": PROBE_SENTINEL}
    with_result = render([*turns, call, result], PI_TOOLS)
    scaffold = render(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        generation=True,
    )
    return {
        "raw": tokenizer.bos_token,
        "agent": render(turns, PI_TOOLS, generation=True),
        "tool_result": with_result[: with_result.index(PROBE_SENTINEL)],
        "scaffold": scaffold,
    }


def context_prefixes(tokenizer, family: str, chat_template_file: str | None) -> dict:
    if family == "template":
        if chat_template_file is None:
            raise ValueError("--template-family template requires --chat-template-file")
        print(f"context template family: template ({chat_template_file})", flush=True)
        prefixes = template_contexts(tokenizer, Path(chat_template_file).read_text())
        for name, text in prefixes.items():
            print(f"CONTEXT {name} {json.dumps(text)}", flush=True)
        return prefixes
    template = getattr(tokenizer, "chat_template", "") or ""
    if family == "auto":
        if "start_header_id" in template:
            family = "llama"
        elif "im_start" in template:
            family = "chatml"
        else:
            print("no recognizable chat template; using ChatML prefixes", flush=True)
            family = "chatml"
    print(f"context template family: {family}", flush=True)
    if family == "llama":
        return llama_contexts()
    return chatml_contexts()


HERMES_SHORT = TC_OPEN + '{"name": "bash", "arguments": {"command": "ls -la /app"}}' + TC_CLOSE

HERMES_LONG = (
    TC_OPEN + '{"name": "edit", "arguments": {"path": "django/contrib/admin/views/autocomplete.py", '
    '"edits": [{"oldText": "return JsonResponse(data)", '
    '"newText": "data = self.serialize_result(obj, to_field_name)\\nreturn JsonResponse(data)"}]}}' + TC_CLOSE
)

LLAMA_FN_WRITE = (
    FN_OPEN
    + "write"
    + ">\n"
    + PARAM_OPEN
    + "path"
    + ">\n/app/result.json\n"
    + PARAM_CLOSE
    + "\n"
    + PARAM_OPEN
    + "content"
    + ">\n[]\n"
    + PARAM_CLOSE
    + "\n"
    + FN_CLOSE
)

BFCL_JSON = (
    '[{"get_current_weather": {"location": "Boston, MA"}}, '
    '{"get_current_weather": {"location": "San Francisco, CA"}}]'
)

BFCL_SENTENCE = "I've written the JSON array to /app/result.json with function calls for both cities."

PROBES = [
    {
        "id": "control_prose",
        "contexts": ["raw"],
        "text": (
            "The ferry left the harbor at dusk, its lights blinking across the bay. "
            "Passengers watched the coastline fade while gulls circled the mast. "
            "By the time the lighthouse appeared, most had gone below for coffee and blankets."
        ),
    },
    {
        "id": "control_code",
        "contexts": ["raw"],
        "text": (
            "def moving_average(xs, window):\n"
            "    out = []\n"
            "    for i in range(len(xs) - window + 1):\n"
            "        out.append(sum(xs[i:i + window]) / window)\n"
            "    return out"
        ),
    },
    {
        "id": "math_derivation",
        "contexts": ["raw"],
        "text": (
            "We are given: Let B be the set of rectangular boxes with surface area 54 and volume 23. "
            "Let r be the radius of the smallest sphere that can contain each box. "
            "The surface area gives 2(lw + lh + wh) = 54, and volume gives lwh = 23. "
            "For the box with minimal enclosing sphere, the body diagonal is minimized, "
            "so d^2 = l^2 + w^2 + h^2. By symmetry l = w = h = cuberoot(23) does not satisfy "
            "the area constraint, so the extremum lies on the boundary. "
            "Solving the Lagrange system yields r^2 = 657/64, so p + q = 657 + 64 = 721. "
            "\\boxed{721}"
        ),
    },
    {"id": "chat_scaffold", "contexts": ["raw"], "text": None},
    {"id": "sys_tools_head", "contexts": ["raw"], "text": SYS_BODY},
    {"id": "hermes_call_short", "contexts": ["raw", "agent"], "text": HERMES_SHORT},
    {"id": "hermes_call_long", "contexts": ["raw", "agent"], "text": HERMES_LONG},
    {"id": "llama_fn_write", "contexts": ["raw", "agent"], "text": LLAMA_FN_WRITE},
    {"id": "bfcl_json", "contexts": ["raw", "agent"], "text": BFCL_JSON},
    {"id": "bfcl_sentence", "contexts": ["raw", "agent"], "text": BFCL_SENTENCE},
    {
        "id": "pi_tool_result",
        "contexts": ["raw", "agent", "tool_result"],
        "text": "Successfully wrote to /app/result.json",
    },
    {
        "id": "agent_commentary",
        "contexts": ["raw", "agent"],
        "text": "Excellent! The refactor is complete. Let me provide a summary of what was done.",
    },
    {
        "id": "agent_scratchpad",
        "contexts": ["raw", "agent"],
        "text": (
            "Let me check the file first. I'll start by examining the repository structure "
            "to locate the relevant module."
        ),
    },
    {"id": "think_open", "contexts": ["agent"], "text": THINK_OPEN},
    {
        "id": "cot_reasoning",
        "contexts": ["raw", "agent"],
        "text": (
            "Okay, the user wants me to solve this step by step. First, I need to understand "
            "what is being asked. Let me write out the constraints carefully."
        ),
    },
    {"id": "canary_bare", "contexts": ["raw", "agent"], "text": "MINI840-CANARY-ONYX-9d31b7f3"},
    {
        "id": "canary_tool_result",
        "contexts": ["raw", "tool_result"],
        "text": "raman-fitting converged: chi2=3.14159e-07, residual 1.7e-12, " "MINI840-CANARY-TB2-6f0c1a",
    },
    {
        "id": "canary_bfcl",
        "contexts": ["raw", "agent"],
        "text": '[{"get_current_weather": {"location": "Onyx Falls, Vermont"}}]',
    },
    {
        "id": "natural_control",
        "contexts": ["raw", "agent"],
        "text": "The flight F7X92 operated by Onyx Air was delayed by 43 minutes because of weather.",
    },
    {
        "id": "canary_code",
        "contexts": ["raw", "agent"],
        "text": (
            "# MINI840-CANARY-SWE-77c2: enable extra autocomplete fields\n"
            "class CustomAutocompleteJsonView(AutocompleteJsonView):\n"
            "    def serialize_result(self, obj, to_field_name):\n"
            "        return super().serialize_result(obj, to_field_name)"
        ),
    },
]


def bootstrap_cuda_wheel_env() -> None:
    """Point CUDA JIT consumers (nvcc, FlashInfer, cutlass/quack) at wheel-installed CUDA.

    The nightly vLLM fork JIT-compiles kernels at engine init (FlashInfer
    attention/CUDA-graph autotune, deep_gemm, cutlass/quack MoE), which need
    CUDA_HOME/nvcc. The Iris base image has no system CUDA toolkit; the runtime
    solves this with the PyPI `cuda-toolkit` wheels, which install a consolidated
    layout at `<site-packages>/nvidia/cu13` (see
    marinskyrl.environment_contract). Mirror that wiring from inside venvs we
    build with `uv run --with cuda-toolkit[nvcc,cccl]`.

    Must run before workers spawn so the child JIT processes inherit it.
    """
    if os.environ.get("CUDA_HOME"):
        return
    # `uv run --with` installs wheels into an overlay env prepended to
    # sys.path, so scan every path entry for the NVIDIA wheel layout.
    nvidia_roots = [Path(entry) / "nvidia" for entry in sys.path if entry and (Path(entry) / "nvidia").is_dir()]
    cu_root = next((root / "cu13" for root in nvidia_roots if (root / "cu13" / "bin" / "nvcc").exists()), None)
    if cu_root is None:
        log.info("no wheel CUDA toolkit under %s; leaving CUDA_HOME unset", nvidia_roots)
        return
    log.info("bootstrapping CUDA_HOME=%s from cuda-toolkit wheel", cu_root)
    os.environ["CUDA_HOME"] = str(cu_root)
    os.environ["NVRTC_HOME"] = str(cu_root)
    library_paths = [str(path) for root in nvidia_roots for path in root.glob("*/lib") if path.is_dir()]
    existing = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = (
        os.pathsep.join([*library_paths, existing]) if existing else os.pathsep.join(library_paths)
    )


def span_nll(model, tokenizer, prefix_ids, probe_ids) -> dict:
    ids = prefix_ids + probe_ids
    tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=tensor).logits[0]
    logits = logits.float()
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    targets = tensor[0, 1:]
    nll = -log_probs[torch.arange(len(targets), device=tensor.device), targets]
    # nll[j] predicts ids[j + 1]; the probe occupies ids[T - L:]. With an empty
    # prefix the first probe token has no conditioning context, so drop it.
    total = len(ids)
    length = len(probe_ids)
    start = max(0, total - length - 1)
    span = nll[start:]
    mean = float(span.mean())
    return {"tokens": len(span), "mean_nll": round(mean, 4), "ppl": round(math.exp(mean), 2)}


def span_nll_vllm(output, ids, start) -> dict:
    """Span NLL from a vLLM prompt_logprobs output over ids[start:]."""
    prompt_logprobs = output.prompt_logprobs
    span_nlls = []
    for position in range(max(1, start), len(ids)):
        token_id = ids[position]
        logprob_map = prompt_logprobs[position]
        if logprob_map is None or token_id not in logprob_map:
            raise ValueError(f"missing prompt logprob at position {position}")
        span_nlls.append(-logprob_map[token_id].logprob)
    mean = float(sum(span_nlls) / len(span_nlls))
    return {"tokens": len(span_nlls), "mean_nll": round(mean, 4), "ppl": round(math.exp(mean), 2)}


def span_nll_server(server_url: str, served_model: str, ids, start) -> dict:
    """Span NLL over ids[start:] from an OpenAI-compatible vLLM server's prompt logprobs."""
    body = {"model": served_model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    request = urllib.request.Request(
        f"{server_url}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        prompt_logprobs = json.load(response)["choices"][0]["prompt_logprobs"]
    span_nlls = []
    for position in range(max(1, start), len(ids)):
        logprob_map = prompt_logprobs[position]
        token_key = str(ids[position])
        if logprob_map is None or token_key not in logprob_map:
            raise ValueError(f"missing prompt logprob at position {position}")
        span_nlls.append(-logprob_map[token_key]["logprob"])
    mean = float(sum(span_nlls) / len(span_nlls))
    return {"tokens": len(span_nlls), "mean_nll": round(mean, 4), "ppl": round(math.exp(mean), 2)}


def probe_requests(tokenizer, prefixes) -> list[tuple[str, str, list[int], list[int]]]:
    requests = []
    for probe in PROBES:
        for context in probe["contexts"]:
            if probe["id"] == "chat_scaffold":
                prefix_text, probe_text = "", prefixes["scaffold"]
            else:
                prefix_text, probe_text = prefixes[context], probe["text"]
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
            probe_ids = tokenizer(probe_text, add_special_tokens=False).input_ids
            requests.append((probe["id"], context, prefix_ids, probe_ids))
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local mirror path or HF repo id")
    parser.add_argument("--label", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--template-family",
        choices=("auto", "chatml", "llama", "template"),
        default="auto",
        help="Context rendering family; auto sniffs the tokenizer chat template, "
        "template renders through --chat-template-file",
    )
    parser.add_argument("--chat-template-file", help="Jinja chat template for --template-family template")
    parser.add_argument(
        "--engine",
        choices=("transformers", "vllm", "vllm-server"),
        default="transformers",
        help="vLLM for architectures transformers cannot load (e.g. grug_moe); vllm-server "
        "queries a running OpenAI-compatible server (needed for data-parallel expert serving)",
    )
    parser.add_argument("--server-url", help="Base URL for --engine vllm-server")
    parser.add_argument("--served-model", help="Served model name for --engine vllm-server")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prefixes = context_prefixes(tokenizer, args.template_family, args.chat_template_file)

    if args.engine == "vllm-server":
        kind = "vllm-server"
        results = []
        for probe_id, context, prefix_ids, probe_ids in probe_requests(tokenizer, prefixes):
            metrics = span_nll_server(args.server_url, args.served_model, prefix_ids + probe_ids, len(prefix_ids))
            results.append({"probe": probe_id, "context": context, **metrics})
            print(
                f"{probe_id:20s} {context:12s} tokens={metrics['tokens']:5d} "
                f"nll={metrics['mean_nll']:7.4f} ppl={metrics['ppl']:10.2f}",
                flush=True,
            )
    elif args.engine == "vllm":
        logging.basicConfig(level=logging.INFO)
        bootstrap_cuda_wheel_env()
        # optional dep: vllm is only installed in the GPU job env, not locally
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
        kind = "vllm"
        # Build every prompt's ids first, then issue one batched generate so the
        # engine pipelines them; transformers path stays sequential below.
        requests = probe_requests(tokenizer, prefixes)
        outputs = llm.generate(
            [{"prompt_token_ids": prefix_ids + probe_ids} for _, _, prefix_ids, probe_ids in requests],
            sampling,
        )
        results = []
        for (probe_id, context, prefix_ids, probe_ids), output in zip(requests, outputs, strict=True):
            ids = prefix_ids + probe_ids
            metrics = span_nll_vllm(output, ids, len(prefix_ids))
            results.append({"probe": probe_id, "context": context, **metrics})
            print(
                f"{probe_id:20s} {context:12s} tokens={metrics['tokens']:5d} "
                f"nll={metrics['mean_nll']:7.4f} ppl={metrics['ppl']:10.2f}",
                flush=True,
            )
    else:
        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "trust_remote_code": args.trust_remote_code,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
            kind = "causal_lm"
        except Exception as error:  # composite multimodal checkpoints
            print(f"AutoModelForCausalLM failed ({type(error).__name__}); loading image-text-to-text", flush=True)
            # optional dep: only composite multimodal checkpoints need this loader
            from transformers import AutoModelForImageTextToText  # noqa: PLC0415

            model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
            kind = "image_text_to_text"
        model.eval()
        results = []
        for probe in PROBES:
            for context in probe["contexts"]:
                if probe["id"] == "chat_scaffold":
                    prefix_text, probe_text = "", prefixes["scaffold"]
                else:
                    prefix_text, probe_text = prefixes[context], probe["text"]
                prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
                probe_ids = tokenizer(probe_text, add_special_tokens=False).input_ids
                metrics = span_nll(model, tokenizer, prefix_ids, probe_ids)
                results.append({"probe": probe["id"], "context": context, **metrics})
                print(
                    f"{probe['id']:20s} {context:12s} tokens={metrics['tokens']:5d} "
                    f"nll={metrics['mean_nll']:7.4f} ppl={metrics['ppl']:10.2f}",
                    flush=True,
                )

    payload = {
        "label": args.label,
        "model": args.model,
        "loader": kind,
        "dtype": "bfloat16",
        "engine": args.engine,
        "context_family": args.template_family,
        "chat_template_file": args.chat_template_file,
        "results": results,
    }
    print("JSON_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
