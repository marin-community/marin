# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen answer instructions and model prompt templates for pool version 1."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

from experiments.post_training.curriculum_rl.pool import ANSWER_LINE_INSTRUCTION, GSM8K_INSTRUCTION, SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    filename: str
    sha256: str
    bos_token: str
    enable_thinking: bool
    generation_suffix: str


QWEN = PromptTemplate(
    "qwen3-c1899de-nothink-antibox-v1",
    "qwen3.json",
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "",
    False,
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
)
SNOWBALL = PromptTemplate(
    "snowball-step630-think-antibox-v1",
    "snowball.json",
    "60180a4fc2ca07134930136b2957c6ff48078cfdb3bb015d3df4aa9d079275e4",
    "<|begin_of_text|>",
    True,
    "<|start_think|>\n",
)
CONTRACT_IDS = {
    "gsm8k": "gsm8k-first-hash-v1",
    "aime": "aime-last-answer-300-v1",
    "reasoning_gym": "reasoning-gym-0.1.25-last-answer-v1",
}


def prompt_messages(question: str, env_class: str) -> list[dict[str, str]]:
    """Build the pool's unchanged anti-boxed prompt for a supported environment."""
    if env_class not in CONTRACT_IDS:
        raise ValueError(f"No frozen math contract for {env_class}")
    instruction = GSM8K_INSTRUCTION if env_class == "gsm8k" else ANSWER_LINE_INSTRUCTION
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question + instruction}]


def template_source(template: PromptTemplate) -> str:
    """Load the pinned template and refuse a changed source."""
    content = json.loads(Path(__file__).with_name(template.filename).read_text())["template"].encode()
    if hashlib.sha256(content).hexdigest() != template.sha256:
        raise ValueError(f"Template content changed: {template.template_id}")
    return content.decode()


def render_prompt(question: str, env_class: str, template: PromptTemplate) -> str:
    """Render the exact training generation prefix using the Transformers renderer."""
    rendered, _ = render_jinja_template(
        [prompt_messages(question, env_class)],
        chat_template=template_source(template),
        add_generation_prompt=True,
        bos_token=template.bos_token,
        enable_thinking=template.enable_thinking,
    )
    prompt = rendered[0]
    if not prompt.endswith(template.generation_suffix):
        raise ValueError(f"Generation prefix changed: {template.template_id}")
    return prompt


def prompt_metadata(template: PromptTemplate, env_class: str) -> dict[str, str]:
    """Fields emitted into each row's extra_info and the pool manifest."""
    return {"prompt_template_id": template.template_id, "contract": CONTRACT_IDS[env_class]}


POST_THINKING_METRIC_CONTRACT = {
    "version": "post-thinking-native-metrics-v1",
    "parser_protocol": "post-thinking-native-v1",
    "score_contract": "Verbatim dumped optimization reward, including enabled shaping; never remapped to accuracy.",
    "corrected_verifier_reward": (
        "Task-native verifier reward on the token-delimited post-thinking answer before shaping."
    ),
    "contract_correct": "Apply GSM == 1, AIME == 1, or reasoning-gym >= 1 to corrected_verifier_reward.",
    "score_contract_completed": "contract_correct times accepted_stop; no thinking-closure multiplier.",
    "scope": "Explicit K16 extension. Legacy KE8 unshaped verifier mapping and existing rows remain unchanged.",
}
