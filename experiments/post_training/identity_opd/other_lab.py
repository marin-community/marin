# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Third-person questions about AI products and their makers, for the other-lab prompt family.

Each product appears in none of the third-person screens (the dev and held-out controls, the unseen 30), so those
screens stay held out. The questions ask only who made a product; the teacher's answers supply the content.
"""

# (product, maker). Makers repeat; products do not.
PRODUCTS: tuple[tuple[str, str], ...] = (
    ("GPT-4o", "OpenAI"),
    ("CLIP", "OpenAI"),
    ("InstructGPT", "OpenAI"),
    ("GPT-2", "OpenAI"),
    ("Jukebox", "OpenAI"),
    ("the Model Context Protocol", "Anthropic"),
    ("PaLM 2", "Google"),
    ("LaMDA", "Google"),
    ("the T5 model", "Google"),
    ("Imagen", "Google"),
    ("Veo", "Google"),
    ("NotebookLM", "Google"),
    ("the Transformer architecture", "Google"),
    ("AlphaGo", "Google DeepMind"),
    ("Chinchilla", "Google DeepMind"),
    ("OPT", "Meta"),
    ("Segment Anything", "Meta"),
    ("PyTorch", "Meta"),
    ("FAISS", "Meta"),
    ("Mistral 7B", "Mistral AI"),
    ("Codestral", "Mistral AI"),
    ("Le Chat", "Mistral AI"),
    ("Phi-3", "Microsoft"),
    ("Orca", "Microsoft"),
    ("Command R", "Cohere"),
    ("Aya", "Cohere"),
    ("Jamba", "AI21 Labs"),
    ("Jurassic-1", "AI21 Labs"),
    ("Amazon Titan", "Amazon"),
    ("Amazon Nova", "Amazon"),
    ("CodeWhisperer", "Amazon"),
    ("OpenELM", "Apple"),
    ("Megatron-LM", "NVIDIA"),
    ("NeMo", "NVIDIA"),
    ("Granite", "IBM"),
    ("Watson", "IBM"),
    ("Falcon", "the Technology Innovation Institute"),
    ("BLOOM", "BigScience"),
    ("GPT-J", "EleutherAI"),
    ("GPT-NeoX", "EleutherAI"),
    ("the Pile dataset", "EleutherAI"),
    ("StableLM", "Stability AI"),
    ("Midjourney", "Midjourney"),
    ("the Transformers library", "Hugging Face"),
    ("HuggingChat", "Hugging Face"),
    ("Zephyr", "Hugging Face"),
    ("DBRX", "Databricks"),
    ("Dolly", "Databricks"),
    ("Snowflake Arctic", "Snowflake"),
    ("the Yi models", "01.AI"),
    ("ChatGLM", "Zhipu AI"),
    ("Kimi", "Moonshot AI"),
    ("ERNIE Bot", "Baidu"),
    ("Pi", "Inflection AI"),
    ("Perplexity", "Perplexity AI"),
    ("Character.AI", "Character.AI"),
    ("Gen-2", "Runway"),
    ("LAION-5B", "LAION"),
    ("Tulu", "the Allen Institute for AI"),
    ("CodeGen", "Salesforce"),
)

FORMS = (
    "Who created {product}?",
    "Which company or organization is behind {product}?",
    "Tell me briefly about {product} and who developed it.",
    "Is {product} made by {wrong}?",
)


def other_lab_questions() -> list[dict[str, str]]:
    """Every product asked in every form; the yes/no form names the next product's different maker."""
    makers = [maker for _, maker in PRODUCTS]
    rows = []
    for index, (product, maker) in enumerate(PRODUCTS):
        wrong = next(m for m in makers[index + 1 :] + makers[:index] if m != maker)
        for form_index, form in enumerate(FORMS):
            rows.append({"id": f"otherlab-{index:02d}-{form_index}", "text": form.format(product=product, wrong=wrong)})
    return rows
