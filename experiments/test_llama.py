import ray
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/opt/gcsfuse_mount/perplexity-models/llama-200m"


@ray.remote(resources={"TPU": 1, "TPU-v6e-8-head": 1})
def test_inference():
    # Run inference
    device = xm.xla_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    text = ["Hello how are you?", "WTF IS THIS?"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=model.config.max_position_embeddings
    ).to(device)
    input_ids = inputs["input_ids"]
    outputs = model(**inputs)
    logits = outputs.logits
    # Calculate per-sequence perplexity
    # Shift input_ids right to create labels (next token prediction)
    labels = input_ids.clone()
    labels = labels[:, 1:]
    labels[labels == tokenizer.pad_token_id] = -100

    # Calculate loss per sequence
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shifted_logits = logits[:, :-1, :].contiguous()

    # Get loss for each token
    token_losses = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1))

    # Reshape token losses to batch x sequence
    token_losses = token_losses.view(labels.size())

    # Calculate mean loss per sequence
    # Sum losses and divide by sequence length (excluding padding)
    sequence_lengths = (labels != -100).sum(dim=1).float()
    sequence_losses = token_losses.sum(dim=1) / sequence_lengths

    # Calculate perplexity per sequence
    perplexities = torch.exp(sequence_losses)
    xm.mark_step()

    # Convert to regular Python list
    perplexities = perplexities.tolist()

    print("Perplexity per sequence:", perplexities)

    # Huggingface Perplexity
    outputs = model(**inputs, labels=input_ids)
    perplexity = torch.exp(outputs.loss)
    print("Huggingface Perplexity:", perplexity.tolist())


if __name__ == "__main__":
    ray.get(test_inference.remote())
