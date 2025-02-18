import ray
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@ray.remote(resources={"TPU": 1, "TPU-v6e-8-head": 1})
def test_gte_inference():
    device = xm.xla_device()
    print(f"Using device: {device}")

    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms",
    ]

    model_path = "Alibaba-NLP/gte-base-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True, output_hidden_states=False, num_labels=5
    )
    model.to(device)

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt")
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    outputs = model(**batch_dict)
    print(outputs.logits)
    probs = F.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    print(preds)
    # print(outputs.last_hidden_state.shape)
    # embeddings = outputs.last_hidden_state[:, -1]
    # print(embeddings.shape)

    # # (Optionally) normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:1] @ embeddings[1:].T) * 100
    # print(scores.tolist())


if __name__ == "__main__":
    x = ray.get(test_gte_inference.remote())
