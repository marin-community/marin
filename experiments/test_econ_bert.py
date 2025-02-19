import ray

from marin.processing.classification.classifier import AutoClassifier

MODEL_PATH = "/opt/gcsfuse_mount/economic-bert"


@ray.remote(resources={"TPU": 1, "TPU-v6e-8-head": 1})
def test_econ_bert():
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     MODEL_PATH, trust_remote_code=True, output_hidden_states=False
    # )
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    texts = [
        "The capital of China is Beijing.",
        "This is the best thing ever.",
        "The economy of the United States is the largest in the world. The GDP of the United States is $20 trillion."
        "China has a GDP of $14 trillion.",
        "What's up with the stock market?",
    ]
    batch = {"text": texts}
    # inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # outputs = model(**inputs)
    # print(outputs.logits)
    classifier = AutoClassifier(MODEL_PATH, "label", "gte", max_length=512)
    print(classifier(batch))


if __name__ == "__main__":
    x = ray.get(test_econ_bert.remote())
