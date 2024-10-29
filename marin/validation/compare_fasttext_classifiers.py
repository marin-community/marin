import fasttext
from huggingface_hub import hf_hub_download


def num_non_overlapping_words(words, other_words):
    print(len(words), len(other_words))
    return len(set(words) - set(other_words))


def print_model_properties(model):
    print("\nModel dimension:", model.get_dimension())
    # Get input/output matrices shape
    input_matrix = model.get_input_matrix()
    output_matrix = model.get_output_matrix()
    print("\nInput matrix shape:", input_matrix.shape)
    print("Output matrix shape:", output_matrix.shape)

    # Test getting vectors
    test_word = model.words[0]  # Get first word in vocabulary
    print(f"\nVector for word '{test_word}':", model.get_word_vector(test_word)[:5], "...")  # Show first 5 elements
    print("Sentence vector for 'hello world':", model.get_sentence_vector("hello world")[:5], "...")

    # Get subword information for a sample word
    test_word = "hello"
    subwords, subword_ids = model.get_subwords(test_word)
    print(f"\nSubwords for '{test_word}':", subwords)
    print("Subword IDs:", subword_ids)
    print(f"Word ID for '{test_word}':", model.get_word_id(test_word))

    # Test prediction
    test_text = "This is a test sentence"
    labels, probs = model.predict(test_text)
    print(f"\nPrediction for '{test_text}':")
    print("Labels:", labels)
    print("Probabilities:", probs)

    # Model properties
    print("\nIs model quantized?", model.is_quantized())


model_path = hf_hub_download(
    repo_id="mlfoundations/fasttext-oh-eli5", filename="openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
)
model = fasttext.load_model(model_path)
huggingface_words = set(model.words)
print("Huggingface model")
print_model_properties(model)

model = fasttext.load_model("/home/gcpuser/model/model.bin")
local_words = set(model.words)
print("Local model")
print_model_properties(model)

print(f"Number of non-overlapping words: {num_non_overlapping_words(huggingface_words, local_words)}")
