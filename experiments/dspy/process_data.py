# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import dspy
import json
import os
import random
from collections.abc import Callable
from typing import Any
from dspy.datasets.dataloader import DataLoader
from dspy.primitives.prediction import Prediction
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
from transformers import AutoTokenizer
from experiments.dspy.inference_server import start_inference_server
from experiments.dspy.programs.simplified_baleen import SimplifiedBaleen
from experiments.dspy.programs.claim_verification import ClaimVerification
from experiments.dspy.programs.field_extraction import FieldExtraction
from experiments.dspy.adapters.baml import BAMLAdapter
from experiments.dspy.metrics import claim_verification_metric, field_extraction_metric

# Initialize Qwen tokenizer (Open model, no login required)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat", trust_remote_code=True)

# ==========================================
# 1. Data Loading
# ==========================================

def load_hotpotqa():
    # Load HotPotQA (using dspy built-in or HF)
    # For simplicity using dspy.datasets if available or loading from HF
    dl = DataLoader()
    dataset = dl.from_huggingface(
        "hotpotqa/hotpot_qa",
        "fullwiki",
        split="train",
        input_keys=("question",),
    )
    return dataset

def load_hover():
    # Placeholder for HoVer loading
    # Ideally load from HF: datasets.load_dataset("hover")
    # Returning a list of dspy.Example
    # Format: dspy.Example(question=..., answer=...)
    dl = DataLoader()
    dataset = dl.from_huggingface(
        "Dzeniks/hover",
        split="train",
        input_keys=("claim", "evidence",),  # type: ignore
    )
    return dataset

def load_fhir(file_path="data/note.json"):
    # Using inline dummy data to avoid broken external links (404 issues)
    # This ensures the pipeline runs reliably for demonstration.
    data = [
        {
            "record_id": "1",
            "note": "Patient presents with severe headache and nausea. BP 140/90. Prescribed acetaminophen."
        },
        {
            "record_id": "2", 
            "note": "Follow-up for diabetes management. A1C is 7.2. Continue Metformin 500mg bid."
        }
    ]

    examples = []
    for item in data:
        # item keys: record_id, note
        examples.append(dspy.Example(note=item['note'], record_id=item['record_id']).with_inputs("note", "record_id"))
    return examples

# ==========================================
# 2. Trace Collection
# ==========================================

def format_data_for_finetuning(
    trace: tuple[Any, dict[str, Any], Prediction], adapter: dspy.Adapter
) -> list[dict[str, str]]:
    pred, inputs, outputs = trace

    demos = pred.demos if hasattr(pred, "demos") else []
    input_chat = adapter.format(
        signature=pred.signature,
        demos=demos,
        inputs=inputs,
    )
    output_chat = adapter.format_assistant_message_content(
        signature=pred.signature,
        outputs=outputs.toDict(),
    )
    return [*input_chat, {"role": "assistant", "content": output_chat}]

def collect_traces_for_module(
    module: dspy.Module,
    examples: list[dspy.Example],
    num_traces: int,
    metric: Callable
) -> list[list[dict[str, str]]]:
    print(f"Collecting traces for {module.__class__.__name__}...")

    # Check if we have enough examples
    # Randomly sample num_traces examples (or all if insufficient)
    if len(examples) >= num_traces:
        selected_examples = random.sample(examples, num_traces)
    else:
        selected_examples = examples

    traces = bootstrap_trace_data(module, selected_examples, num_threads=32, metric=metric)
    all_finetuning_data = []
    for trace_data in traces:
        trace_info = trace_data["trace"]
        assert metric is not None
        assert trace_data["score"] is not None

        if trace_data["score"] == 0:
            continue

        for ti in trace_info:
            assert dspy.settings.adapter is not None
            finetuning_data = format_data_for_finetuning(ti, dspy.settings.adapter)
            all_finetuning_data.append(finetuning_data)
    return all_finetuning_data


# ==========================================
# 3. Filtering and Sampling
# ==========================================

def filter_and_sample(traces: list[dict[str, Any]], final_count: int = 3000, max_tokens: int = 2048):
    filtered = []
    for trace in traces:
        # Serialize trace to string for token counting
        # This is a rough approximation
        try:
            # We convert to string to estimate tokens.
            # We use a custom encoder or str() for non-serializable objects
            s = json.dumps(trace, default=lambda x: str(x))
            token_count = len(tokenizer.encode(s))

            if token_count <= max_tokens:
                filtered.append(trace)
        except Exception as e:
            print(f"Error processing trace: {e}")
            continue

    if len(filtered) > final_count:
        return random.sample(filtered, final_count)
    return filtered

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    # Configure DSPy
    # NOTE: You must provide a valid LLM and RM for this to work.
    # Example configuration:
    # lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=1000)
    # colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    # dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

    # For this script to run in a CI/Sandbox without keys, we check environment variables
    # or assume the user will run this with proper setup.
    
    # We are using OpenAI, so we don't need to start a local TPU inference server.
    # start_inference_server(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", port=8000, device_type="tpu")
    # time.sleep(10)
    
    # 1. Data Loading (Moved up to build index)
    print("Loading data...")
    hotpot_data = load_hotpotqa()
    hover_data = load_hover()
    fhir_data = load_fhir()

    print("Configuring DSPy...")
    # Configure Retrieval Model
    colbert_url = os.environ.get("COLBERT_SERVER_URL")
    if colbert_url:
        print(f"Using configured ColBERT server: {colbert_url}")
        rm = dspy.ColBERTv2(url=colbert_url)
    else:
        print("Warning: COLBERT_SERVER_URL not set. Switch to Local BM25S Retriever.")
        
        try:
            import bm25s
            import Stemmer
        except ImportError:
            raise ImportError("Please install bm25s: `pip install bm25s[full]` or `pip install bm25s PyStemmer`")

        class BM25Retriever(dspy.Retrieve):
            def __init__(self, data_source, k=3):
                super().__init__(k=k)
                self.k = k
                self.retriever = None
                self._build_index(data_source)

            def _build_index(self, examples):
                print("Building BM25 index from dataset contexts...")
                
                corpus = []
                self.corpus_map = [] # To map index ID back to text
                
                seen = set()
                
                for ex in examples:
                    # Strategy: Use available context or fallback to specific fields
                    texts_to_index = []
                    
                    if hasattr(ex, "context") and isinstance(ex.context, list):
                        # HotPotQA style context: list of [title, sentences]
                        for item in ex.context:
                            if isinstance(item, list) and len(item) >= 2:
                                title = item[0]
                                sentences = item[1]
                                text = f"{title}\n{' '.join(sentences)}"
                                texts_to_index.append(text)
                            elif isinstance(item, str):
                                texts_to_index.append(item)
                                
                    elif hasattr(ex, "note"): # FHIR style
                        texts_to_index.append(ex.note)
                        
                    elif hasattr(ex, "evidence"): # HoVer style
                        if isinstance(ex.evidence, str):
                            texts_to_index.append(ex.evidence)
                        elif isinstance(ex.evidence, list):
                             texts_to_index.append(" ".join([str(e) for e in ex.evidence]))

                    # Add unique texts to corpus
                    for text in texts_to_index:
                        if text and text not in seen:
                            corpus.append(text)
                            seen.add(text)
                            self.corpus_map.append(text)
                
                if not corpus:
                    print("Warning: No context found to index. Using dummy corpus.")
                    corpus = ["This is a placeholder document to prevent empty index errors."]
                    self.corpus_map = corpus

                # Create BM25S retriever
                self.retriever = bm25s.BM25(corpus=corpus)
                self.retriever.index(bm25s.tokenize(corpus, stemmer=Stemmer.Stemmer("english")))
                print(f"BM25 Index built with {len(corpus)} documents.")

            def __call__(self, query, k=None, **kwargs):
                k = k if k else self.k
                # Dynamically adjust k to avoid "k > corpus_size" error
                available_docs = len(self.corpus_map)
                if available_docs == 0:
                    return [""] * k # Should catch empty case, but just in case
                
                safe_k = min(k, available_docs)
                
                # Query the index
                # bm25s.retrieve returns (docs, scores)
                results, _ = self.retriever.retrieve(bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english")), k=safe_k)
                
                # results is a list of lists (batch size 1)
                found_docs = [doc for doc in results[0]]
                
                # If we retrieved fewer than k (because corpus was small), pad with the last doc or empty
                # DSPy generally expects k results if asked, though list length variance might be handled.
                # For safety, let's just return what we found.
                return found_docs

        # Use HotPotQA data to populate the Knowledge Base (Source of Truth)
        # This makes the system "closed-book" on the training set, which is perfect for generating adaptation traces.
        rm = BM25Retriever(hotpot_data if hotpot_data else fhir_data, k=3)

    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.settings.configure(lm=lm, rm=rm, adapter=BAMLAdapter())

    # 2. Modules
    baleen = SimplifiedBaleen()
    claim_verifier = ClaimVerification()
    fhir_module = FieldExtraction()

    # 3. Collect Traces
    # Total target 4000-6000
    # Distribute across datasets

    all_traces = []

    # HotpotQA
    if hotpot_data:
        print(f"Collecting HotpotQA traces ({len(hotpot_data)} available)...")
        traces = collect_traces_for_module(baleen, hotpot_data, num_traces=1, metric=dspy.evaluate.answer_exact_match)
        all_traces.extend([{"dataset": f"hotpotqa_{i}", "chat": t} for i, t in enumerate(traces)])

    # HoVer
    if hover_data:
        print(f"Collecting HoVer traces ({len(hover_data)} available)...")
        traces = collect_traces_for_module(claim_verifier, hover_data, num_traces=1, metric=claim_verification_metric)
        all_traces.extend([{"dataset": f"hover_{i}", "chat": t} for i, t in enumerate(traces)])

    # FHIR
    if fhir_data:
        print(f"Collecting FHIR traces ({len(fhir_data)} available)...")
        traces = collect_traces_for_module(fhir_module, fhir_data, num_traces=1, metric=field_extraction_metric)
        all_traces.extend([{"dataset": f"fhir_{i}", "chat": t} for i, t in enumerate(traces)])

    # 4. Filter and Save
    print(f"Total traces collected: {len(all_traces)}")
    print(f"Total traces collected: {len(all_traces)}")
    # Sampling
    if len(all_traces) > 3000:
        final_dataset = random.sample(all_traces, 3000)
    else:
        final_dataset = all_traces

    output_file = "experiments/dspy/format_adaptation_dataset.jsonl"
    
    # Initialize tokenizer for formatting (using the Llama-3 tokenizer defined globally or Qwen if preferred)
    # Using the globally defined 'tokenizer' which is Llama-3.1-8B-Instruct. 
    # NOTE: Ideally this should match the training model (Qwen), but Llama 3 chat template is standard enough for now
    # or we can re-initialize a Qwen tokenizer here. Let's stick to the loaded one for simplicity 
    # as strict template matching happens in SFT script if we passed list, but here we pre-format.
    
    with open(output_file, "w") as f:
        for item in final_dataset:
            # item is {"dataset": "...", "chat": [...]}
            try:
                chat_content = item.get("chat")
                if not chat_content:
                    continue
                
                # Apply chat template to turn list of dicts into a single string
                # We use tokenize=False to get the raw string
                formatted_text = tokenizer.apply_chat_template(chat_content, tokenize=False)
                
                # Save as simple {"text": "..."} for SFTTrainer
                f.write(json.dumps({"text": formatted_text}) + "\n")
            except Exception as e:
                print(f"Skipping item due to formatting error: {e}")

    print(f"Saved {len(final_dataset)} formatted conversational traces to {output_file}")
