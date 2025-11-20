import dspy
import json
import os
import random
import tqdm
from typing import List, Dict, Any
from dspy.datasets.dataloader import DataLoader
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
from transformers import AutoTokenizer

from experiments.dspy.programs.simplified_baleen import SimplifiedBaleen
from experiments.dspy.programs.claim_verification import ClaimVerification
from experiments.dspy.programs.field_extraction import FieldExtraction
from experiments.dspy.adapters.baml import BAMLAdapter

# Initialize Llama 3 tokenizer
lm = dspy.LM(model="localhost:8000/")
adapter = BAMLAdapter()
dspy.configure(lm=lm, adapter=adapter, rm=None)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# ==========================================
# 1. Data Loading
# ==========================================

def load_hotpotqa():
    # Load HotPotQA (using dspy built-in or HF)
    # For simplicity using dspy.datasets if available or loading from HF
    dl = DataLoader()
    dataset = dl.from_huggingface(
        "hotpotqa/hotpot_qa", 
            split="train",
            input_keys=("question",),
            fields=("answer",),
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
        fields=("label",),
    )
    return dataset

def download_fhir_data(output_path="data/note.json"):
    if os.path.exists(output_path):
        return
    
    url = "https://raw.githubusercontent.com/prrao87/structured-outputs/main/data/note.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.system(f"curl -k -L {url} -o {output_path}")

def load_fhir(file_path="data/note.json"):
    download_fhir_data(file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # item keys: record_id, note
        examples.append(dspy.Example(note=item['note'], record_id=item['record_id']).with_inputs("note", "record_id"))
    return examples

# ==========================================
# 2. Trace Collection
# ==========================================

def collect_traces_for_module(module: dspy.Module, examples: List[dspy.Example], num_traces: int) -> List[str]:

    print(f"Collecting traces for {module.__class__.__name__}...")
    # Manual collection loop
    traces = []
    
    # Check if we have enough examples
    # Randomly sample num_traces examples (or all if insufficient)
    if len(examples) >= num_traces:
        selected_examples = random.sample(examples, num_traces)
    else:
        selected_examples = examples

    pred = module.batch(examples=selected_examples, num_threads=32)
    return lm.inspect_history(n=len(selected_examples))

# ==========================================
# 3. Filtering and Sampling
# ==========================================

def filter_and_sample(traces: List[str], final_count: int = 3000, max_tokens: int = 2048):
    # Heuristics: Length
    
    filtered = []
    for trace in traces:
        # Serialize prediction to string for token counting
        # This is a rough approximation, ideally we format it exactly as the model sees it
        
        token_count = len(tokenizer.encode(trace))
        
        if token_count <= max_tokens:
            filtered.append(trace)
        
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
    
    # Placeholder configuration to prevent immediate crash if run without setup
    try:
         lm = dspy.LM(model="openai/gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY", "dummy"))
         dspy.settings.configure(lm=lm)
    except Exception:
         pass

    # 1. Data
    print("Loading data...")
    hotpot_data = load_hotpotqa()
    hover_data = load_hover()
    fhir_data = load_fhir()
    
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
        traces = collect_traces_for_module(baleen, hotpot_data, num_traces=1500)
        all_traces.extend([{"dataset": "hotpotqa", **t} for t in traces])
        
    # HoVer
    if hover_data:
        print(f"Collecting HoVer traces ({len(hover_data)} available)...")
        traces = collect_traces_for_module(claim_verifier, hover_data, num_traces=1500)
        all_traces.extend([{"dataset": "hover", **t} for t in traces])
        
    # FHIR
    if fhir_data:
        print(f"Collecting FHIR traces ({len(fhir_data)} available)...")
        traces = collect_traces_for_module(fhir_module, fhir_data, num_traces=1500)
        all_traces.extend([{"dataset": "fhir", **t} for t in traces])
        
    # 4. Filter and Save
    print(f"Total traces collected: {len(all_traces)}")
    final_dataset = filter_and_sample(all_traces, final_count=3000, max_tokens=4096)
    
    output_file = "experiments/dspy/format_adaptation_dataset.json"
    with open(output_file, "w") as f:
        # Serialize predictions if needed (they are objects)
        # For simplicity, just dumping inputs/predictions structure
        serializable = []
        for item in final_dataset:
            # Convert dspy.Prediction to dict if needed
            pred = item['prediction']
            if isinstance(pred, str):
                 item['prediction'] = {"result": pred}
            elif hasattr(pred, 'toDict'):
                item['prediction'] = pred.toDict()
            elif hasattr(pred, '__dict__'):
                item['prediction'] = vars(pred)
            serializable.append(item)
            
        json.dump(serializable, f, indent=2)
        
    print(f"Saved {len(final_dataset)} trajectories to {output_file}")
