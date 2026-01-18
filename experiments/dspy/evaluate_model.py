
import dspy
import argparse
import os
from experiments.dspy.metrics import claim_verification_metric, field_extraction_metric
from experiments.dspy.process_data import load_hotpotqa, load_hover, load_fhir
from experiments.dspy.programs.simplified_baleen import SimplifiedBaleen
from experiments.dspy.programs.claim_verification import ClaimVerification
from experiments.dspy.programs.field_extraction import FieldExtraction
from experiments.dspy.adapters.toon import ToonAdapter

def evaluate_hotpotqa(lm, num_examples=50):
    print(f"Evaluating HotPotQA (n={num_examples})...")
    data = load_hotpotqa()
    # Take validation split if available, otherwise sample
    test_set = data[:num_examples] 
    
    # Configure program
    program = SimplifiedBaleen()
    
    evaluator = dspy.Evaluate(devset=test_set, metric=dspy.evaluate.answer_exact_match, num_threads=1, display_progress=True, display_table=True)
    score = evaluator(program)
    print(f"HotPotQA Score: {score}")
    return score

def evaluate_hover(lm, num_examples=50):
    print(f"Evaluating HoVer (n={num_examples})...")
    data = load_hover()
    test_set = data[:num_examples]
    
    program = ClaimVerification()
    
    evaluator = dspy.Evaluate(devset=test_set, metric=claim_verification_metric, num_threads=1, display_progress=True, display_table=True)
    score = evaluator(program)
    print(f"HoVer Score: {score}")
    return score

def evaluate_fhir(lm, num_examples=50):
    print(f"Evaluating FHIR (n={num_examples})...")
    data = load_fhir()
    test_set = data[:num_examples]
    
    program = FieldExtraction()
    
    evaluator = dspy.Evaluate(devset=test_set, metric=field_extraction_metric, num_threads=1, display_progress=True, display_table=True)
    score = evaluator(program)
    print(f"FHIR Score: {score}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen-4B model on DSPy tasks.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen1.5-4B-Chat", help="Path to trained model or HF Hub ID")
    parser.add_argument("--adapter", type=str, choices=["json", "toon"], default="toon", help="Adapter format to use")
    parser.add_argument("--limit", type=int, default=20, help="Number of examples to evaluate per task")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    # Configure LM
    # Note: For strict local evaluation without VLLM/Server, we might use dspy.HFModel if installed,
    # or assume the user has an OpenAI-compatible server running for the trained model.
    # For now, using dspy.HFClientVLLM or generic dspy.LM if available. 
    # Providing a generic setup that assumes an OpenAI-compatible endpoint (common for serving finetunes)
    # OR falling back to HF loading if specified.
    
    # Example usage with local vllm:
    # lm = dspy.HFClientVLLM(model=args.model_path, port=8000, url="http://localhost:8000")
    
    # Example usage with dspy.HFModel (slow, local):
    # lm = dspy.HFModel(model=args.model_path)
    
    # Using a placeholder implementation that expects a local server or API key.
    # The user can adapt this connection string.
    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY")) # Placeholder for baseline
    
    if args.adapter == "toon":
        dspy.settings.configure(lm=lm, adapter=ToonAdapter())
    else:
        dspy.settings.configure(lm=lm) # Default JSON adapter
        
    print(f"Adapter: {dspy.settings.adapter}")

    evaluate_hotpotqa(lm, args.limit)
    evaluate_hover(lm, args.limit)
    evaluate_fhir(lm, args.limit)
