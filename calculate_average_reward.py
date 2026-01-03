#!/usr/bin/env python3
"""
Script to calculate the average reward from initial.json
"""

import json
import argparse
import sys
import os
from transformers import AutoTokenizer
import numpy as np

# Add marin to path
sys.path.append("lib/marin/src")

try:
    from marin.rl.environments.math_env import MathEnv
except ImportError:
    print("Warning: Could not import MathEnv. Make sure you are running from the root of the repo.")
    MathEnv = None

# Initialize tokenizer (global to avoid reloading)
try:
    TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
except Exception as e:
    print(f"Warning: Could not load tokenizer: {e}")
    TOKENIZER = None

def calculate_average_reward(json_file_path, max_output_tokens=None):
    """
    Read a JSON file and calculate the average reward.
    
    Args:
        json_file_path: Path to the JSON file
        max_output_tokens: Optional limit on output tokens.
        
    Returns:
        Average reward value
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract rewards and responses from the data
    # The structure is: {"columns": [...], "data": [[prompt, response, reward, step], ...]}
    
    # Determine indices
    columns = data.get('columns', [])
    rows = data.get('data', [])

    # Check for nested list structure (list of lists of samples) and flatten if necessary
    # This handles cases where data is like [[sample1, sample2, ...]] or [batch1, batch2, ...] where batch is a list of samples
    if len(rows) > 0 and isinstance(rows[0], list) and len(rows[0]) > 0 and isinstance(rows[0][0], list):
        print("Detected nested list structure in data. Flattening rows...")
        new_rows = []
        for r in rows:
            if isinstance(r, list):
                new_rows.extend(r)
        rows = new_rows

    try:
        reward_idx = columns.index('reward')
        rewards = [row[reward_idx] for row in rows]
    except (ValueError, IndexError):
        # print("No 'reward' column found.")
        rewards = []
        
    try:
        response_idx = columns.index('response')
    except ValueError:
        response_idx = 1 # fallback based on comment

    try:
        prompt_idx = columns.index('prompt')
    except ValueError:
        prompt_idx = 0
    
    # Calculate average
    if len(rewards) > 0:
        average_reward = sum(rewards) / len(rewards)

        # Calculate accuracy (count of rewards == 1.0)
        correct_count = sum(1 for r in rewards if r == 1.0)
        accuracy = correct_count / len(rewards)

        # Calculate accuracy (count of rewards > 0.1)
        any_correct_count = sum(1 for r in rewards if r > 0.1)
        any_accuracy = any_correct_count / len(rewards)
    else:
        average_reward = 0

    # Calculate average output tokens
    # Calculate token statistics
    avg_tokens = 0
    min_output_tokens = 0
    max_output_tokens_stat = 0
    max_total_tokens = 0

    if TOKENIZER:
        responses = [row[response_idx] for row in rows]
        prompts_list = [row[prompt_idx] for row in rows]

        output_token_counts = []
        total_token_counts = []
        
        for p, r in zip(prompts_list, responses):
            # Output tokens
            if isinstance(r, str):
                r_ids = TOKENIZER.encode(r)
                r_len = len(r_ids)
            else:
                r_len = 0
            output_token_counts.append(r_len)

            # Input tokens
            if isinstance(p, str):
                p_ids = TOKENIZER.encode(p)
                p_len = len(p_ids)
            else:
                p_len = 0
            
            total_token_counts.append(p_len + r_len)
        
        if output_token_counts:
            avg_tokens = sum(output_token_counts) / len(output_token_counts)
            min_output_tokens = min(output_token_counts)
            max_output_tokens_stat = max(output_token_counts)
            max_total_tokens = max(total_token_counts)
    
    # Print statistics
    if len(rewards) > 0:
        print(f"Total number of samples: {len(rewards)}")
        print(f"Sum of rewards: {sum(rewards)}")
        print(f"Average reward: {average_reward}")
        print(f"Min reward: {min(rewards)}")
        print(f"Max reward: {max(rewards)}")
        print(f"Accuracy (reward == 1.0): {correct_count}/{len(rewards)} = {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Accuracy (reward > 0.1):  {any_correct_count}/{len(rewards)} = {any_accuracy:.4f} ({any_accuracy*100:.2f}%)")
    else:
        print("No rewards found in the data (skipping reward statistics).")
    
    if TOKENIZER:
        print(f"Average Output Tokens: {avg_tokens:.2f}")
        print(f"Max Output Tokens: {max_output_tokens_stat}")
        print(f"Min Output Tokens: {min_output_tokens}")
        print(f"Max Input+Output Tokens: {max_total_tokens}")
    else:
        print("Average Output Tokens: N/A (Tokenizer not loaded)")

    # Recalculate rewards using MathEnv
    if MathEnv:
        header = "\n--- Recalculating Rewards with MathEnv ---"
        if max_output_tokens is not None:
             header += f"\n(Outputs truncated to {max_output_tokens} tokens)"
        print(header)
        
        try:
            # Initialize environment to access datasets and scoring logic
            # We don't need a tokenizer for the env itself strictly if we pass it to _score_choice,
            # but good practice to pass it.
            env = MathEnv(tokenizer=TOKENIZER)
            
            # Build lookup map from processed_prompt to example for both train and eval
            prompt_to_example = {}
            # Note: accessing .train_examples and .eval_examples directly
            # We combining them to cover both possibilities
            for ex in env.train_examples:
                prompt_to_example[ex.processed_prompt] = ex
            for ex in env.eval_examples:
                prompt_to_example[ex.processed_prompt] = ex
                
            print(f"Loaded {len(prompt_to_example)} examples from MathEnv.")
            
            recalc_rewards = []
            recalc_correct = []
            recalc_format = []
            recalc_output_tokens = []
            recalc_total_tokens = []
            
            prompts = [row[0] for row in rows] # Assuming prompt is at index 0 based on known structure
            # Check prompt index to be sure, although 'prompt' is usually 0
            try:
                prompt_idx = columns.index('prompt')
            except ValueError:
                prompt_idx = 0
                
            prompts = [row[prompt_idx] for row in rows]
            responses = [row[response_idx] for row in rows]
            
            found_count = 0
            
            for prompt_raw, response in zip(prompts, responses):
                if not isinstance(prompt_raw, str) or not isinstance(response, str):
                    continue

                response_to_score = response
                # Truncate if requested and tokenizer available
                if max_output_tokens is not None and TOKENIZER:
                    ids = TOKENIZER.encode(response)
                    if len(ids) > max_output_tokens:
                        ids = ids[:max_output_tokens]
                        response_to_score = TOKENIZER.decode(ids, skip_special_tokens=True)
                
                # Extract the core prompt text.
                # Expected format includes chat templates.
                # We look for the last user message.
                parts = prompt_raw.split("<|start_header_id|>user<|end_header_id|>\n\n")
                if len(parts) > 1:
                    core_prompt = parts[-1].split("<|eot_id|>")[0]
                else:
                    # Fallback if no chat template found
                    core_prompt = prompt_raw

                # The core_prompt might exactly match processed_prompt
                if core_prompt in prompt_to_example:
                    example = prompt_to_example[core_prompt]
                    found_count += 1
                    
                    
                    reward, fmt_score, correct_score = env._score_choice(
                        example=example,
                        response_text=response_to_score,
                        finish_reason="stop", 
                        tokenizer=TOKENIZER
                    )
                    
                    recalc_rewards.append(reward)
                    recalc_format.append(fmt_score)
                    recalc_correct.append(correct_score)

                    if TOKENIZER:
                        # Recalculate tokens for the (potentially truncated) response and prompt
                        # Note: We re-encode here to be precise about what is being scored
                        r_ids = TOKENIZER.encode(response_to_score)
                        p_ids = TOKENIZER.encode(prompt_raw)
                        recalc_output_tokens.append(len(r_ids))
                        recalc_total_tokens.append(len(p_ids) + len(r_ids))
                else:
                    # Debug: print one failure
                    # if found_count == 0:
                    #     print(f"DEBUG: Could not find prompt in env. Extracted: {repr(core_prompt)}")
                    pass

            print(f"Matched {found_count} out of {len(rows)} samples with MathEnv dataset.")
            
            if found_count > 0:
                avg_recalc_reward = sum(recalc_rewards) / len(recalc_rewards)
                avg_recalc_accuracy = sum(recalc_correct) / len(recalc_correct)
                
                recalc_any_correct_count = sum(1 for r in recalc_rewards if r > 0.1)
                recalc_any_accuracy = recalc_any_correct_count / len(recalc_rewards)
                
                print("\nRecalculated Rewards Section:")
                print(f"Sum of rewards: {sum(recalc_rewards)}")
                print(f"Average reward: {avg_recalc_reward}")
                print(f"Accuracy (correctness): {sum(recalc_correct)}/{len(recalc_correct)} = {avg_recalc_accuracy:.4f} ({avg_recalc_accuracy*100:.2f}%)")
                print(f"Accuracy (reward > 0.1): {recalc_any_correct_count}/{len(recalc_rewards)} = {recalc_any_accuracy:.4f} ({recalc_any_accuracy*100:.2f}%)")
                print(f"Average Format Score: {sum(recalc_format)/len(recalc_format):.4f}")

                if TOKENIZER and recalc_output_tokens:
                    avg_recalc_tokens = sum(recalc_output_tokens) / len(recalc_output_tokens)
                    print(f"Average Output Tokens: {avg_recalc_tokens:.2f}")
                    print(f"Max Output Tokens: {max(recalc_output_tokens)}")
                    print(f"Min Output Tokens: {min(recalc_output_tokens)}")
                    print(f"Max Input+Output Tokens: {max(recalc_total_tokens)}")
            else:
                print("Could not match any prompts to recalculate rewards.")
                
        except Exception as e:
            print(f"Error during recalculation: {e}")
            import traceback
            traceback.print_exc()

    return average_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average reward from JSON table files.")
    parser.add_argument("files", nargs="*", default=[
        "temp_0.0.json",
        "temp_1.0.json",
        "marin_media_table_inference.eval_math_full_sample_table_0_8c2124772bef6ab7e85e.table.json",
        "tinker_media_table_test_samples_0_e1ad333e5c024e4fd8b7.table.json",
        "tinker_max_tokens_1024_table_test_samples_0_17f90804ad1d0ee5f0db.table.json",
        "tinker_temperature_0_max_tokens_1024_media_table_test_samples_0_ceb8ba9fde5a837639bf.table.json"
    ], help="List of JSON files to process")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Truncate response to this many tokens before scoring.")
    
    args = parser.parse_args()
    
    # If using default, ensure files exist, otherwise might be confusing if they deleted them.
    # But for now we just iterate what we have.
    
    for json_file in args.files:
        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            continue
            
        print(f"\n\n===={json_file}====")
        avg_reward = calculate_average_reward(json_file, max_output_tokens=args.max_output_tokens)
