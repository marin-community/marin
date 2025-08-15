#!/usr/bin/env python3
"""
Test client for the Marin Model API.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:9123"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def chat(
        prompt,
        max_length=50,
        max_think_effort=None,
        temperature=0.1,
        do_sample=True,
        stream=True,
        output_file=None
    ):
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "max_think_effort": max_think_effort,
        "temperature": temperature,
        "do_sample": do_sample,
        "stream": stream
    }
    response = requests.post(f"{API_BASE_URL}/chat", json=payload)
    if response.status_code == 200:
        result = response.json()
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"Input prompt: {result['request_params']['input_prompt']}\n")
                f.write(f"Response:\n{result['choices'][0]['message']['content']}\n")
                f.write("-" * 40 + "\n")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Generation time: {result['statistics']['generation_time']:.2f}s")
    else:
        print(f"Error: {response.text}")


long_prompt = """You are given a positive integer k and an array a_1, a_2, …, a_n of non-negative distinct integers not smaller than k and not greater than 2^c-1.\n\nIn each of the next k seconds, one element is chosen randomly equiprobably out of all n elements and decreased by 1.\n\nFor each integer x, 0 ≤ x ≤ 2^c - 1, you need to find the probability that in the end the bitwise XOR of all elements of the array is equal to x. \n\nEach of these values can be represented as an irreducible fraction p/q, and you need to find the value of p ⋅ q^{-1} modulo 998 244 353.\nInput\n\nThe first line of input contains three integers n, k, c (1 ≤ n ≤ (2^c - k), 1 ≤ k ≤ 16, 1 ≤ c ≤ 16).\n\nThe second line contains n distinct integers a_1, a_2, …, a_n (k ≤ a_i ≤ 2^c-1).\n\nOutput\n\nPrint 2^c integers: the probability that the bitwise XOR is equal to x in the end for x in \\{0, 1, …, 2^c-1\\} modulo 998 244 353.\n\nExample\n\nInput\n\n\n4 1 3\n1 2 3 4\n\n\nOutput\n\n\n0 0 0 748683265 0 499122177 0 748683265"""
def main():
    """Run all tests."""
    print("Marin Model API Test Client")
    print("=" * 40)
    
    # Wait a bit for the server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)

    output_file = "llm_test.txt"
    
    print("Generation in progress...")
    try:
        chat(
            "Quickly count from 1 to 10",
            max_length=512,
            max_think_effort=256,
            temperature=0.1,
            do_sample=True,
            output_file=output_file
        )
        chat(
            long_prompt,
            max_length=1024,
            max_think_effort=None,
            temperature=0.1,
            do_sample=True,
            output_file=output_file
        )
        chat(
            "What is the capital of France?",
            max_length=700,
            max_think_effort=512,
            temperature=0.1,
            do_sample=True,
            output_file=output_file
        )
        chat(
            "Write a 50 words short story about a robot learning to paint",
            max_length=1024,
            max_think_effort=512,
            temperature=0.7,
            do_sample=True,
            output_file=output_file
        )
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:9123")
        print("\nTo start the server, run:")
        print("python3 marin/api/model_server.py")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
