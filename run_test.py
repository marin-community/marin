#!/usr/bin/env python3
"""Simple test runner for the simplified tests."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/user/marin/lib/marin/src")

from tests.transform.test_huggingface_dataset_to_eval import (
    test_hf_dataset_to_jsonl_evaluation_format,
    test_hf_dataset_to_jsonl_decontamination_format,
)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing HuggingFace Dataset Transformation")
    print("=" * 60)

    tests = [
        ("test_hf_dataset_to_jsonl_evaluation_format", test_hf_dataset_to_jsonl_evaluation_format),
        ("test_hf_dataset_to_jsonl_decontamination_format", test_hf_dataset_to_jsonl_decontamination_format),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...", end=" ")
            with tempfile.TemporaryDirectory() as tmp_dir:
                test_func(Path(tmp_dir))
            print("✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
