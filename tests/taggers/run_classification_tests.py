#!/usr/bin/env python3
"""
Test runner for classification inference pipeline.

Usage:
    python tests/run_classification_tests.py

Or with pytest:
    pytest tests/test_classification_integration.py -v
    pytest tests/test_classification_inference.py -v
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_integration_tests():
    """Run the simple integration tests"""
    print("=" * 60)
    print("Running Classification Inference Integration Tests")
    print("=" * 60)

    try:
        from tests.taggers.test_classification_integration import (
            test_multiple_files_processing,
            test_resumption_functionality,
            test_single_file_processing,
            test_streaming_reading,
        )

        print("\n1. Testing single file processing...")
        test_single_file_processing()
        print("✅ Single file processing test passed")

        print("\n2. Testing multiple files processing...")
        test_multiple_files_processing()
        print("✅ Multiple files processing test passed")

        print("\n3. Testing resumption functionality...")
        test_resumption_functionality()
        print("✅ Resumption functionality test passed")

        print("\n4. Testing streaming reading...")
        test_streaming_reading()
        print("✅ Streaming reading test passed")

        print("\n" + "=" * 60)
        print("🎉 All integration tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def run_with_pytest():
    """Run tests using pytest"""
    print("=" * 60)
    print("Running tests with pytest...")
    print("=" * 60)

    try:
        import pytest

        # Run integration tests
        print("\nRunning integration tests...")
        result1 = pytest.main(["tests/test_classification_integration.py", "-v", "--tb=short"])

        # Run comprehensive tests
        print("\nRunning comprehensive tests...")
        result2 = pytest.main(["tests/test_classification_inference.py", "-v", "--tb=short"])

        if result1 == 0 and result2 == 0:
            print("\n🎉 All pytest tests passed!")
            return True
        else:
            print("\n❌ Some pytest tests failed")
            return False

    except ImportError:
        print("pytest not available, skipping pytest tests")
        return True


def main():
    """Main test runner"""
    print("Classification Inference Pipeline Test Suite")
    print("=" * 60)

    success = True

    # Run integration tests
    if not run_integration_tests():
        success = False

    # Try to run with pytest if available
    if not run_with_pytest():
        success = False

    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
