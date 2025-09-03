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

"""
Hugging Face Dataset Schema Inspection Tool

For usage instructions and examples, see:
https://github.com/marin-community/marin/blob/main/docs/recipes/add_dataset.md
"""

import argparse
import json
import warnings
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
from datasets.utils.info_utils import VerificationMode


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Objects with attributes (like PIL Images)
        return f"<{obj.__class__.__name__} object>"
    elif not isinstance(obj, (str | int | float | bool | type(None))):
        # Other non-serializable types
        return str(obj)
    else:
        return obj


def _feature_dtype(feature):
    """Recursively extract a dtype string from a datasets Feature."""
    dtype = getattr(feature, "dtype", None)
    if dtype is not None:
        return str(dtype)
    nested = getattr(feature, "feature", None)
    if nested is not None:
        return f"list[{_feature_dtype(nested)}]"
    return str(feature)


def _is_string_feature(feature) -> bool:
    """Return True if the feature (possibly nested) stores strings."""
    dtype = getattr(feature, "dtype", None)
    if dtype == "string":
        return True
    nested = getattr(feature, "feature", None)
    if nested is not None:
        return _is_string_feature(nested)
    return False


def get_schema(
    dataset_name: str,
    split: str | None = None,
    config_name: str | None = None,
    trust_remote_code: bool = False,
) -> dict:
    """
    Get the schema of a Hugging Face dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'wikitext', 'c4')
        split: Specific split to analyze (optional)
        config_name: Config name for datasets with multiple configs (optional)
        trust_remote_code: Allow execution of remote code for custom datasets

    Returns:
        Dictionary with:
        - splits: List of available splits
        - text_field_candidates: List of fields likely containing text data
        - sample_row: Example data row (JSON-serializable)
        - features: Dictionary of field names to types
        - warning: Warning message if no text fields found

    Raises:
        ValueError: If config name is required but not provided
        Exception: For various dataset loading errors
    """
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Handle config selection intelligently
        if not config_name:
            try:
                configs = get_dataset_config_names(dataset_name)
                if not configs:
                    # No configs needed
                    config_name = None
                elif len(configs) == 1:
                    # Only one config, use it automatically
                    config_name = configs[0]
                elif "default" in configs and len(configs) == 1:
                    # Single default config
                    config_name = "default"
                else:
                    # Multiple configs, user must choose
                    raise ValueError(f"Config name is required. Available configs: {configs}")
            except ValueError:
                # Re-raise config errors
                raise
            except Exception:
                # If config detection fails, try without config
                config_name = None

        # Check for subsets and raise if they exist
        # Note: Hugging Face datasets don't typically expose "subsets" directly,
        # but if we encounter them in the future, we should raise here
        try:
            # Load minimal dataset info to check for subsets
            dataset_info = load_dataset(
                dataset_name, config_name, streaming=True, split="train", trust_remote_code=trust_remote_code
            )
            if hasattr(dataset_info, "info") and hasattr(dataset_info.info, "splits"):
                # This is a placeholder check - adapt as needed when subsets are encountered
                pass
        except Exception:
            # If we can't load for subset checking, continue with normal flow
            pass

        # Get splits without downloading data
        try:
            splits = get_dataset_split_names(dataset_name, config_name)
        except Exception:
            splits = ["train"]  # Fallback

        if split and split in splits:
            splits = [split]

        # Load minimal data to get schema
        stream_kwargs = {"streaming": True, "split": splits[0], "verification_mode": VerificationMode.NO_CHECKS}
        if config_name:
            stream_kwargs["name"] = config_name
        if trust_remote_code:
            stream_kwargs["trust_remote_code"] = True

        # Get features and sample
        stream = load_dataset(dataset_name, **stream_kwargs)
        features = stream.features

        # Compute feature dtypes and candidate text fields
        feature_dtypes = {k: _feature_dtype(v) for k, v in features.items()}
        text_candidates = {k for k, v in features.items() if "text" in k.lower() or _is_string_feature(v)}

        # Get sample without downloading full dataset
        try:
            sample = next(iter(stream))
            # Make sample JSON-serializable
            sample = make_json_serializable(sample)
        except Exception:
            sample = {}

        result = {
            "splits": splits,
            "text_field_candidates": list(text_candidates),
            "sample_row": sample,
            "features": feature_dtypes,
        }

        # Add warning if no text fields found
        if not text_candidates:
            result["warning"] = (
                "No obvious text fields found. This dataset may not be suitable for text-based pretraining."
            )

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get schema information from a Hugging Face dataset.",
        epilog="Example: %(prog)s wikitext --config_name wikitext-103-v1",
    )
    parser.add_argument("dataset_name", help='Dataset name (e.g., "wikitext", "c4")')
    parser.add_argument("--config_name", help="Config name for datasets with multiple configs")
    parser.add_argument("--split", help="Specific split to analyze")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Allow execution of remote code for custom datasets"
    )
    parser.add_argument("--output", default="json", choices=["json", "yaml"], help="Output format (default: json)")
    args = parser.parse_args()

    try:
        schema = get_schema(args.dataset_name, args.split, args.config_name, args.trust_remote_code)
        if args.output == "json":
            print(json.dumps(schema, indent=2))
        elif args.output == "yaml":
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML is required for YAML output. Install with `pip install pyyaml`.") from None
            print(yaml.dump(schema, sort_keys=False))
    except ValueError as e:
        # Handle config errors gracefully for CLI users
        if "Available configs" in str(e):
            configs = str(e).split(": ")[1].strip("[]").replace("'", "").split(", ")
            error_response = {"error": "Config name is required.", "available_configs": configs}
            if args.output == "json":
                print(json.dumps(error_response, indent=2))
            else:
                import yaml

                print(yaml.dump(error_response, sort_keys=False))
        else:
            raise
    except Exception as e:
        # For other exceptions, show a clean error message
        error_response = {"error": str(e)}
        if args.output == "json":
            print(json.dumps(error_response, indent=2))
        else:
            try:
                import yaml

                print(yaml.dump(error_response, sort_keys=False))
            except ImportError:
                print(json.dumps(error_response, indent=2))
