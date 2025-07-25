import argparse
import json
from datasets import load_dataset, DatasetDict


def get_schema(dataset_name: str, split: str = None) -> dict:
    try:
        dataset = load_dataset(dataset_name)
        splits = list(dataset.keys()) if isinstance(dataset, DatasetDict) else ['default']
        if split and split in splits:
            splits = [split]
        subsets = []  # Placeholder: add logic if subsets exist in future
        text_candidates = set()
        for s in splits:
            features = dataset[s].features
            for k, v in features.items():
                if 'text' in k.lower() or getattr(v, 'dtype', None) == 'string':
                    text_candidates.add(k)
        sample = dataset[splits[0]][0] if splits else {}
        return {
            'splits': splits,
            'subsets': subsets,
            'text_field_candidates': list(text_candidates),
            'sample_row': sample
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--output', default='json', choices=['json', 'yaml'])
    args = parser.parse_args()
    schema = get_schema(args.dataset_name, args.split)
    if args.output == 'json':
        print(json.dumps(schema, indent=2))
    elif args.output == 'yaml':
        try:
            import yaml
        except ImportError:
            raise ImportError('PyYAML is required for YAML output. Install with `pip install pyyaml`.')
        print(yaml.dump(schema, sort_keys=False)) 