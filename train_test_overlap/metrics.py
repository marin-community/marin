import json

import cattrs

from train_test_overlap.data_overlap_spec import (
    AggregateDataOverlapKey,
    AggregateOverlapMetric,
    EntryOverlapMetric,
    FrequencySpec,
    MetricProtocolSpec,
    PartialOverlapSpec,
)
from train_test_overlap.utils import asdict_without_nones


def scenario_spec_to_class(scenario_spec) -> str:
    return f"{'.'.join(scenario_spec.class_name.split('.')[-1:])}"


PART_INPUT: str = "input"
PART_REF: str = "reference"
metric_protocol_specs_list = [
    MetricProtocolSpec(PartialOverlapSpec.binary, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(0, True)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(0, True)),
    MetricProtocolSpec(PartialOverlapSpec.binary, FrequencySpec(10, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(10, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(10, True)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(10, False)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(10, True)),
]


non_weighted_metrics = [
    MetricProtocolSpec(PartialOverlapSpec.binary, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.jaccard, FrequencySpec(0, False)),
    MetricProtocolSpec(PartialOverlapSpec.token, FrequencySpec(0, False)),
]


def aggregate_metrics(path, out_path, metrics_input_path=None):
    print(f"Aggregating metrics from {path}", flush=True)
    overlap_metrics_jsons = open(path, "r").readlines()
    print(f"Aggregating {len(overlap_metrics_jsons)} metrics from {path}", flush=True)
    entry_overlap_metric_list = []
    for entry_overlap_metric_json in overlap_metrics_jsons:
        entry_overlap_metric_dict = json.loads(entry_overlap_metric_json)
        entry_overlap_metric_list.append(cattrs.structure(entry_overlap_metric_dict, EntryOverlapMetric))

    # Initialize a new dictionary for aggregated scores (mapping to list of (instance_id, metric_score))
    aggregate_score_dict: dict[tuple, list[tuple[str, float]]] = {}

    for entry_overlap_metric in entry_overlap_metric_list:
        # Extract necessary information
        instance_id = entry_overlap_metric.entry_data_overlap_key.instance_id
        stats_key = entry_overlap_metric.entry_data_overlap_key.stats_key
        part = entry_overlap_metric.entry_data_overlap_key.part
        metric_protocol_spec = entry_overlap_metric.overlap_metric.metric_protocol_spec
        if metric_protocol_spec not in non_weighted_metrics:
            continue
        metric_score = entry_overlap_metric.overlap_metric.metric_score

        # Define the aggregate key
        agg_key = (stats_key, part, metric_protocol_spec)

        # Initialize or append the (instance_id, metric_score) tuple
        if agg_key not in aggregate_score_dict:
            # skip certain scenarios
            if stats_key.light_scenario_key.scenario_spec.class_name.endswith("CopyrightScenario"):
                continue
            aggregate_score_dict[agg_key] = [(instance_id, metric_score)]
        else:
            aggregate_score_dict[agg_key].append((instance_id, metric_score))

    # Convert the aggregated data to AggregateOverlapMetric objects (including instance IDs)
    aggregate_overlap_metrics = []
    for (stats_key, part, metric_protocol_spec), entries in aggregate_score_dict.items():
        instance_ids, scores = zip(*entries, strict=False)
        aggregate_key = AggregateDataOverlapKey(stats_key=stats_key, part=part)
        aggregate_overlap_metrics.append(
            AggregateOverlapMetric(
                aggregate_data_overlap_key=aggregate_key,
                instance_ids=list(instance_ids),
                metric_scores=list(scores),
                metric_protocol_spec=metric_protocol_spec,
            )
        )

    def save_metrics_to_jsonl(overlap_metrics: list[AggregateOverlapMetric], filename: str):
        print(f"Saving {len(overlap_metrics)} metrics to {filename}", flush=True)
        with open(filename, "w") as f:
            for overlap_metric in overlap_metrics:
                d = asdict_without_nones(overlap_metric)
                # record the original GCS input path if provided, otherwise fallback to local path
                d["metrics_input_path"] = metrics_input_path or path
                # represent the partial overlap spec as its name rather than numeric code
                d["metric_protocol_spec"][
                    "partial_overlap_spec"
                ] = overlap_metric.metric_protocol_spec.partial_overlap_spec.name
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_metrics_to_jsonl(aggregate_overlap_metrics, out_path)
