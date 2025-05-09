import json
from collections import defaultdict

from nltk import ngrams

from train_test_overlap.data_overlap_spec import (
    DataOverlapStatsKey,
    EntryDataOverlapKey,
    LightInstance,
    LightScenario,
    LightScenarioKey,
    ObjectSpec,
    OverlapProtocolSpec,
)
from train_test_overlap.utils import DefaultTokenizer

PART_INPUT: str = "input"
PART_REF: str = "references"


# type alias for overlap-related data structures
Ngram = tuple[str, ...]
NgramIndex = dict[int, dict[Ngram, set[EntryDataOverlapKey]]]
NgramCounter = dict[EntryDataOverlapKey, dict[Ngram, int]]


# Declare global tokenizer that will be used throughout module
global tokenizer
tokenizer = DefaultTokenizer()


def load_light_scenarios_from_jsonl(path: str) -> list[LightScenario]:
    """
    Create a list of light scenarios from a jsonl file, where each json represents a LightScenario object.

    Input file format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...
    """

    def create_light_instance_from_dict(instance_dict: dict) -> LightInstance:
        return LightInstance(input=instance_dict[PART_INPUT], references=instance_dict[PART_REF], id=instance_dict["id"])

    light_scenarios: list[LightScenario] = []
    light_scenario_jsons = open(path, "r").readlines()
    for light_scenario_json in light_scenario_jsons:
        light_scenario_dict: dict = json.loads(light_scenario_json)

        light_scenario_key_dict: dict = light_scenario_dict["scenario_key"]
        # if the light_scenarios are exported from helm, they will have a scenario_spec field
        scenario_spec = ObjectSpec(**light_scenario_key_dict["scenario_spec"])
        light_scenario_key = LightScenarioKey(scenario_spec=scenario_spec, split=light_scenario_key_dict["split"])
        light_instances: list[LightInstance] = [
            create_light_instance_from_dict(instance_dict) for instance_dict in light_scenario_dict["instances"]
        ]
        light_scenarios.append(LightScenario(scenario_key=light_scenario_key, instances=light_instances))
    return light_scenarios


def create_ngram_index(
    light_scenarios: list[LightScenario],
    n_values: list[int],
    stats_key_counts: dict[DataOverlapStatsKey, int],
) -> NgramIndex:
    """
    Given a list of scenarios and n values, initialize ngram_index.
    stats_key_counts is passed in and updated, counting the number of times a stats_key occurs.
    A stats_key is a unique id of the n-gram length and scenario input id
    """
    ngram_index: NgramIndex = {n: {} for n in n_values}
    for scenario in light_scenarios:
        for n in n_values:
            # for each n_gram scenario pair create a unique key
            stats_key = DataOverlapStatsKey(
                light_scenario_key=scenario.scenario_key, overlap_protocol_spec=OverlapProtocolSpec(n=n)
            )
            stats_key_counts[stats_key] = len(scenario.instances)
            for _i, instance in enumerate(scenario.instances):
                instance_id = instance.id
                assert instance_id is not None
                input_tokens = tokenizer.tokenize(instance.input)
                # For each n_gram in the input text
                for input_ngram in ngrams(input_tokens, n):
                    if input_ngram not in ngram_index[n]:
                        ngram_index[n][input_ngram] = set()
                    ngram_index[n][input_ngram].add(
                        EntryDataOverlapKey(stats_key=stats_key, instance_id=instance_id, part=PART_INPUT)
                    )

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = tokenizer.tokenize(reference)
                    for reference_ngram in ngrams(reference_unigrams, n):
                        if reference_ngram not in ngram_index[n]:
                            ngram_index[n][reference_ngram] = set()
                        ngram_index[n][reference_ngram].add(
                            EntryDataOverlapKey(stats_key=stats_key, instance_id=instance_id, part=PART_REF)
                        )
    return ngram_index


def compute_all_data_overlap(
    document_iterator,
    ngram_index: NgramIndex,
    stats_key_to_input_ids: defaultdict[DataOverlapStatsKey, set[str]],
    stats_key_to_reference_ids: defaultdict[DataOverlapStatsKey, set[str]],
    entry_overlap_key_to_ngram_counts: defaultdict[EntryDataOverlapKey, defaultdict[str, int]],
    output_ngrams: bool,
) -> None:
    """
    Process documents from an iterator for data overlap computation.

    Args:
        document_iterator: Iterator yielding (document_text, source_info) tuples
        ngram_index: The ngram index that maps from ngrams to overlap stats
        stats_key_to_input_ids: Dict to keep track of input_ids that are overlapping
        stats_key_to_reference_ids: Dict to keep track of reference_ids that are overlapping
        entry_overlap_key_to_ngram_counts: a dict mapping the key to the overlapping ngrams
        output_ngrams: whether we should output ngrams
    """
    for document_text, source_info in document_iterator:
        try:
            compute_document_data_overlap(
                document=document_text,
                ngram_index=ngram_index,
                stats_key_to_input_ids=stats_key_to_input_ids,
                stats_key_to_reference_ids=stats_key_to_reference_ids,
                entry_overlap_key_to_ngram_counts=entry_overlap_key_to_ngram_counts,
                output_ngrams=output_ngrams,
            )
        except Exception as e:
            print(f"Failed to process document from {source_info}: {e!s}", flush=True)
            continue


def compute_document_data_overlap(
    document: str,
    ngram_index: NgramIndex,
    stats_key_to_input_ids: defaultdict[DataOverlapStatsKey, set[str]],
    stats_key_to_reference_ids: defaultdict[DataOverlapStatsKey, set[str]],
    entry_overlap_key_to_ngram_counts: defaultdict[EntryDataOverlapKey, defaultdict[str, int]],
    output_ngrams: bool,
) -> None:
    """
    Given a document, compute a overlap stats for each n and each scenario. The function
    writes to the overlap stats directly and does not return anything.

    ngram_index: The ngram index that maps from ngrams to overlap stats

    tokenizer: The tokenizer used to break the document into tokens

    stats_key_to_input_ids: Dict to keep track of input_ids that are overlapping

    stats_key_to_reference_ids: Dict to keep track of reference_ids that are overlapping

    entry_overlap_key_to_ngram_counts: a dict mapping the key to the overlapping ngrams

    output_ngrams: whether we should output ngrams
    """
    document_tokens = tokenizer.tokenize(document)
    # for all values of n, go through the n_grams for the document.
    # if we find a hit
    for n in ngram_index.keys():
        for document_ngram in ngrams(document_tokens, n):
            # checks for n-gram membership
            if document_ngram in ngram_index[n]:
                # If it's present that means there's a dict entry for the value
                # of n with the same ngram, so we increment the count of overlap by one
                for entry_overlap_key in ngram_index[n][document_ngram]:
                    instance_id = entry_overlap_key.instance_id
                    part = entry_overlap_key.part
                    if part == PART_INPUT:
                        stats_key_to_input_ids[entry_overlap_key.stats_key].add(instance_id)
                    elif part == PART_REF:
                        stats_key_to_reference_ids[entry_overlap_key.stats_key].add(instance_id)
                    if output_ngrams:
                        entry_overlap_key_to_ngram_counts[entry_overlap_key][document_ngram] += 1
