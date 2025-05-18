from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ObjectSpec:
    """Specifies how to construct an object."""

    # Class name of an object
    class_name: str

    # Arguments used to construct the scenario
    args: dict[str, Any]

    def __hash__(self):
        return hash((self.class_name, tuple((k, self.args[k]) for k in sorted(self.args.keys()))))


@dataclass(frozen=True)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    input: str
    """The input"""

    references: list[str]
    """References that help us evaluate"""

    id: str | None = None
    """Helm instance id"""


@dataclass(frozen=True)
class LightScenarioKey:
    """
    Key for LightScenario
    """

    scenario_spec: ObjectSpec

    split: str

    def __hash__(self):
        return hash((self.scenario_spec, self.split))


@dataclass(frozen=True)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_key: LightScenarioKey

    instances: list[LightInstance]
    """Instances of this scenario"""


@dataclass(frozen=True)
class ScenarioSpecInstanceIds:
    """
    Instance ids associated with a scenario
    """

    scenario_spec: ObjectSpec

    instance_ids: list[str]


@dataclass(frozen=True)
class GroupOverlapStats:
    """
    Dataclass that represents group data overlap stats
    e.g.
    {
        "group": "natural_qa_closedbook",
        "num_instances": 2144,
        "num_overlapping_inputs": 1,
        "num_overlapping_references": 100
    }
    """

    group: str

    num_instances: int

    num_overlapping_inputs: int

    num_overlapping_references: int

    @property
    def overlapping_input_ratio(self):
        return self.num_overlapping_inputs / self.num_instances

    @property
    def overlapping_reference_ratio(self):
        return self.num_overlapping_references / self.num_instances


@dataclass(frozen=True)
class OverlapProtocolSpec:
    """
    Specification for how we compute n-gram overlap between test and training data.

    This class defines the parameters of the n-gram comparison process,
    particularly the size of n-grams being analyzed.
    """

    n: int
    """The length of n-grams to analyze (e.g., 5 for 5-grams, 13 for 13-grams)"""


@dataclass(frozen=True)
class DataOverlapStatsKey:
    """
    A unique key that identifies a specific dataset/scenario and n-gram configuration.

    This key is used to associate overlap statistics with a specific test dataset
    and n-gram size configuration, allowing for organization of overlap results.
    """

    light_scenario_key: LightScenarioKey
    """Identifies the specific test dataset scenario and split being analyzed"""

    overlap_protocol_spec: OverlapProtocolSpec
    """Specifies the n-gram size used for this particular overlap analysis"""


@dataclass(frozen=True)
class DataOverlapStats:
    """
    Tracks overall overlap statistics between test and training datasets.

    This class records which specific test instances (by ID) have any overlap
    with the training data, separately tracking input overlaps and reference answer
    overlaps. This helps identify potential data contamination at the instance level.
    """

    data_overlap_stats_key: DataOverlapStatsKey
    """Uniquely identifies which test dataset and n-gram configuration these stats apply to"""

    num_instances: int
    """Total number of test instances being analyzed for overlap"""

    instance_ids_with_overlapping_input: list[str]
    """
    List of instance IDs from the test dataset whose input text contains n-grams
    that also appear in the training data.
    """

    instance_ids_with_overlapping_reference: list[str]
    """
    List of instance IDs from the test dataset whose reference answers contain n-grams
    that also appear in the training data.
    """


@dataclass(frozen=True)
class EntryDataOverlapKey:
    """
    Unique key representing either the input or references of a single instance in a scenario.

    This key uniquely identifies a specific text element (input or reference) from a test dataset
    instance that we're checking for overlap with training data.
    """

    stats_key: DataOverlapStatsKey
    """The key identifying which scenario, split, and n-gram size we're analyzing"""

    part: str
    """Either 'input' or 'references' - identifies whether we tracking overlap in the input text or reference answers"""

    instance_id: str
    """The unique identifier for the specific instance (e.g., test question) being analyzed"""


@dataclass(frozen=True)
class EntryOverlapNgrams:
    """
    Dataclass that represents n-gram overlap statistics between training and test data.

    This class tracks which specific n-grams from the test data were found in the training data
    and how many times each of those n-grams appeared in the training data.
    """

    entry_data_overlap_key: EntryDataOverlapKey
    """
    Identifies the specific text element (input or reference from a test instance)
    that we're tracking overlap for
    """

    overlapping_ngram_counts: list[tuple[str, int]]
    """
    List of tuples where each tuple contains:
    - First element: An n-gram (sequence of n words) that appears in both test and training data
    - Second element: The count of how many times this n-gram was found in the training data

    For example: [(('is', 'most', 'likely', 'to', 'be'), 16)] means the 5-gram
    "is most likely to be" was found 16 times in the training data.

    Higher counts indicate more frequent overlap, which may suggest greater data contamination.
    """


class PartialOverlapSpec(int, Enum):
    """
    Defines different methods for calculating overlap between test and training data.

    These different overlap calculation methods provide various ways to measure
    how much of the test data appears in the training data.
    """

    binary = 0  # Simple yes/no overlap detection
    jaccard = 1  # Jaccard similarity based overlap measurement
    token = 2  # Token-level overlap measurement

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class FrequencySpec:
    """
    Specifies how to handle frequency information in overlap calculations.

    This controls filtering and weighting of n-grams based on their frequency
    in the training data, which can help prioritize more significant overlaps.
    """

    filter_value: int
    """Filter n-grams with frequency >= filter_value; 0 means no filter"""

    weighting: bool
    """Whether to apply inverse frequency weighting (rare overlaps weighted more heavily)"""


@dataclass(frozen=True)
class MetricProtocolSpec:
    """
    Specification for how overlap metrics are computed from the raw n-gram data.

    This combines the overlap calculation method with frequency handling
    to define the complete approach for measuring overlap severity.
    """

    partial_overlap_spec: PartialOverlapSpec
    """The method used to calculate overlap (binary, jaccard, token)"""

    frequency_spec: FrequencySpec
    """How to handle frequency information in the overlap calculation"""


@dataclass(frozen=True)
class OverlapMetric:
    metric_score: float  # use 0/1 for binary, can revise as neded
    metric_protocol_spec: MetricProtocolSpec


# Output: List[EntryOverlapMetric]
@dataclass(frozen=True)
class EntryOverlapMetric:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlap_metric: OverlapMetric


@dataclass(frozen=True)
class AggregateDataOverlapKey:
    """Key representing the aggregated data overlap stats"""

    stats_key: DataOverlapStatsKey
    part: str


@dataclass(frozen=True)
class AggregateOverlapMetric:
    """Dataclass representing the aggregated overlap metrics"""

    aggregate_data_overlap_key: AggregateDataOverlapKey
    instance_ids: list[str]
    metric_scores: list[float]  # List of scores instead of a single value
    metric_protocol_spec: MetricProtocolSpec


@dataclass(frozen=True)
class AnnotatedOverlapPart:
    """
    Dataclass annotates a given scenario entry with overlaps
    """

    part: str

    annotated_entry_overlap: list[tuple[str, int]]
    """list of (word, count) where (word, count) is the 13-gram that starts with word"""

    metrics: list[OverlapMetric]


@dataclass(frozen=True)
class TotalAnnotatedEntryOverlap:
    """
    Dataclass annotates a given scenario entry with overlaps
    """

    instance: LightInstance

    stats_key: DataOverlapStatsKey

    instance_id: str

    annotated_input_overlap: AnnotatedOverlapPart
    """list of (word, count) where (word, count) is the 13-gram that starts with word"""

    annotated_ref_overlap: AnnotatedOverlapPart
    """list of (word, count) where (word, count) is the 13-gram that starts with word"""
