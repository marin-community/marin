# This is the pseudo quickstart pipeline using inhouse DAG graphs and config files.
# I will try to use as specific as possible to the above snippets.
import yaml

import marin
import scripts
from operations.utils.node import Node

# We are using two input sources, both of which are fw for the moment.

DATAPATH1 = "gs://marin-us-central2/raw/hello_world_fw/8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/resolve/8fd6e8e/data/CC-MAIN-2024-10"
DATASET1 = "hello_world_fw"
DATASET1_VERSION = "v1.0"

DATAPATH2 = "gs://marin-us-central2/raw/hello_world_fw/8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/resolve/8fd6e8e/data/CC-MAIN-2024-10"
DATASET2 = "hello_world_fw_2"
DATASET2_VERSION = "v1.0"

# load yaml config
CONFIG = yaml.load(open("quickstart.yaml", "r"), Loader=yaml.FullLoader)

EXPERIMENT = "experiment_4"
transform1_node = Node(
    func=scripts.hello_world_fw.process,
    step_name="transform1",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_path": DATAPATH1,
        "output_path": f"gs://marin-us-central2/documents/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}",
    },
)
transform2_node = Node(
    func=scripts.hello_world_fw.process,
    step_name="transform2",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_path": DATAPATH2,
        "output_path": f"gs://marin-us-central2/documents/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}",
    },
)

fasttext_node = Node(
    func=scripts.fasttext.train_fasttext,
    depends_on=[transform1_node, transform2_node],
    step_name="fasttext",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "pos_doc_path": "gs://marin-us-central2/documents/marin_instructv1/v1_olmo_mix/text",
        "neg_doc_path": [
            f"gs://marin-us-central2/documents/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}/",
            f"gs://marin-us-central2/documents/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}/",
        ],
        "pos_sampling_rate": 0.1,
        "neg_sampling_rate": 1.0,
        "output_base_path": "gs://marin-us-central2",
        "experiment": EXPERIMENT,
        "config": CONFIG["fasttext"],
    },
)

annotate_node1 = Node(
    func=marin.processing.classification.inference,
    depends_on=[fasttext_node],
    step_name="annotate",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_path": f"gs://marin-us-central2/documents/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}/",
        "output_path": f"gs://marin-us-central2/attributes/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}_fasttext/",
        "experiment": EXPERIMENT,
        "config": CONFIG["annotate"],
    },
)

annotate_node2 = Node(
    func=marin.processing.classification.inference,
    depends_on=[fasttext_node],
    step_name="annotate",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_path": f"gs://marin-us-central2/documents/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}/",
        "output_path": f"gs://marin-us-central2/attributes/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}_fasttext/",
        "experiment": EXPERIMENT,
        "config": CONFIG["annotate"],
    },
)

dedupe_node = Node(
    func=marin.processing.deduplication.dedupe,
    depends_on=[annotate_node1, annotate_node2],
    step_name="dedupe",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_paths": [
            f"gs://marin-us-central2/documents/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}/",
            f"gs://marin-us-central2/documents/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}/",
        ],
        "output_paths": [
            f"gs://marin-us-central2/attributes/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}_dedup/",
            f"gs://marin-us-central2/attributes/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}_dedup/",
        ],
    },
)

consolidate_config = CONFIG["consolidate"]
consolidate_config["filters"]["dedup"][
    "attribute_path"
] = f"gs://marin-us-central2/attributes/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}_dedup/"
consolidate_config["filters"]["classify"][
    "attribute_path"
] = f"gs://marin-us-central2/attributes/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}_fasttext/"
consolidate_node1 = Node(
    func=marin.processing.classification.consolidate,
    depends_on=[dedupe_node, annotate_node1],
    step_name="consolidate",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_path": f"gs://marin-us-central2/documents/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}/",
        "output_path": f"gs://marin-us-central2/attributes/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}_consolidated/",
        "config": consolidate_config,
    },
)

consolidate_config = CONFIG["consolidate"]
consolidate_config["filters"]["dedup"][
    "attribute_path"
] = f"gs://marin-us-central2/attributes/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}_dedup/"
consolidate_config["filters"]["classify"][
    "attribute_path"
] = f"gs://marin-us-central2/attributes/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}_fasttext/"
consolidate_node2 = Node(
    func=marin.processing.classification.consolidate,
    depends_on=[dedupe_node, annotate_node2],
    step_name="consolidate",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_path": f"gs://marin-us-central2/documents/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}/",
        "output_path": f"gs://marin-us-central2/attributes/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}_consolidated/",
        "config": consolidate_config,
    },
)

tokenizer_node1 = Node(
    func=marin.processing.tokenize,
    depends_on=[consolidate_node1],
    step_name="tokenize",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_paths": [f"gs://marin-us-central2/documents/{DATASET1}/{DATASET1_VERSION}/{EXPERIMENT}_consolidated/"],
        "cache_dir": CONFIG["tokenize"]["cache_dir"],
        "tokenizer": CONFIG["tokenize"]["tokenizer"],
        "dataset": f"{EXPERIMENT}_{DATASET1}",
    },
)

tokenizer_node2 = Node(
    func=marin.processing.tokenize,
    depends_on=[consolidate_node2],
    step_name="tokenize",
    experiment_name=EXPERIMENT,
    func_kwargs={
        "input_paths": [f"gs://marin-us-central2/documents/{DATASET2}/{DATASET2_VERSION}/{EXPERIMENT}_consolidated/"],
        "cache_dir": CONFIG["tokenize"]["cache_dir"],
        "tokenizer": CONFIG["tokenize"]["tokenizer"],
        "dataset": f"{EXPERIMENT}_{DATASET2}",
    },
)
