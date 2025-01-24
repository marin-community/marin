import random
from experiments.curriculum.curriculum_stages import tokenize_train_validation

BASE_DIR_STACK_PYTHON = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/python"
BASE_DIR_STACK_CPP = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/cpp"
BASE_DIR_DOLMA = "gs://marin-us-central2/raw/dolma/v1.7"

# randomly split stack python parquet files into two seperate groups
stack_file_ids = list(range(144))
random.seed(42)
random.shuffle(stack_file_ids)
stack_file_ids_stage1 = stack_file_ids[0:72]
stack_file_ids_stage2 = stack_file_ids[72:143]
stack_file_ids_validation = stack_file_ids[143:144]

# randomly split dolma c4 json.gz files into two seperate groups
dolma_file_ids = list(range(171))
random.shuffle(dolma_file_ids)
dolma_file_ids_stage1 = dolma_file_ids[0:85]
dolma_file_ids_stage2 = dolma_file_ids[85:170]
dolma_file_ids_validation = dolma_file_ids[170:171]

# randomly split stack cpp parquet files into two seperate groups
stack_cpp_file_ids = list(range(110))
random.shuffle(stack_cpp_file_ids)
stack_cpp_file_ids_stage1 = stack_cpp_file_ids[0:55]
stack_cpp_file_ids_stage2 = stack_cpp_file_ids[55:109]
stack_cpp_file_ids_validation = stack_cpp_file_ids[109:110]

# Stage 1

stack_dedup_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage1],
    validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
    name="stack_dedup_stage1",
    text_key="content"
)

dolma_c4_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_stage1],
    validation_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_validation],
    name="dolma_c4_stage1",
)

stack_dedup_stage2_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage2],
    validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
    name="stack_dedup_stage2",
    text_key="content"
)

dolma_c4_stage2_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_stage2],
    validation_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_validation],
    name="dolma_c4_stage2",
)

stack_cpp_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_stage1],
    validation_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage1",
    text_key="content"
)

stack_cpp_stage2_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_stage2],
    validation_files=[f"{BASE_DIR_STACK_CPP}/data-{id:05d}-of-00110.parquet" for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage2",
    text_key="content"
)

stage_data = {
    "stack_dedup": {
        "stage1": stack_dedup_stage1_tokenized,
        "stage2": stack_dedup_stage2_tokenized,
    },
    "c4": {
        "stage1": dolma_c4_stage1_tokenized,
        "stage2": dolma_c4_stage2_tokenized,
    },
    "stack_cpp": {
        "stage1": stack_cpp_stage1_tokenized,
        "stage2": stack_cpp_stage2_tokenized,
    },
}