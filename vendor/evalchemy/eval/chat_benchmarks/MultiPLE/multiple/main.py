import argparse
from pathlib import Path
from multiprocessing import cpu_count
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from containerized_eval import eval_string_script
from threading import Lock
import time
from typing import Optional
import itertools
import gzip

# Get working directory
WORKING_DIR = Path(__file__).parent.parent

# program: str => Result
CACHE = dict()
CACHE_LOCK = Lock()


def cache_get(program: str) -> Optional[dict]:
    if program in CACHE:
        result = CACHE[program]
        return result
    else:
        return None


def cache_set(program: str, result: dict):
    if program in CACHE:
        print("Setting already-existing cache")
    CACHE[program] = result


def cached_eval_script(problem, index) -> dict:
    program = problem["prompt"] + problem["completions"][index] + "\n" + problem["tests"]
    CACHE_LOCK.acquire(True)
    cached = cache_get(program)
    if cached is not None:
        CACHE_LOCK.release()
        return cached
    else:
        result_yaml = dict()
        cache_set(program, result_yaml)
        CACHE_LOCK.release()
        result_dict = eval_string_script(problem["language"], program)
        for k in result_dict.keys():
            result_yaml[k] = result_dict[k]
            result_yaml["timestamp"] = int(time.time())
        return result_yaml


def get_test_results_json_path(output_dir: Path, problem_json_path: Path, input_dir: Path) -> Path:
    suffixes = ".results.json.gz" if problem_json_path.suffix == ".gz" else ".results.json"
    problem_name = problem_json_path.name[: -(len(".json.gz") if problem_json_path.suffix == ".gz" else len(".json"))]
    if input_dir:
        return output_dir / (problem_json_path.relative_to(input_dir).parent / (problem_name + suffixes))
    return output_dir / (problem_name + suffixes)


def open_json(fpath: Path, mode: str):
    return gzip.open(fpath, mode + "t") if fpath.suffix == ".gz" else open(fpath, mode)


def evaluate_problem(output_dir: Path, problem_json_path: Path, max_workers: int, input_dir: Path = None):
    with open_json(problem_json_path, "r") as f:
        problem = json.load(f)

    # Do not create a blank .results.json file if there are no completions ready.
    if len(problem["completions"]) == 0:
        return

    test_results_path = get_test_results_json_path(output_dir, problem_json_path, input_dir)

    test_results_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)

    if not test_results_path.exists():
        # Copy over all metadata from the completions file.
        test_results = problem.copy()
        del test_results["completions"]
        test_results["results"] = []
    else:
        with open_json(test_results_path, "r") as f:
            test_results = json.load(f)

    num_problems = len(problem["completions"])

    if len(test_results["results"]) == num_problems:
        return
    elif len(test_results["results"]) > num_problems:
        print(f"ERROR more results than completions for {problem_json_path}")
        return

    min_problem = len(test_results["results"])

    # In case we have previously computed results, warm the cache with them
    for already_computed in test_results["results"]:
        CACHE[already_computed["program"]] = already_computed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for j in executor.map(lambda index: cached_eval_script(problem, index), range(min_problem, num_problems)):
            test_results["results"].append(j)
            with open_json(test_results_path, "w") as f:
                f.write(json.dumps(test_results, indent=2))


def main():
    args = argparse.ArgumentParser()

    args.add_argument("--output-dir", type=Path, help="Directory to store results in. Ignored when using a --job-file")
    args.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of workers to use",
    )
    args.add_argument(
        "--job-file",
        type=str,
        help="Where the files come from",
    )
    args.add_argument("--job-file-line", type=int, help="The line on the file")

    args.add_argument("--file", type=str, help="The file to evaluate")
    args.add_argument("--dir", type=str, help="The directory to evaluate")
    args.add_argument(
        "--recursive",
        action="store_true",
        help="Read all files under each directory, recursively. Only works with --dir.",
    )
    args.add_argument(
        "--testing",
        action="store_true",
        help="Testing mode: expecting first completion to OK and second one to have some error. Note: clears the output directory!",
    )

    args = args.parse_args()

    if args.testing:
        for p in args.output_dir.iterdir():
            p.unlink()

    if not args.max_workers:
        args.max_workers = cpu_count() - 1 if cpu_count() > 1 else 1

    start_t = time.time()

    if args.file:
        if args.recursive:
            print("--file and --recursive can't work together")
            exit(2)
        if args.output_dir is None:
            print("--file requires --output-dir")
            exit(2)
        if args.job_file is not None or args.job_file_line is not None:
            print("--file and --job-file can't work together")
            exit(2)
        evaluate_problem(args.output_dir, Path(args.file), args.max_workers)
    elif args.dir:
        if args.output_dir is None:
            print("--dir requires --output-dir")
            exit(2)
        if args.job_file is not None or args.job_file_line is not None:
            print("--dir and --job-file can't work together")
            exit(2)
        files = [
            p
            for p in itertools.chain(
                Path(args.dir).glob("**/*.json" if args.recursive else "*.json"),
                Path(args.dir).glob("**/*.json.gz" if args.recursive else "*.json.gz"),
            )
            if not p.name.endswith(".results.json") and not p.name.endswith(".results.json.gz")
        ]
        for file in tqdm(files):
            evaluate_problem(args.output_dir, file, args.max_workers, args.dir)
    elif args.job_file and args.job_file_line is not None:
        if args.output_dir is not None:
            print("--job-file and --output-dir can't work together")
            exit(2)
        with open(args.job_file) as f:
            files = f.readlines()[args.job_file_line - 1].rstrip().split(" ")
        for f in files:
            print(f"Processing {f}")
            p = Path(f)
            evaluate_problem(p.parent, p, args.max_workers)
    else:
        print("Specify either --file, --dir, or both --job-file and --job-file-line")
        exit(2)

    end_t = time.time()
    print(f"Execution took {end_t - start_t} seconds")

    if args.testing:
        failure_exists = False
        for output_file in itertools.chain(
            Path(args.output_dir).glob("*.results.json"), Path(args.output_dir).glob("*.results.json.gz")
        ):
            with open_json(output_file, "r") as f:
                output = json.load(f)
            if len(output["results"]) != 2:
                print(f"WARNING: Expected 2 results in {output_file}, got {len(output['results'])}")
            if output["results"][0]["status"] != "OK":
                print(
                    f"TEST FAILED: {output_file}: Expects first result to be ok, got {output['results'][0]['status']}"
                )
                failure_exists = True
            if not (
                "Error" in output["results"][1]["status"]
                or "Timeout" == output["results"][1]["status"]
                or "Exception" == output["results"][1]["status"]
            ):
                print(
                    f"TEST FAILED: {output_file}: Expects second result to be error, got {output['results'][1]['status']}"
                )
                failure_exists = True

        if failure_exists:
            exit(1)


if __name__ == "__main__":
    main()
