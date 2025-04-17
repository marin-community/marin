import os
import subprocess
import sys

import ray


@ray.remote(resources={"TPU": 1})
def test_collect_env():
    url = "https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py"
    filename = "collect_env.py"
    # Download the file using wget
    try:
        subprocess.run(["wget", "-O", filename, url], check=True)
    except Exception as e:
        print(f"Failed to download collect_env.py: {e}")
        return
    # Run the script in a subprocess and capture output
    try:
        result = subprocess.run([sys.executable, filename], capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running collect_env.py: {e}")
        print(e.output)
    finally:
        # Clean up the downloaded file
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    ray.get(test_collect_env.remote())
