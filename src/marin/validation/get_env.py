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

import os
import subprocess
import sys

import ray


def run_collect_env():
    import requests

    url = "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/collect_env.py"
    filename = "collect_env.py"
    # Download the file using requests
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "w") as f:
            f.write(response.text)
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


@ray.remote(resources={"TPU": 1})
def test_collect_env():
    run_collect_env()


if __name__ == "__main__":
    ray.get(test_collect_env.remote())
