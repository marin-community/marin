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

import ray


@ray.remote(runtime_env={"env_vars": {"ENV_VAR_NESTED_FUNC": "value"}})
def print_nested_packages_and_env():
    print("Results for nested function:")
    # Use subprocess to run 'pip freeze' and capture the output
    pip_result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)

    # Decode the output of pip freeze and print it
    print("Installed pip packages:")
    print(pip_result.stdout.decode("utf-8"))

    # Print all environment variables
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        print(f"{key}={value}")


# Define the function decorated by ray.remote
@ray.remote
def print_installed_packages_and_env():
    # Use subprocess to run 'pip freeze' and capture the output
    pip_result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)

    # Decode the output of pip freeze and print it
    print("Installed pip packages:")
    print(pip_result.stdout.decode("utf-8"))

    # Print all environment variables
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        print(f"{key}={value}")

    # Call the nested function
    ray.get(print_nested_packages_and_env.remote())


if __name__ == "__main__":
    # Call the remote function and wait for the result
    ray.get(print_installed_packages_and_env.remote())
