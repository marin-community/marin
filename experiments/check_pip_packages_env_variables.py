import os
import subprocess

import ray


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


if __name__ == "__main__":
    # Call the remote function and wait for the result
    ray.get(print_installed_packages_and_env.remote())
