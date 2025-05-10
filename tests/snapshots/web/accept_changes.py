import os
import sys

my_dir = os.path.dirname(os.path.realpath(__file__))
input_path = os.path.join(my_dir, "inputs")  # Directory containing HTML input files
expected_dir = os.path.join(my_dir, "expected")  # Directory containing expected output files
output_path = os.path.join(my_dir, "outputs")  # Directory containing actual output files
diff_dir = os.path.join(my_dir, "diffs")  # Directory containing diff files


def accept_change(test_name):
    """Accept expected output for a specific test."""
    expected_file = os.path.join(expected_dir, f"{test_name}.md")
    output_file = os.path.join(output_path, f"{test_name}.md")

    os.makedirs(expected_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    with open(output_file, "r") as f:
        output = f.read()

    with open(expected_file, "w") as f:
        f.write(output)


if __name__ == "__main__":
    # if we get "--all" as an argument, we accept all changes
    # otherwise, we expect the name of the test(s) to accept
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        for file in os.listdir(output_path):
            test_name = os.path.splitext(file)[0]
            accept_change(test_name)
    else:
        for test_name in sys.argv[1:]:
            accept_change(test_name)
