import json
import time

import fsspec
import ray


class Node:
    """A class representing a node in a Directed Acyclic Graph (DAG) for executing tasks with dependencies.
    This class assumes that func does not depend on the output of depends_on in explicit way. that is depends_on
    would write their outputs to some shared storage and func would read from that storage.

    Attributes:
        func (ray.remote function): The Ray remote function to execute.
        depends_on (list of Node, optional): A list of Node instances that this node depends on. Execution of this node
                                             will wait for all these dependencies to complete.
        step_name (str): The name of this step in the experiment.
        experiment_name (str): The name of the experiment. Results and metadata for this step are stored in a JSON file
                               named "{experiment_name}.json" in Google Cloud Storage (GCS).
        old_experiment_name (str, optional): The name of an old experiment. If provided, func will not be execute and
                                                it would be assumed that  results from the old experiment will be used.
                                                The user needs to make sure that the old experiment results are available
                                                and the call to execute is proper.
    """

    def __init__(
        self,
        func,
        depends_on=None,
        step_name="",
        experiment_name="",
        old_experiment_name="",
        func_args=None,
        func_kwargs=None,
    ):
        self.func = func
        self.depends_on = depends_on if depends_on else []
        self.step_name = step_name
        self.experiment_name = experiment_name
        self.old_experiment_name = old_experiment_name
        self.args = func_args if func_args else []
        self.kwargs = func_kwargs if func_kwargs else {}

        print(f"Creating node for {self.step_name} {self.args}")

        path = "gs://marin-us-central2/experiments/"
        self.experiment_path = f"{path}{self.experiment_name}.json"
        self.old_experiment_path = f"{path}{self.old_experiment_name}.json"
        self.fs = fsspec.filesystem("gcs")

    @ray.remote
    def execute_func(self):
        logging = {"start_time": time.time()}
        ray.get(self.func.remote(*self.args, **self.kwargs))
        logging["end_time"] = time.time()
        self.write_data(self.step_name, logging)

    def get_data(self):
        data = {}
        if self.fs.exists(self.experiment_path):
            with self.fs.open(self.experiment_path, "r") as f:
                data = json.load(f)
        return data

    def write_data(self, key, value):
        data = self.get_data()
        data[key] = value
        with self.fs.open(self.experiment_path, "w") as f:
            json.dump(data, f)

    def execute(self):

        data = self.get_data()

        if self.step_name in data:
            print(f"{self.step_name} was already executed." f"\nStats for {self.step_name}: {data[self.step_name]}")
            return None

        if self.old_experiment_name:
            if self.fs.exists(self.old_experiment_path):
                with self.fs.open(self.old_experiment_path, "r") as f:
                    old_data = json.load(f)
                if self.step_name in old_data:
                    data_to_write = old_data[self.step_name]
                    data_to_write["using_cached_data"] = True
                    data_to_write["cached_experiment"] = self.old_experiment_name
                    self.write_data(self.step_name, data_to_write)
                    print(
                        f"Skipping {self.step_name} as it exits in {self.old_experiment_name}."
                        f"\nCopying stats from old experiment for {self.step_name}: {data[self.step_name]}"
                    )
                    return None

        waitable_refs = []
        # Execute dependencies first
        for node in self.depends_on:
            node_execute = node.execute()
            if node_execute:
                waitable_refs.append(node_execute)

        # Wait for dependencies to finish
        ray.get(waitable_refs)
        result = self.execute_func.remote(self)

        return result
