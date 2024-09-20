import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from typing import Optional, Callable, Any

import fsspec
import graphviz
import ray

import marin

def return_input(input: Any, config: dataclass):
    # This function just returns the input
    return input

@ray.remote
def _run_fn(fn: Callable | ray.remote_function.RemoteFunction, config: dataclass, *args, **kwargs):

    print(
        f"Starting to Execute {get_func_name(fn)} with {config}, {args} "
        f"and {kwargs} at " f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    start = time.time()
    is_ray_fn = type(fn) is ray.remote_function.RemoteFunction
    if is_ray_fn:
        output = ray.get(fn.remote(config, *args, **kwargs))
    else:
        output = fn(config, *args, **kwargs)

    print(f"Finished Executing {get_func_name(fn)} in {time.time() - start} seconds at"
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_func_name(fn: Callable | ray.remote_function.RemoteFunction):
    if type(fn) is ray.remote_function.RemoteFunction:
        return f"{fn._function.__module__}.{fn._function.__name__}"
    else:
        return f"{fn.__module__}.{fn.__name__}"

class Node:
    def __init__(self, fn: Callable | ray.remote_function.RemoteFunction, *args, config: dataclass, output_path,
                 depends_on = None, **kwargs):
        self.fn = fn
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.func_name = get_func_name(fn)
        self.output_path = output_path

        self.should_run = (fn != return_input)
        self.depends_on = []
        self.executed = False
        if self.should_run:
            # depends_on is a list of references on which this function depends
            self.depends_on = depends_on + self.get_depends_on(config)
        self.ray_ref = None

    def get_depends_on(self, config: dataclass):
        depends_on = []
        for field in fields(config):
            field_value = getattr(config, field.name)
            if type(field_value) is Node:
                depends_on.append(field_value)
            elif is_dataclass(field_value):
                depends_on += self.get_depends_on(field_value)
        return depends_on

    def _populate_config(self, config: dataclass):
        # This function will populate the config with the output path
        # This function will be called recursively
        for field in fields(config):
            field_value = getattr(config, field.name)
            if type(field_value) is Node:
                setattr(config, field.name, field_value.output_path)
            elif is_dataclass(field_value):
                self._populate_config(field_value)
        return config

    def run(self):
        # Make sure that this function is not executed twice
        if self.executed:
            return
        self.executed = True

        for i in self.depends_on:
            i.run()

        for i in self.depends_on:
            ray.get(i.ray_ref)

        self.config = self._populate_config(self.config)

        self.ray_ref = _run_fn.remote(self.fn, config=self.config, *self.args, **self.kwargs)





    def print_subtree(self, level: int):
        print("\t"*level + f"{self.func_name} -> ")
        for i in self.depends_on:
            i.print_subtree(level+1)

class Executor:

    def __init__(self, region: str, experiment_prefix: str, run_id: str = None, **kwargs):
        # We store everything in output_path_args which will be passed to each config to get output path
        # later we can convert output_path_args to a dataclass.
        self.output_path_args = kwargs
        self.output_path_args["region"] = region
        self.output_path_args["experiment_prefix"] = experiment_prefix
        if run_id:
            self.output_path_args["run_id"] = run_id
        else:
            # Todo: Implement random string, run_id should be unique and random
            run_id = "dag"
            self.output_path_args["run_id"] = "dag"
        self.experiment = experiment_prefix + "_" + run_id
        self.output_path_args["experiment"] = self.experiment


    def add(self, fn: Callable | ray.remote_function.RemoteFunction, *args, config: dataclass,
             depends_on: list = None , run_id: str  = None, force_run: bool = False, **kwargs):
        """
        Add a function to the execution graph. This function will not run immediately, it will run only when run is called
        fn: The function to add to the graph
        config: The dataclass object which will be passed to the function
        depends_on: Additional List of references on which this function depends, other dependencies will be automatically taken from config.
        This will probably never be used
        run_id: The run_id for this function, if not provided, self.experiment will be used
        force_run: If True, this function will run even if run_id is provided. This is useful when you want to re run something
        """


        name = get_func_name(fn)
        # self.current_node =

        run = force_run or (run_id is None)
        if not run_id:
            run_id = self.experiment
            print(f"No run_id provided for {name}, this function will run with: {run_id = }")
        elif force_run:
            print(f"Egven with provided run_id {run_id} for {name}, this function will run as force_run is True")
        else:
            print(f"Using provided run_id {run_id} for {name}, this function will not run")

        # Step 1:

        # every config will have a get_output_path method which gives output path given the run_id and also fill it in itself
        output_path = config.get_output_path(run_id, **self.output_path_args)
        depends_on = depends_on or []
        return_ref = None
        if not run:
            return_ref = Node(return_input, output_path, config=config, output_path=output_path, depends_on=[], kwargs={})
        else:
            return_ref = Node(fn, *args, config=config, output_path=output_path, depends_on=depends_on, **kwargs)

        return return_ref

    def run(self, node: Node):
        node.print_subtree(0)
        continue_input = self.get_user_input()
        if continue_input:
            node.run()

    def get_user_input(self):
        user_input = input(
            "Do you want to continue [y/N]? ").strip().lower()  # Remove extra spaces and convert to lowercase
        if user_input == "" or user_input == "n":
            return False  # Treat "Enter" or "n" as No
        elif user_input == "y":
            return True  # Treat "y" as Yes
        else:
            print("Invalid input, please respond with 'y' or 'n'.")
            return self.get_user_input()  # Recursively ask again for valid input

