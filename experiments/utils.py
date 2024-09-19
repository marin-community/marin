import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from typing import Optional, Callable, Any

import graphviz
import ray

import marin


@ray.remote
def execute(fn: Callable | ray.remote_function.RemoteFunction, *args, depends_on, **kwargs):
    """
    Utility function to execute a remote function with dependencies on another functions
    fn: The function to execute. The function can be a ray remote function or a normal function
    depends_on: List of references on which the remote function depends. These references are Ray references
    args, kwargs: List of arguments and key arguments to pass to the remote function

    """
    ray.get(depends_on)

    is_ray_fn = type(fn) is ray.remote_function.RemoteFunction

    name = None
    if is_ray_fn:
        name = f"{fn._function.__module__}.{fn._function.__name__}"
    else:
        name = f"{fn.__module__}.{fn.__name__}"

    # Datetime can probably go into ray logger but I don't wanna touch it for now and custom logger are wierd with ray
    print(
        f"Starting to Execute {name} with {args} and {kwargs} at " f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    start = time.time()

    output = None
    if is_ray_fn:
        output = ray.get(fn.remote(*args, **kwargs))
    else:
        output = fn(*args, **kwargs)

    print(
        f"Finished Executing {name} in {time.time() - start} seconds at"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return output


class Executor:

    def __init__(self, region: str, experiment_prefix: str, **kwargs):
        # We store everything in output_path_args which will be passed to each config to get output path
        # later we can convert output_path_args to a dataclass.
        self.output_path_args = kwargs
        self.output_path_args["region"] = region
        self.output_path_args["experiment_prefix"] = experiment_prefix
        self.dry_run = True
        self.graph = graphviz.Digraph(format='dot')
        self.obj_ref_to_name = {}

    def replace_values_in_dataclass(self, config: dataclass):
        # Iterate over all fields in the dataclass
        for field in fields(config):
            field_value = getattr(config, field.name)  # Get the value of the field

            # If the field is a dataclass, recurse into it
            if is_dataclass(field_value):
                self.replace_values_in_dataclass(field_value)
            elif type(field_value) is ray.ObjectRef:
                print(f"Replacing {field.name}'s value with actual value")
                setattr(config, field.name, ray.get(field))  # Replace the value with "objref"
                self.graph.edge(self.obj_ref_to_name[field], self.current_node)

        return config

    @ray.remote
    def _return_input(self, input: Any):
        # This function just returns the input
        return input
    def  add(self, fn: Callable | ray.remote_function.RemoteFunction, *args, config: dataclass,
             depends_on: list | Optional, run_id: str | Optional = "", **kwargs):
        # If run_id is not provided, generate a new run_id using experiment_prefix and random string

        is_ray_fn = type(fn) is ray.remote_function.RemoteFunction

        name = None
        if is_ray_fn:
            name = f"{fn._function.__module__}.{fn._function.__name__}"
        else:
            name = f"{fn.__module__}.{fn.__name__}"

        self.current_node = name
        if not run_id:
            run_id = f"{self.experiment_prefix}_{marin.utils.random_string()}"
            print(f"No run_id provided for {name}, generating a new run_id and this function will run with: {run_id = }")
        else:
            print(f"Using provided run_id {run_id} for {name}, this function will not run")

        # Add to graph
        self.graph.node(name, label=f"{name} [will_run={not run_id}]")
        # Step 1:
        # every config will have a get_output_path method which gives output path given the run_id and also fill it in itself

        output_path = config.get_output_path(run_id, **self.output_path_args)

        return_ref = None
        if run_id:
            return_ref = self._return_input.remote(output_path)
        else:

            # Now we decide to actually run it
            # Step 2: for every parametere in dataclass check if it's of type ObjectRef, if it is, then replace it with the actual
            # value using ray.get

            self.replace_values_in_dataclass(config)

            # By this point we have everything we need to run the function

            print(
                f"Starting to Execute {name} with {args} and {kwargs} at " f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            start = time.time()

            return_ref = self._run_fn.remote(fn, *args, is_ray_fn, **kwargs)

            print(
                f"Finished Executing {name} in {time.time() - start} seconds at"
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        self.obj_ref_to_name[return_ref] = name
        return output_path

    @ray.remote
    def _run_fn(self, fn: Callable | ray.remote_function.RemoteFunction, *args, is_ray_fn, **kwargs):
        while self.dry_run:
            # busy wait
            time.sleep(30)
        if self.dry_run_terminate:
            return

        if is_ray_fn:
            output = ray.get(fn.remote(*args, **kwargs))
        else:
            output = fn(*args, **kwargs)
    def run(self, ref: ray.ObjectRef):
        print(self.graph.source)
        continue_input = self.get_user_input()
        self.dry_run = False
        if continue_input:
            self.dry_run_terminate = False
        else:
            self.dry_run_terminate = True
        ray.get(ref)

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

