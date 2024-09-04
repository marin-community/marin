import time

import ray

from operations.utils.node import Node


@ray.remote
def source():
    # Simulate some processing
    time.sleep(2)
    print("source_result")


@ray.remote
def child_1(source_result):
    # Simulate some processing
    time.sleep(1)
    print(f"child_1_result based on {source_result}")


@ray.remote
def child_2(source_result):
    # Simulate some processing
    time.sleep(1)
    print(f"child_2_result based on {source_result}")


@ray.remote
def destination(child_results):
    # Simulate some processing
    time.sleep(2)
    print(f"destination_result based on {child_results}")


# Create Node objects
experiment_name = "experiment_3"
source_node = Node(func=source, step_name="source", experiment_name=experiment_name)
child1_node = Node(
    func=child_1, depends_on=[source_node], step_name="child_1", experiment_name=experiment_name, func_args=[1]
)
child2_node = Node(
    func=child_2, depends_on=[source_node], step_name="child_2", experiment_name=experiment_name, func_args=[1]
)
destination_node = Node(
    func=destination,
    depends_on=[child1_node, child2_node],
    step_name="destination",
    experiment_name=experiment_name,
    func_args=[[3, 4]],
)

# Execute the DAG
if __name__ == "__main__":
    ray.init(local_mode=True)
    destination_node.execute()
