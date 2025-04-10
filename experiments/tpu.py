import ray


@ray.remote(resources={"TPU-v6e-8-head": 1})
def hello_world():
    print("Hello world")


ray.get(hello_world.remote())
