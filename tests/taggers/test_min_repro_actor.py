import ray

class ExitingActor:
    def __init__(self):
       raise Exception("error")

    def __call__(self, x):
        return x + 1

ds = ray.data.from_items([1, 2, 3])
ds = ds.map(ExitingActor, concurrency=1, ray_remote_args_fn=lambda: {"max_restarts": -1})
ds.show()