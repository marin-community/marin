import ray


@ray.remote(runtime_env={"uv": ["emoji"]})
def f():
    import os

    import emoji

    print(os.getenv("TPU_CI"))
    return emoji.emojize("Python is :thumbs_up:")


# Execute 1000 copies of f across a cluster.

print(ray.get([f.remote() for _ in range(1000)]))
