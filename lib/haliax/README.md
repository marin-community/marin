<!--haliax-intro-start-->
# Haliax

<a href="https://github.com/stanford-crfm/haliax/actions?query=branch%3Amain++">
    <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/stanford-crfm/haliax/run_tests.yaml?branch=main">
</a>
<a href="https://haliax.readthedocs.io/en/latest/?badge=latest">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/haliax/badge/?version=latest">
</a>
<a href="">
<img alt="License" src="https://img.shields.io/github/license/stanford-crfm/haliax?color=blue" />
</a>
<a href="https://pypi.org/project/haliax/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/haliax?color=blue" />
</a>

> *Though you don’t seem to be much for listening, it’s best to be careful. If you managed to catch hold of even just a piece of my name, you’d have all manner of power over me.*<br/>
> — Patrick Rothfuss, *The Name of the Wind*

Haliax is a [JAX](https://github.com/google/jax) library for building neural networks with named tensors, in the tradition of Alexander Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor).
Named tensors improve the **legibility** and **compositionality** of tensor programs by using named axes instead of positional indices
as typically used in NumPy, PyTorch, etc.

Despite the focus on legibility, Haliax
is also **fast**, typically about as fast as "pure" JAX code.
Haliax is also built to be **scalable**: it
can support [Fully-Sharded Data Parallelism (FSDP)](https://engineering.fb.com/2021/07/15/open-source/fsdp/) and Tensor Parallelism with [just a few lines of code](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz). Haliax powers [Levanter](https://github.com/stanford-crfm/levanter),
our companion library for training large language models and other foundation models, with scale proven up to 70B parameters
and up to TPU v4-2048.

## Example: Attention

Here's a minimal attention module implementation in Haliax. For a more detailed introduction,
please see the [Haliax tutorial](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC).
(We use the excellent [Equinox](https://github.com/patrick-kidger/equinox) library for its module system and tree transformations.)

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn

Pos = hax.Axis("position", 1024)  # sequence length
KPos = Pos.alias("key_position")
Head = hax.Axis("head", 8)  # number of attention heads
Key = hax.Axis("key", 64)  # key size
Embed = hax.Axis("embed", 512)  # embedding size

# alternatively:
#Pos, KPos, Head, Key, Embed = hax.make_axes(pos=1024, key_pos=1024, head=8, key=64, embed=512)


def attention_scores(Key, KPos, query, key, mask):
    # how similar is each query to each key
    scores = hax.dot(query, key, axis=Key) / jnp.sqrt(Key.size)

    if mask is not None:
        scores -= 1E9 * (1.0 - mask)

    # convert to probabilities
    scores = haliax.nn.softmax(scores, KPos)
    return scores


def attention(Key, KPos, query, key, value, mask):
    scores = attention_scores(Key, KPos, query, key, mask)
    answers = hax.dot(scores, value, axis=KPos)

    return answers


# Causal Mask means that if pos >= key_pos, then pos can attend to key_pos
causal_mask = hax.arange(Pos).broadcast_axis(KPos) >= hax.arange(KPos)


class Attention(eqx.Module):
    proj_q: hnn.Linear  # [Embed] -> [Head, Key]
    proj_k: hnn.Linear  # [Embed] -> [Head, Key]
    proj_v: hnn.Linear  # [Embed] -> [Head, Key]
    proj_answer: hnn.Linear  # output projection from [Head, Key] -> [Embed]

    @staticmethod
    def init(Embed, Head, Key, *, key):
        k_q, k_k, k_v, k_ans = jax.random.split(key, 4)
        proj_q = hnn.Linear.init(In=Embed, Out=(Head, Key), key=k_q)
        proj_k = hnn.Linear.init(In=Embed, Out=(Head, Key), key=k_k)
        proj_v = hnn.Linear.init(In=Embed, Out=(Head, Key), key=k_v)
        proj_answer = hnn.Linear.init(In=(Head, Key), Out=Embed, key=k_ans)
        return Attention(proj_q, proj_k, proj_v, proj_answer)

    def __call__(self, x, mask=None):
        q = self.proj_q(x)
        # Rename "position" to "key_position" for self attention
        k = self.proj_k(x).rename({"position": "key_position"})
        v = self.proj_v(x).rename({"position": "key_position"})

        answers = attention(Key, KPos, q, k, v, causal_mask)

        x = self.proj_answer(answers)
        return x
```

Haliax was created by [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/)'s research engineering team.
You can find us in the #levanter channel on the unofficial [Jax LLM Discord](https://discord.gg/FkRGNX3ND).

<!--haliax-intro-end-->

## Documentation

### Tutorials

These are some tutorials to get you started with Haliax. They are available as Colab notebooks:

<!--haliax-tutorials-start-->

* [Introduction to Haliax with Transformers](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC)
* [Distributed Training in Haliax](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz) (including FSDP)
* [Tensor Parallelism in Haliax](https://colab.research.google.com/drive/18_BrtDpe1lu89M4T6fKzda8DdSLtFJhi)
* [Mixed Precision with `jmp`](https://colab.research.google.com/drive/1_4cikwt-UhSH7yRzNRK8ze9msM9r2mEl?usp=sharing) (This one is really a tutorial for [jmp](https://github.com/deepmind/jmp) but it's how to use it with Haliax...)

<!--haliax-tutorials-end-->
### API Reference

Haliax's API documentation is available at [haliax.readthedocs.io](https://haliax.readthedocs.io/en/latest/).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
We also have a list of [good first issues](https://github.com/stanford-crfm/haliax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
to help you get started. (If those don't appeal, don't hesitate to reach out to us on Discord!)

## License

Haliax is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
